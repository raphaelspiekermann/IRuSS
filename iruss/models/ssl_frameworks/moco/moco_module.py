from copy import copy, deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import lightning as L
import matplotlib as mpl
import torch
import torchvision
import torchvision.transforms.functional as TF
from lightning.pytorch.accelerators import AcceleratorRegistry
from lightning.pytorch.callbacks import EarlyStopping
from PIL.Image import Image as PILImage
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn, optim
from torch.nn import functional as F  # noqa: N812

from iruss.models.components.fm_reconstructor import FM_Reconstructor
from iruss.models.ssl_frameworks.moco import utils as moco_utils

# It seems to be impossible to avoid mypy errors if using import instead of getattr().
# See https://github.com/python/mypy/issues/8823
try:
    LRScheduler: Any = getattr(optim.lr_scheduler, "LRScheduler")
except AttributeError:
    LRScheduler = getattr(optim.lr_scheduler, "_LRScheduler")


def precision_at_k(output, target, top_k=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def nearest_neighbor_accuracy_at_k(query: torch.Tensor, target: List[Any], top_k=5) -> float:
    """Calculates the accuracy of the nearest neighbor prediction. The query is compared against
    the target list, which contains the ground truth labels for the nearest neighbor search. The
    accuracy is calculated by comparing the prediction of the query against the ground truth labels
    of the nearest neighbors.

    Args:
        query: A tensor containing the query representations.
        target: A list containing the ground truth labels for the nearest neighbor search.
        top_k: The number of nearest neighbors to consider.

    Returns:
        The accuracy of the nearest neighbor prediction.
    """

    assert query.shape[0] == len(
        target
    ), f"Query and target must have same length, got {query.shape[0]} and {len(target)}"

    if not isinstance(target, list):
        raise TypeError("Target must be a list of labels.")

    with torch.no_grad():
        dist = torch.mm(query, query.T)
        _, indices = torch.topk(dist, k=top_k + 1, dim=1)

        n_correct = 0
        for _, idx in enumerate(indices):
            nn_preds = [target[j] for j in idx]
            if all(pred == nn_preds[0] for pred in nn_preds[1:]):
                # TODO: multiclass: top5 acc wie bei torchmetrics (einer von topk hits reicht für treffer)
                # TODO: multilabel: 1) für nächsten nachbarn (k=1) multi-label-macro accuracy von torchmetrics
                n_correct += 1

        return n_correct / len(target)


class RepresentationQueue(nn.Module):
    """The queue is implemented as list of representations and a pointer to the location where the
    next batch of representations will be overwritten."""

    def __init__(self, representation_size: int, queue_size: int):
        super().__init__()

        self.representations: Tensor
        self.register_buffer("representations", torch.randn(representation_size, queue_size))
        self.representations = nn.functional.normalize(self.representations, dim=0)

        self.pointer: Tensor
        self.register_buffer("pointer", torch.zeros([], dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, x: Tensor) -> None:
        """Replaces representations in the queue, starting at the current queue pointer, and
        advances the pointer.

        Args:
            x: A mini-batch of representations. The queue size has to be a multiple of the total number of
                representations across all devices.
        """
        # Gather representations from all GPUs into a [batch_size * world_size, num_features] tensor, in case of
        # distributed training.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            x = moco_utils.concatenate_all(x)

        queue_size = self.representations.shape[1]
        batch_size = x.shape[0]
        if queue_size % batch_size != 0:
            raise ValueError(
                f"Queue size ({queue_size}) is not a multiple of the batch size ({batch_size})."
            )

        end = self.pointer + batch_size
        self.representations[:, int(self.pointer) : int(end)] = x.T
        self.pointer = end % queue_size


class LocalMocoLitModule(L.LightningModule):
    def __init__(
        self,
        ssl_model: nn.Module,
        fm_reconstructor: Optional[FM_Reconstructor] = None,
        blocking_masks: bool = True,
        passing_masks: bool = True,
        noise_masks: bool = False,
        linear_eval: Optional[dict] = None,
        lin_eval_metric_lookup: Optional[List] = [
            "linear_eval_acc",
            "linear_eval_MIoU",
        ],
        lin_eval_period: int = 30,
        num_negatives: int = 65536,
        local_loss_weight=0.5,
        encoder_momentum: float = 0.999,
        temperature: float = 0.07,
        exclude_bn_bias: bool = False,
        optimizer: Type[optim.Optimizer] = optim.SGD,
        optimizer_params: Optional[Dict[str, Any]] = None,
        lr_scheduler: Type[LRScheduler] = optim.lr_scheduler.CosineAnnealingLR,
        lr_scheduler_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        # defaults
        self.local_loss_multiplier = 10  # TODO: Maybe make this a parameter

        self.linear_eval = linear_eval
        self.lin_eval_metric_lookup = lin_eval_metric_lookup
        self.lin_eval_period = lin_eval_period

        _representation_size = ssl_model.representation_size
        if fm_reconstructor is not None:
            assert fm_reconstructor.num_channels == _representation_size
        self.fm_reconstructor = fm_reconstructor

        self.local_loss_weight = local_loss_weight

        self.blocking_masks = blocking_masks
        self.passing_masks = passing_masks
        self.noise_masks = noise_masks

        self.num_negatives = num_negatives
        self.encoder_momentum = encoder_momentum
        self.temperature = temperature
        self.exclude_bn_bias = exclude_bn_bias
        self.optimizer_class = optimizer
        if optimizer_params is not None:
            self.optimizer_params = optimizer_params
        else:
            self.optimizer_params = {"lr": 0.03, "momentum": 0.9, "weight_decay": 1e-4}
        self.lr_scheduler_class = lr_scheduler
        if lr_scheduler_params is not None:
            self.lr_scheduler_params = lr_scheduler_params
        else:
            self.lr_scheduler_params = {"T_max": 100}

        self.model_q = ssl_model
        # print('model_q', self.model_q) # debug

        self.model_k = deepcopy(ssl_model)
        for param in self.model_k.parameters():
            param.requires_grad = False

        # Two different queues of representations are needed, one for training and one for validation data.
        self.train_queue = RepresentationQueue(_representation_size, num_negatives)
        self.val_queue = RepresentationQueue(_representation_size, num_negatives)
        self.test_queue = RepresentationQueue(_representation_size, num_negatives)

    def forward(self, query_images: Tensor) -> Tensor:
        return self.model_q(query_images)

    @property
    def MAX_IMAGES_TO_LOG(self) -> int:
        return 8

    @property
    def queue(self) -> RepresentationQueue:
        queue_map = {
            "train": self.train_queue,
            "val": self.val_queue,
            "test": self.test_queue,
        }
        return queue_map[self.stage]

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def batch_idx(self) -> int:
        return self._batch_idx

    @property
    def is_logging_step(self) -> bool:
        return self._batch_idx == 0 and self.stage != "test"

    @property
    def is_lin_eval_step(self) -> bool:
        return not (self.current_epoch % self.lin_eval_period)

    @property
    def is_pure_moco(self) -> bool:
        return self.local_loss_weight == 0

    def _log_metric(self, metric_name: str, metric_value: Tensor):
        prog_bar = (self.stage, metric_name) in [
            ("train", "loss"),
            ("val", "acc1"),
            ("val", "acc5"),
            ("val", "nn_acc"),
        ]

        self.log(
            f"{self.stage}/{metric_name}",
            metric_value,
            sync_dist=True,
            prog_bar=prog_bar,
        )

    def _log_images(
        self,
        name: str,
        images: Union[List[PILImage], Tensor],
        cmap: str = "none",
        max_images_to_log: int = 0,
    ):
        pass

    def _process_step(self, batch: Tuple[List[List[Tensor]], List[Any]]) -> Optional[STEP_OUTPUT]:
        # Calculate loss and metrics
        loss = self._calculate_loss(batch)
        self._log_metric("loss", loss)

        if self.stage == "train":
            return {"loss": loss}

    def _setup_step(self, idx, stage):
        self._batch_idx = idx
        self._stage = stage

    def on_train_end(self) -> None:
        pass

    def training_step(
        self, batch: Tuple[List[List[Tensor]], List[Any]], batch_idx: int
    ) -> STEP_OUTPUT:
        self._setup_step(batch_idx, "train")
        # saving best lin eval accuracy in case of shutdown
        if self.logger is not None and hasattr(self, "best_lin_eval_metric_lookup"):
            for metric in self.best_lin_eval_metric_lookup:
                self.logger.experiment.add_scalar(
                    metric,
                    self.best_lin_eval_metric_lookup[metric],
                    self.global_step,
                )
        if self.is_logging_step:
            for param_group in self.optimizers().param_groups:
                self.log("train/lr", param_group["lr"], sync_dist=True, prog_bar=False)

        self._momentum_update_key_encoder()  # EMA update of the key encoder
        return self._process_step(batch)

    def validation_step(
        self, batch: Tuple[List[List[Tensor]], List[Any]], batch_idx: int
    ) -> Optional[STEP_OUTPUT]:
        self._setup_step(batch_idx, "val")
        return self._process_step(batch)

    def on_validation_epoch_end(self):
        pass

    def test_step(
        self, batch: Tuple[List[List[Tensor]], List[Any]], batch_idx: int
    ) -> Optional[STEP_OUTPUT]:
        self._setup_step(batch_idx, "test")
        return self._process_step(batch)

    def configure_optimizers(
        self,
    ) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler._LRScheduler]]:
        """Constructs the optimizer and learning rate scheduler based on ``self.optimizer_params``
        and ``self.lr_scheduler_params``.

        If weight decay is specified, it will be applied only to convolutional layer weights.
        """
        if (
            ("weight_decay" in self.optimizer_params)
            and (self.optimizer_params["weight_decay"] != 0)
            and self.exclude_bn_bias
        ):
            defaults = copy(self.optimizer_params)
            weight_decay = defaults.pop("weight_decay")

            wd_group = []
            nowd_group = []
            for name, tensor in self.named_parameters():
                if not tensor.requires_grad:
                    continue
                if ("bias" in name) or ("bn" in name):
                    nowd_group.append(tensor)
                else:
                    wd_group.append(tensor)

            params = [
                {"params": wd_group, "weight_decay": weight_decay},
                {"params": nowd_group, "weight_decay": 0.0},
            ]
            optimizer = self.optimizer_class(params, **defaults)
        else:
            optimizer = self.optimizer_class(self.parameters(), **self.optimizer_params)
        lr_scheduler = self.lr_scheduler_class(optimizer, **self.lr_scheduler_params)
        return [optimizer], [lr_scheduler]

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """Momentum update of the key encoder."""
        momentum = self.encoder_momentum
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)

    def _init_batch(self, batch):
        q_images: Tensor = batch[0]
        k_images: Tensor = batch[1]
        q_conditionings: List[Optional[Any]] = batch[2]  # Can be a List of None-Values
        k_conditionings: List[Optional[Any]] = batch[3]  # Can be a List of None-Values
        targets: List[Optional[Any]] = batch[4]  # None if ssl-targets are not available

        if all([qc is None for qc in q_conditionings]):
            q_conditionings = None
        else:
            q_conditionings = torch.stack(q_conditionings)
        if all(kc is None for kc in k_conditionings):
            k_conditionings = None
        else:
            k_conditionings = torch.stack(k_conditionings)

        return q_images, k_images, q_conditionings, k_conditionings, targets

    def _calculate_loss(self, batch) -> Tuple[Tensor, Tensor, Tensor]:
        """Calculates the normalized temperature-scaled cross entropy loss from a mini-batch of
        image pairs.

        Args:
            batch: A tuple containing a list for q_images, k_images, q_conditionings, k_conditionings, and the targets.
        """
        q_imgs, k_imgs, q_conds, k_conds, y_hats = self._init_batch(batch)

        self._log_images("q_images_augmented", q_imgs)

        # queries
        q_feature_maps = self.model_q._forward_backbone(q_imgs)

        self._log_images("q_feature_maps", q_feature_maps[0].unsqueeze(1), max_images_to_log=64)
        self._log_images("q_argmax_feature_maps", q_feature_maps, cmap="voc")

        # (B, C, H', W')
        q_local_embed = self.model_q._forward_local_projector(q_feature_maps)
        self._log_images("q_feature_maps_up", q_local_embed[0].unsqueeze(1), max_images_to_log=64)
        self._log_images("q_argmax_local_proj", q_local_embed, cmap="voc")

        # (B, representation_size)
        q_global_embeddings = self.model_q._forward_global_projector(q_feature_maps)

        # keys (stop gradient)
        with torch.no_grad():
            # The keys are shuffled between the GPUs before encoding them, to avoid batch normalization leaking
            # information between the samples. This works only when using the DDP strategy.
            if isinstance(self.trainer.strategy, DDPStrategy):
                k_imgs, original_order = moco_utils.shuffle_batch(k_imgs)

            k_feature_maps = self.model_k._forward_backbone(k_imgs)

            # (B, C, H', W')
            k_local_embed = self.model_k._forward_local_projector(k_feature_maps)
            self._log_images("k_argmax_local_proj", k_local_embed, cmap="voc")
            # (B, representation_size)
            k_global_embeddings = self.model_k._forward_global_projector(k_feature_maps)

            if isinstance(self.trainer.strategy, DDPStrategy):
                k_local_embed = moco_utils.sort_batch(k_local_embed, original_order)
                k_global_embeddings = moco_utils.sort_batch(k_global_embeddings, original_order)

        q = nn.functional.normalize(
            q_global_embeddings, dim=1
        )  # Each embedding is normalized to have a length of 1.
        k = nn.functional.normalize(k_global_embeddings, dim=1)

        # Concatenate logits from the positive pairs (batch_size x 1) and the negative pairs (batch_size x queue_size).
        # (B, 1), B[i] = dot(q[i], k[i]) --> Only the diagonal of the dot product matrix
        pos_logits = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        neg_logits = torch.einsum("nc,ck->nk", [q, self.queue.representations.clone().detach()])
        logits = torch.cat([pos_logits, neg_logits], dim=1) / self.temperature

        # The correct label for every query is 0. Calculate the cross entropy of classifying each query correctly.
        target_idxs = torch.zeros(logits.shape[0], dtype=torch.long).type_as(logits)
        moco_loss = F.cross_entropy(logits, target_idxs.long())
        self._log_metric("moco_loss", moco_loss)

        acc1, acc5 = precision_at_k(logits, target_idxs, top_k=(1, 5))
        self._log_metric("acc1", acc1)
        self._log_metric("acc5", acc5)

        self.queue.dequeue_and_enqueue(k)

        beta = self.local_loss_weight

        # local, must be called (even if local_loss_weight==0) otherwise ddp-strategy throws an error
        local_loss = self._local_loss(q_local_embed, k_local_embed, q_conds, k_conds)

        self._log_metric("local_loss", local_loss)

        return (1 - beta) * moco_loss + beta * local_loss

    def _local_loss(
        self,
        q_local_embed: Tensor,
        k_local_embed: Tensor,
        q_conditionings: Optional[List[Any]] = None,
        k_conditionings: Optional[List[Any]] = None,
    ) -> Tensor:
        self._log_images("q_local_embeddings", q_local_embed[0].unsqueeze(1), max_images_to_log=64)
        self._log_images("k_local_embeddings", k_local_embed[0].unsqueeze(1), max_images_to_log=64)

        # Create conditionings for the FM-Reconstructor
        if q_conditionings is not None:
            q_conditionings_tensor = torch.stack(q_conditionings)
            if q_conditionings_tensor.shape[1] < 3:
                self._log_images("q_masks", q_conditionings_tensor[:, 0, :, :])
            elif q_conditionings_tensor.shape[1] >= 3:
                self._log_images("q_masks", q_conditionings_tensor[:, :3, :, :])

        if k_conditionings is not None:
            k_conditionings_tensor = torch.stack(k_conditionings)
            if k_conditionings_tensor.shape[1] < 3:
                self._log_images("k_masks", k_conditionings_tensor[:, 0, :, :])
            elif k_conditionings_tensor.shape[1] >= 3:
                self._log_images("k_masks", k_conditionings_tensor[:, :3, :, :])

        # Initialize the reversed local embeddings with the query local embeddings
        q_reversed = q_local_embed

        # If available, use the FM-Reconstructor to reconstruct the local features
        if self.fm_reconstructor is not None:
            # use the FM-Reconstructor to reconstruct the local features
            masks = [q_conditionings_tensor] if q_conditionings_tensor is not None else []
            if k_conditionings is not None:
                masks.append(k_conditionings_tensor)

            # handling passing and blocking masks
            if self.passing_masks or self.blocking_masks:
                _, h_q, w_q = TF.get_dimensions(q_local_embed)

                masks_temp = [
                    TF.resize(
                        mask,
                        size=(h_q, w_q),
                        interpolation=TF.InterpolationMode.NEAREST,
                    )
                    for mask in masks
                ]
                masks_temp_tensor = torch.cat(masks_temp, dim=1)
            else:
                masks_temp_tensor = None

            if self.blocking_masks and masks_temp_tensor is not None:
                passing_masks = torch.clone(masks_temp_tensor[:, 0, :, :].detach()).to(
                    masks_temp_tensor.device
                )
                passing_masks[passing_masks == 0] = 1.0
                passing_masks[passing_masks < 1.0] = 0
                self._log_images("q_passing_masks", passing_masks.unsqueeze(1))

                # ~ build signal blocking mask q ~
                # derive the blocking masks from the passing masks, to be compliant with the range of values for the pos embedding masks

                # blocking_masks = torch.clone(masks_temp_tensor[:, 0, :, :].detach()).to(
                #     masks_temp_tensor.device
                # )
                # blocking_masks[blocking_masks > 0] = 1.0
                blocking_masks = torch.clone(passing_masks.detach()).to(passing_masks.device)
                blocking_masks[blocking_masks == 1.0] = 0.9
                blocking_masks[blocking_masks == 0] = 1.0
                blocking_masks[blocking_masks == 0.9] = 0

                self._log_images("q_blocking_masks", blocking_masks.unsqueeze(1))

                q_local_embed = q_local_embed * blocking_masks.unsqueeze(1)  # new

            # adding noise to the mask to possibly stabilize training
            if self.noise_masks and masks_temp_tensor is not None:
                blocking_tf = torchvision.transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4
                )
                # inv_block_mask
                inv_block_mask = torch.clone(masks_temp_tensor[:, 0, :, :].detach()).to(
                    masks_temp_tensor.device
                )
                inv_block_mask[inv_block_mask == 0] = 1.0
                inv_block_mask[inv_block_mask < 1.0] = 0
                # noise_mask
                noise_mask = torch.clone(blocking_masks.detach()).to(blocking_masks.device)
                noise_mask = noise_mask.unsqueeze(1) + blocking_tf(noise_mask.unsqueeze(1))
                noise_mask = noise_mask * inv_block_mask.unsqueeze(1)
                # print(f'noise_mask: {noise_mask.min()} | {noise_mask.max()}') # debug
                q_local_embed = q_local_embed + noise_mask  # new
                self._log_images("q_noise_masks", noise_mask)

            if self.fm_reconstructor.num_masks > 0:
                q_reversed = self.fm_reconstructor(q_local_embed, masks)
            else:
                q_reversed = self.fm_reconstructor(q_local_embed)

        # Log the reversed local embeddings
        self._log_images(
            "q_local_embeddings_reversed",
            q_reversed[0].unsqueeze(1),
            max_images_to_log=64,
        )
        self._log_images("q_argmax_local_rev", q_reversed, cmap="voc")

        # signal passing masks k
        if self.passing_masks and "masks_temp_tensor" in locals():
            k_local_embed = k_local_embed * passing_masks.unsqueeze(1)  # new

        # l2-loss
        return F.mse_loss(q_reversed, k_local_embed) * self.local_loss_multiplier
