from typing import Any, Dict, List, Optional, Tuple, Type, Union

import lightning as L
import torch
import torchvision
import torchvision.transforms.functional as TF
from torch import Tensor, nn, optim
from torch.nn import functional as F  # noqa: N812

# It seems to be impossible to avoid mypy errors if using import instead of getattr().
# See https://github.com/python/mypy/issues/8823
try:
    LRScheduler: Any = getattr(optim.lr_scheduler, "LRScheduler")
except AttributeError:
    LRScheduler = getattr(optim.lr_scheduler, "_LRScheduler")

import pytorch_metric_learning.testers as pml_testers
from pytorch_metric_learning import losses as pml_losses
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torchmetrics import MaxMetric, MeanMetric

from iruss.models.backbones import get_model

VAL_EVERY_N_EPOCHS = 20


class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features * 2)
        self.fc2 = nn.Linear(in_features * 2, out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ReidLitModule(L.LightningModule):
    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        freeze_backbone: bool = True,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        lr_scheduler: torch.optim.lr_scheduler = None,
        lr_scheduler_config: Dict[str, Any] = None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)  # , ignore="backbone, head")

        self.backbone = backbone or get_model("resnet50_im1k")
        self.head = head or MLP(self.backbone.out_channels, 256)

        if self.hparams.lr_scheduler_config is None and self.hparams.lr_scheduler is not None:
            self.hparams.lr_scheduler_config = {
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            }

        for param in self.backbone.parameters():
            param.requires_grad = False  # freeze encoder

        self.triplet_loss = pml_losses.TripletMarginLoss(margin=0.1)

        self.tester = pml_testers.GlobalEmbeddingSpaceTester(
            dataloader_num_workers=2,
            use_trunk_output=True,  # uses the output of this model, which already includes the head
            accuracy_calculator=AccuracyCalculator(k=10),  # k="max_bin_count"),
        )

        self.train_loss = MeanMetric()

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.backbone(x))

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        x_embed = self.forward(x)
        loss = self.triplet_loss(x_embed, y)

        return loss, x_embed, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        if self.hparams.freeze_backbone:
            self.backbone.eval()
            assert self.backbone.parameters().__next__().requires_grad is False
        else:
            assert self.backbone.parameters().__next__().requires_grad is True

        loss, _, _ = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    @property
    def epoch(self):
        return self.trainer.current_epoch

    def on_train_epoch_end(self) -> None:
        if self.epoch % VAL_EVERY_N_EPOCHS == 0:
            print(f"Validation at epoch {self.epoch}")

            datamodule = self.trainer.datamodule

            gallery_dataset = datamodule.data_gallery
            query_dataset = datamodule.data_query

            dataset_dict = {"gallery": gallery_dataset, "query": query_dataset}
            splits_to_eval = [("query", ["gallery"])]

            all_accuracies = self.tester.test(
                dataset_dict=dataset_dict,
                epoch=self.epoch,
                trunk_model=self,
                splits_to_eval=splits_to_eval,
            )

    def configure_optimizers(self):
        trainable_params = (
            self.head.parameters() if self.hparams.freeze_backbone else self.parameters()
        )

        optimizer = self.hparams.optimizer(params=trainable_params)
        if self.hparams.lr_scheduler is not None:
            scheduler = self.hparams.lr_scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    **self.hparams.lr_scheduler_config,
                },
            }
        return {"optimizer": optimizer}
