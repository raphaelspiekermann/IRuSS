import os
import subprocess
import tempfile
import warnings
import zipfile
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from omegaconf import DictConfig

from iruss.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def dump_source_code(cfg) -> None:
    _root = Path(__file__).parent.parent.parent

    def get_git_revision_hash() -> str:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()

    try:
        sha = get_git_revision_hash()
    except Exception as e:
        log.warning(f"Failed to get git SHA! <{e}>")
        sha = "githash_not_found"

    _files = [
        "environment.yaml",
        ".project-root",
        "setup.py",
        "Makefile",
        "requirements.txt",
        "pyproject.toml",
        "README.md",
        ".gitignore",
    ]

    with tempfile.TemporaryDirectory() as tmpdirname:
        # copy files
        for file in _files:
            src_file = _root / file
            dst_file = Path(tmpdirname) / file
            try:
                dst_file.write_text(src_file.read_text())
            except UnicodeDecodeError:
                print(f"Failed to copy {file}")

        # copy source code
        for src_file in (_root / "src").rglob("*.py"):
            dst_file = Path(tmpdirname) / src_file.relative_to(_root)
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            dst_file.write_text(src_file.read_text())

        # copy config files
        for config_file in (_root / "configs").rglob("*.yaml"):
            dst_file = Path(tmpdirname) / config_file.relative_to(_root)
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            dst_file.write_text(config_file.read_text())

        # zip everything
        zip_file = Path(cfg.paths.output_dir) / f"source_code_{sha}.zip"
        with zipfile.ZipFile(zip_file, "w") as zipf:
            for root, _, files in os.walk(tmpdirname):
                for file in files:
                    zipf.write(
                        os.path.join(root, file),
                        os.path.relpath(os.path.join(root, file), tmpdirname),
                    )

        log.info(f"Source code dumped to {zip_file}")


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
