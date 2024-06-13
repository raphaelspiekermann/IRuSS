import os
import pathlib
import re
import warnings

import randomname
from omegaconf import OmegaConf

RUNS_DIR = None
HYDRA_SEPERATOR = "="
TARGET_SEPARATOR = "="
HYDRA_ITEM_SEPERATOR = ","
TARGET_ITEM_SEPERATOR = ","
FORMAT_STR = "{run_id}_{randomname}_{params}"
BLACKLIST_KEYS = [
    "ckpt_path",
    "checkpoint",
]


def set_runs_dir(dir_path):
    print(f"Setting prefix to {dir_path}")
    global RUNS_DIR
    RUNS_DIR = dir_path
    pathlib.Path(RUNS_DIR).mkdir(parents=True, exist_ok=True)
    return dir_path


def process_dir_name(parameter_string):
    assert RUNS_DIR is not None, "runs_dir must be set before using this resolver!"

    # print(f"{parameter_string = }")

    parameter_string = parameter_string.replace("\\,", "__").replace(r"\=", "__")
    processed_items = []
    for item in parameter_string.split(HYDRA_ITEM_SEPERATOR):
        if item == "":
            continue
        # print(f"{item = }")

        if item.count(HYDRA_SEPERATOR) > 1:
            # only use until first occurrence of separator in string
            key, value = (
                item[: item.index(HYDRA_SEPERATOR)],
                item[item.index(HYDRA_SEPERATOR) :],
            )
        else:
            key, value = item.split(HYDRA_SEPERATOR)

        # print("key", key)
        # print("value", value)

        reduced_key = key.split(".")[-1]  # only take the last part of the key

        if reduced_key in BLACKLIST_KEYS:
            continue

        # if value contains a path separator, skip it to avoid creating directories
        os_sep = os.path.sep
        if isinstance(value, str) and value.find(os_sep) != -1:
            warnings.warn(
                f"Value {value} for key {key} contains a path separator! Skipping in directory-name generation..."
            )
            continue

        processed_items.append(TARGET_SEPARATOR.join([reduced_key, value]))

    processed_parameter_string = TARGET_ITEM_SEPERATOR.join(processed_items)

    def find_next_id():
        names_in_runs_dir = pathlib.Path(RUNS_DIR).iterdir()

        ids = []

        for file in names_in_runs_dir:
            if file.is_dir():
                try:
                    match = re.match(r"(\d+)_", file.name)
                    if match:
                        ids.append(int(match.group(1)))
                except ValueError:
                    pass  # ignore non matching directories

        if len(ids) == 0:
            return 0

        return max(ids) + 1

    directory = FORMAT_STR.format(
        run_id=find_next_id(),
        params=processed_parameter_string,
        randomname=randomname.get_name(),
    )

    print(f"Generated Directory: {directory}")

    return directory


def import_config(config_path):
    return OmegaConf.load(config_path)


OmegaConf.register_new_resolver("set_runs_dir", set_runs_dir)
OmegaConf.register_new_resolver("process_dir_name", process_dir_name)
OmegaConf.register_new_resolver("randomname", randomname.get_name)
OmegaConf.register_new_resolver("import_config", import_config)
