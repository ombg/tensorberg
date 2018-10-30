import json
from easydict import EasyDict
import os


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    config.summary_dir = os.path.join(config.work_dir, config.exp_name, "summaries/")
    config.checkpoint_dir = os.path.join(config.work_dir, config.exp_name, "checkpoints/")
    config.checkpoint_dir_restore = os.path.join(config.checkpoint_dir, config.checkpoint_to_restore)
    config.bottleneck_dir = os.path.join(config.work_dir, config.exp_name, "bottlenecks/")
    return config
