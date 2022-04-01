import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import warnings


STAGE = "STAGE_NAME" ## <<< change stage name 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("[%(asctime)s: %(levelname)s: %(module)s]: %(message)s")

file_handler = logging.FileHandler(os.path.join("logs", "running_logs.log"))
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

warnings.filterwarnings('ignore')


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    pass


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logger.info("\n********************")
        logger.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logger.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e