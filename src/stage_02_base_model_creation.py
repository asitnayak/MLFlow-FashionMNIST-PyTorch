import argparse
import os
import shutil
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import logging
import torch
from src.utils.common import read_yaml, create_directories
import random
import warnings


STAGE = "BASE_MODEL_CREATION" ## <<< change stage name 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("[%(asctime)s: %(levelname)s: %(module)s]: %(message)s")

file_handler = logging.FileHandler(os.path.join("logs", "running_logs.log"))
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

warnings.filterwarnings('ignore')


class CNN(nn.Module):
  def __init__(self, in_, out_):
    super(CNN, self).__init__()
    self.conv_pool_01 = nn.Sequential(
        nn.Conv2d(in_channels=in_, out_channels=8, kernel_size=5, stride=1, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.conv_pool_02 = nn.Sequential(
        nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.flatten = nn.Flatten()
    self.FC_01 = nn.Linear(in_features=16*4*4, out_features=128)
    self.FC_02 = nn.Linear(in_features=128, out_features=64)
    self.FC_03 = nn.Linear(in_features=64, out_features=out_)

  def forward(self, x):
    x = self.conv_pool_01(x)
    x = self.conv_pool_02(x)
    x = self.flatten(x)
    x = self.FC_01(x)
    x = F.relu(x)
    x = self.FC_02(x)
    x = F.relu(x)
    x = self.FC_03(x)
    return x


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    model = CNN(1, 10)
    logger.info("base model created")
    
    model_dir = config['artifacts']['model_dir']
    model_name = config['artifacts']['base_model_name']

    model_save_path = os.path.join(model_dir, model_name)
    create_directories([model_dir])

    torch.save(model, model_save_path)
    logger.info(f"base model saved at {model_save_path}")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logger.info("\n********************")
        logger.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logger.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e