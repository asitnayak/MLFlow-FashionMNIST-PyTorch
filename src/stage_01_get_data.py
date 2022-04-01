import argparse
import os
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import warnings
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


STAGE = "GET_DATA" ## <<< change stage name 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("[%(asctime)s: %(levelname)s: %(module)s]: %(message)s")

file_handler = logging.FileHandler(os.path.join("logs", "running_logs.log"))
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

warnings.filterwarnings('ignore')


def main(config_path):
    ## read config files
    config = read_yaml(config_path)

    root_data_dir = config['data']['ROOT_DATA_DIR']
    create_directories([root_data_dir])

    logger.info("getting data")
    train_data = datasets.FashionMNIST(
        root = root_data_dir,
        train = True,
        download = True,
        transform = transforms.ToTensor()
                    )

    test_data = datasets.FashionMNIST(
        root = root_data_dir,
        train = False,
        download = True,
        transform = transforms.ToTensor()
                    )
    logger.info(f"data available at {root_data_dir}")

    labels_dict = {j:i for i,j in train_data.class_to_idx.items()}
    labels_dict

    logger.info("Preparing data loader")
    train_data_loader = DataLoader(
        dataset = train_data,
        batch_size=config['params']['BATCH_SIZE'],
        shuffle=True
                )

    test_data_loader = DataLoader(
        dataset = test_data,
        batch_size=config['params']['BATCH_SIZE'],
        shuffle=True
                )

    return train_data_loader, test_data_loader, labels_dict

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