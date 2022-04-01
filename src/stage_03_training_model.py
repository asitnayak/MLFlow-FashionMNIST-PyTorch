import argparse
from calendar import EPOCH
import os
import shutil
from tqdm import tqdm
import logging
import torch
import torch.nn as nn
from src.utils.common import read_yaml, create_directories
from src.stage_01_get_data import main as loader_main
from src.stage_02_base_model_creation import CNN
import random
import warnings


STAGE = "TRAINING_MODEL" ## <<< change stage name 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("[%(asctime)s: %(levelname)s: %(module)s]: %(message)s")

file_handler = logging.FileHandler(os.path.join("logs", "running_logs.log"))
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

warnings.filterwarnings('ignore')


def main(config_path):
    ## read config files
    try:
        config = read_yaml(config_path)

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        model_dir = config['artifacts']['model_dir']
        base_model_name = config['artifacts']['base_model_name']
        trained_model_name = config['artifacts']['trained_model_name']
        base_model_path = os.path.join(model_dir, base_model_name)
        trained_model_path = os.path.join(model_dir, trained_model_name)

        logger.info(f"loading base model from {base_model_path}")
        loaded_model = torch.load(base_model_path)
        logger.info("Base model loaded")
        loaded_model.to(DEVICE)
        logger.info(f"Base model loaded into {DEVICE}")
        criterion = nn.CrossEntropyLoss()   # Loss function
        optimizer = torch.optim.Adam(loaded_model.parameters() , lr=config['params']['LEARNING_RATE'])

        train_data_loader, test_data_loader, labels_dict = loader_main(config_path)
        EPOCH = config['params']['EPOCH']

        for epoch in range(EPOCH):
            with tqdm(train_data_loader) as tqdm_epoch:
                for images, labels in tqdm_epoch:
                    tqdm_epoch.set_description(f"Epoch : {epoch + 1} / EPOCH")

                    # put the images on device
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)

                    # forward pass
                    output = loaded_model(images)
                    loss = criterion(output, labels)

                    # backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    tqdm_epoch.set_postfix(loss = loss.item())

        logger.info("Model training completed successfully")
        torch.save(loaded_model, trained_model_path)
        logger.info(f"trained model saved at {trained_model_path}")

    except Exception as e:
        logger.exception(e)
        print(e)


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