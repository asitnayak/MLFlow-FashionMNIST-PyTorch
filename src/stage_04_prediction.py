import argparse
from email.mime import image
import os
from tqdm import tqdm
import pandas as pd
import logging
from src.utils.common import read_yaml, create_directories
from src.stage_01_get_data import main as loader_main
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import warnings
import torch


STAGE = "STAGE_NAME" ## <<< change stage name 

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

    train_data_loader, test_data_loader, labels_dict = loader_main(config_path)
    pred = np.array([])
    target = np.array([])

    prediction_data_dir = config['data']['PRED_DATA_DIR']
    create_directories([prediction_data_dir])
    prediction_data_file_name = config['data']['PRED_DATA_FILE_NAME']
    prediction_data_file_path = os.path.join(prediction_data_dir, prediction_data_file_name)
    model_dir = config['artifacts']['model_dir']
    trained_model_name = config['artifacts']['trained_model_name']
    trained_model_path = os.path.join(model_dir, trained_model_name)

    model = torch.load(trained_model_path)
    logger.info(f"trained model loaded")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(DEVICE)
    logger.info(f"trained model loaded into {DEVICE}")

    with torch.no_grad():
        for batch, data in enumerate(test_data_loader):
            images = data[0].to(DEVICE)
            labels = data[1].to(DEVICE)

            y_pred = model(images)

            pred = np.concatenate((pred, torch.argmax(y_pred, 1).cpu().numpy()))
            target = np.concatenate((target, labels.cpu().numpy()))

    logger.info("prediction for test data finished")
    df = pd.DataFrame({"Actual":target, "Prediction":pred})
    df.to_csv(prediction_data_file_path)
    logger.info(f"saved prediction results into {prediction_data_file_path}")

    cm = confusion_matrix(target, pred)
    print(cm)

    fs = f1_score(target, pred, average=None)
    print(fs)
    logger.info(fs)





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