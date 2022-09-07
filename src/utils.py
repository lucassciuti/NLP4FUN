import logging
import pandas as pd
from config import dataset


def get_logger():
    
    logging.basicConfig(
        filename = "project.log",
        format = '%(asctime)s %(levelname)s %(message)s',
        datefmt = '%m/%d/%Y %I:%M:%S %p',
        level = logging.INFO
    )

    return logging.getLogger(__name__)


def read_train_test():
    """
        Read train and test datasets and return then    
    """

    train = pd.read_csv(
        dataset["path"] + "/train.csv",
        names=dataset["cols"]
    )
    logger.info("Train dataset read")
    test = pd.read_csv(
        dataset["path"] + "/test.csv",
        names=dataset["cols"]
    )
    logger.info("Test dataset read")
    return train, test


logger = get_logger()
read_train_test()


