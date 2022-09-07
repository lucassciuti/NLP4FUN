import pandas as pd
from config import dataset

def read_train_test():
    """
        Read train and test datasets and return then    
    """

    train = pd.read_csv(
        dataset["path"] + "/train.csv",
        names=dataset["cols"]
    )
    print("Train dataset read")
    test = pd.read_csv(
        dataset["path"] + "/test.csv",
        names=dataset["cols"]
    )
    print("Test dataset read")
    return train, test

read_train_test()


