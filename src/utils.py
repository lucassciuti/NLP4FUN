import pandas as pd


def read_train_test():
    """
        Read train and test datasets and return then    
    """
    
    cols = ["score", "title", "review"]

    train = pd.read_csv("../data/amazon_review_full_csv/train.csv", names=cols)
    test = pd.read_csv("../data/amazon_review_full_csv/test.csv", names=cols)

    return train, test


