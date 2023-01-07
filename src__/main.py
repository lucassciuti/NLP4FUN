from utils import read_train_test, get_logger
from text_processing import words_histogram
import numpy as np


logger = get_logger()

train, test = read_train_test(logger)

train['fasttext'] = "__label__" + train["score"].astype(str) + " " + train["review"]
print(train.head())

np.savetxt('./data/amazon_review_full.txt', train.fasttext.values, fmt='%s', encoding="utf-8")
