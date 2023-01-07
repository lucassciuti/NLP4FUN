import re
import seaborn as sns
import matplotlib.pyplot as plt


def pre_process(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = text.lower()
    return text


def words_histogram(train_text_series, test_text_series):

    # TODO: Config hist plot

    train_n_words = [len(text.split()) for text in train_text_series]
    test_n_words = [len(text.split()) for text in test_text_series]

    sns.histplot(train_n_words, bins=30)
    sns.histplot(test_n_words, bins=30)
    plt.savefig("./results/exploratory/words_dist.png")

