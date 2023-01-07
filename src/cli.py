import time
import pandas as pd
import torch.optim
import torch.nn as nn
from data import LoadDataset
from torch.utils.data import DataLoader
from config import config
from model import (
    train,
    evaluate,
    LSTMNet,
    BERTNet,
    LRScheduler,
    EarlyStopping
)
from sklearn.model_selection import train_test_split


def normalize_text(text):
    text = text.lower()  # lowercase
    text = text.replace(r"\#", "")  # replaces hashtags
    text = text.replace(r"http\S+", "URL")  # remove URL addresses
    text = text.replace(r"@", "")
    text = text.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
    text = text.replace(r"\s{2,}", " ")
    return text


def load_data():

    df = pd.read_csv("./data/train.csv")[['text', 'target']]
    df['text'] = df['text'].apply(lambda text: normalize_text(text))

    train_df, valid_df = train_test_split(df, random_state=config['SEED'], train_size=0.7)

    print(f"Training with {len(train_df)} samples")

    train_dataloader = DataLoader(
        LoadDataset(train_df.reset_index(drop=True)),
        batch_size=config['BATCH_SIZE'],
        shuffle=True
    )

    eval_dataloader = DataLoader(
        LoadDataset(valid_df.reset_index(drop=True)),
        batch_size=config['BATCH_SIZE'],
        shuffle=True
    )

    return train_dataloader, eval_dataloader


def run():
    t = time.time()

    # Setting device
    device = torch.device('cpu')

    # Initializing model
    # model = LSTMNet(**config['LSTM_PARAMS']).to(device)
    model = BERTNet(**config['BERT_PARAMS']).to(device)

    # Setting criterion
    criterion = nn.BCEWithLogitsLoss()

    # Setting optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['LEARNING_RATE']
    )

    # Setting scheduler
    lr_scheduler = LRScheduler(optimizer, patience=2)

    # Setting early stopping
    early_stopping = EarlyStopping(patience=3)

    # Loading data
    train_dataloader, eval_dataloader = load_data()

    for epoch in range(config['NUM_EPOCHS']):

        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer, device)
        valid_acc = evaluate(model, eval_dataloader)

        print(
            f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Val. Acc: {valid_acc * 100:.2f}%'
        )

        lr_scheduler(valid_acc)
        early_stopping(valid_acc)
        if early_stopping.early_stop:
            break

    print(f'time:{time.time() - t:.3f} secs')


if __name__ == "__main__":
    run()
