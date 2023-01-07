import torch.nn as nn
import torch
from tqdm import tqdm
from transformers import BertModel, BertConfig, AdamW
from transformers import logging
logging.set_verbosity_error()


class BERTNet(nn.Module):
    def __init__(self, dropout):

        super().__init__()

        self.bert = BertModel.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)
        self.relu = nn.ReLU()

    def forward(self, text, mask, text_len):
        bert_output = self.bert(
            input_ids=text,
            attention_mask=mask
        )
        dropout_output = self.dropout(bert_output["pooler_output"])
        linear_output = self.linear(dropout_output)
        final_output = self.relu(linear_output)

        return final_output


class LSTMNet(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        num_layers,
        bidirectional,
        dropout
    ):

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional, dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, mask, text_len):
        # embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            text,  # embedded,
            text_len,
            batch_first=True,
            enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)  # if num_layers change, this can break
        output = self.sigmoid(self.fc(hidden))
        return torch.squeeze(output)


class LRScheduler:
    """
        If the val_loss does not decrease for the given number of 'patience' epochs, then
        the learning rate will decrease by given factor
    """

    def __init__(self, optimizer, patience, min_lr=1e-6, factor=0.5):

        """
            new_lr = old_lr * factor
        """

        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


class EarlyStopping:
    """
        Stop the training when the loss does not improve after certain epochs
    """
    def __init__(self, patience, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss <= self.min_delta:
            self.counter = 1
            print(f"Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("Early stopping!")
                self.early_stop = True


def binary_accuracy(preds, y):

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, criterion, optimizer, device):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for idx, (texts, mask, text_lengths, labels) in tqdm(
            enumerate(iterator, start=1), total=len(iterator)
    ):

        # texts_dim = batch_size x max_length_tokenizer

        labels = labels.to(device)
        optimizer.zero_grad()
        predictions = model(
            torch.squeeze(texts).to(device),
            mask.to(device),
            text_lengths.to(device)
        )

        loss = criterion(torch.squeeze(predictions), labels.float())
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        acc = binary_accuracy(predictions, labels)

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator):

    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for texts, mask, text_lengths, labels in iterator:
            predictions = model(torch.squeeze(texts), mask, text_lengths)
            acc = binary_accuracy(predictions, labels)

            epoch_acc += acc.item()

    return epoch_acc / len(iterator)
