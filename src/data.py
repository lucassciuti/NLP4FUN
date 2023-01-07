from torch.utils.data import Dataset
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased",
    do_lower_case=True
)


def encode(sentence):
    tokenized = tokenizer.batch_encode_plus(
        [sentence],
        max_length=512,
        padding='max_length',
        return_tensors='pt'
    )
    return tokenized.input_ids, tokenized.attention_mask


class LoadDataset(Dataset):
    def __init__(self, data):
        self.texts = data['text']
        self.labels = data['target']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids, mask = encode(self.texts[idx])
        return (
            ids, mask,
            len(self.texts[idx]),
            self.labels[idx]
        )
