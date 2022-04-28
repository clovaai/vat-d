import torch
import logging
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from transformers import BertTokenizer

logging.getLogger().setLevel(logging.INFO)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data(data_path, n_labeled_per_class, unlabeled_per_class=5000, max_seq_len=256, model='bert-base-uncased'):
    """ Read data, split the dataset, and build dataset for dataloaders.

    Arguments:
        data_path {str}           -- Path to your dataset folder: contain a train.csv and test.csv
        n_labeled_per_class {int} -- Number of labeled data per class

    Keyword Arguments:
        unlabeled_per_class {int} -- Number of unlabeled data per class (default: {5000})
        max_seq_len {int}         -- Maximum sequence length (default: {256})
        model {str}               -- Model name (default: {'bert-base-uncased'})

    AGNEWS (# Class : 4)
    label, title, content
 
    Yahoo! (# Class : 10) 
    label, label, content(preprocessed)
   
    IMDB (# Class : 2)
    content, label

    DBPedia (# Class : 14)
    label, title, content
    """

    # Load the tokenizer for bert

    tokenizer = BertTokenizer.from_pretrained(model)

    # a bit of hack

    train_df = pd.read_csv(data_path + 'train.csv', header=None)
    test_df = pd.read_csv(data_path + 'test.csv', header=None)

    train_df = train_df.to_numpy()
    test_df = test_df.to_numpy()

    train_labels = train_df[:, 0] - 1
    train_text = train_df[:, 2]
    del train_df
    test_labels = test_df[:, 0] - 1
    test_text = test_df[:, 2]
    del test_df

    n_labels = max(test_labels) + 1

    if n_labels == 2:
        train_unsup_df = pd.read_csv(data_path + 'train_unsup.csv', header=None)
        train_unsup_df = train_unsup_df.to_numpy()
        train_unsup_text = train_unsup_df[:, 2]

    # Split the labeled training set, unlabeled training set, development set
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(train_labels, n_labeled_per_class,
                                                                         unlabeled_per_class, n_labels)

    unlabeled_text = train_text[train_unlabeled_idxs] if n_labels != 2 else train_unsup_text

    # Build the dataset class for each set
    train_labeled_dataset = loader_labeled(
        train_text[train_labeled_idxs], train_labels[train_labeled_idxs], tokenizer, max_seq_len, n_labels
    )
    train_unlabeled_dataset = loader_unlabeled(
        unlabeled_text, train_unlabeled_idxs, tokenizer, max_seq_len, n_labels,
    )
    val_dataset = loader_labeled(
        train_text[val_idxs], train_labels[val_idxs], tokenizer, max_seq_len, n_labels
    )
    test_dataset = loader_labeled(
        test_text, test_labels, tokenizer, max_seq_len, n_labels
    )

    logging.info(" | Number of Labeled Samples : {} \t Number of Unlabeled Samples : {} "
                 "\t Number of Valid Samples {} \t Number of Test Samples {}".format(len(
        train_labeled_idxs), len(unlabeled_text), len(val_idxs), len(test_labels)))

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, n_labels


def train_val_split(labels, n_labeled_per_class, unlabeled_per_class, n_labels):
    """Split the original training set into labeled training set, unlabeled training set, development set

    Arguments:
        labels {list}             -- List of labeles for original training set
        n_labeled_per_class {int} -- Number of labeled data per class
        unlabeled_per_class {int} -- Number of unlabeled data per class
        n_labels {int}            -- The number of classes

    Returns:
        [list] -- idx for labeled training set, unlabeled training set, development set
    """

    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(n_labels):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)

        if n_labels == 2:
            train_pool = np.concatenate((idxs[:500], idxs[:-2000]))
            train_labeled_idxs.extend(train_pool[:n_labeled_per_class])
            train_unlabeled_idxs = None
            val_idxs.extend(idxs[-2000:])

        else:
            train_pool = np.concatenate((idxs[:500], idxs[5500:-2000]))
            train_labeled_idxs.extend(train_pool[:n_labeled_per_class])
            train_unlabeled_idxs.extend(idxs[500: 500 + unlabeled_per_class])
            val_idxs.extend(idxs[-2000:])

    np.random.shuffle(train_labeled_idxs)
    if n_labels != 2:
        np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


class loader_labeled(Dataset):
    def __init__(self, dataset_text, dataset_label, tokenizer, max_seq_len, n_labels):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.max_seq_len = max_seq_len
        self.n_labels = n_labels
        self.trans_dist = {}

    def __len__(self):
        return len(self.labels)

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len - 2:
            if self.n_labels == 2:
                length = self.max_seq_len - 2
                tokens = tokens[-length:]
            else:
                tokens = tokens[:self.max_seq_len - 2]
        tokens.insert(0, '[CLS]')
        tokens.append('[SEP]')
        length = len(tokens)
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding

        attention_mask = [1] * length
        attention_mask = attention_mask + padding
        token_type_ids = [0] * self.max_seq_len

        inputs = {}
        inputs['input_ids'] = torch.tensor(encode_result)
        inputs['attention_mask'] = torch.tensor(attention_mask)
        inputs['token_type_ids'] = torch.tensor(token_type_ids)

        return inputs

    def __getitem__(self, idx):
        text = self.text[idx]
        input_label = self.get_tokenized(text)

        return input_label, self.labels[idx]


class loader_unlabeled(Dataset):
    def __init__(self, dataset_text, unlabeled_idxs, tokenizer, max_seq_len, n_labels):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.ids = unlabeled_idxs
        self.max_seq_len = max_seq_len
        self.n_labels = n_labels

    def __len__(self):
        return len(self.text)

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len - 2:
            if self.n_labels == 2:
                length = self.max_seq_len - 2
                tokens = tokens[-length:]
            else:
                tokens = tokens[:self.max_seq_len - 2]
        tokens.insert(0, '[CLS]')
        tokens.append('[SEP]')
        length = len(tokens)
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding

        attention_mask = [1] * length
        attention_mask = attention_mask + padding
        token_type_ids = [0] * self.max_seq_len

        inputs = {}
        inputs['input_ids'] = torch.tensor(encode_result)
        inputs['attention_mask'] = torch.tensor(attention_mask)
        inputs['token_type_ids'] = torch.tensor(token_type_ids)

        return inputs

    def __getitem__(self, idx):

        text = self.text[idx]
        input_unlabel = self.get_tokenized(text)

        return input_unlabel
