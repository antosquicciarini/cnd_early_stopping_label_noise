import torch
import torch.nn as nn
from torch.utils.data import Dataset
import re
from sklearn.datasets import fetch_20newsgroups
import numpy as np


def regroup_dataset(labels):
    """
    Regroups the original 20 newsgroups labels into 7 broader categories.

    Mapping:
    0 -> alt.atheism
    1 -> comp.*
    2 -> misc.forsale
    3 -> rec.*
    4 -> sci.*
    5 -> soc.religion.christian
    6 -> talk.*
    """
    regrouped = labels.copy()
    for idx, val in enumerate(labels):
        if val == 0:
            regrouped[idx] = 0
        elif val in [1, 2, 3, 4, 5]:
            regrouped[idx] = 1
        elif val == 6:
            regrouped[idx] = 2
        elif val in [7, 8, 9, 10]:
            regrouped[idx] = 3
        elif val in [11, 12, 13, 14]:
            regrouped[idx] = 4
        elif val == 15:
            regrouped[idx] = 5
        elif val in [16, 17, 18, 19]:
            regrouped[idx] = 6
    print('Labels regrouped into 7 categories. Shape:', regrouped.shape)
    return regrouped
  


def load_glove_embeddings(file_path, vocab_limit=None):
    vocab = {"<UNK>": 0}
    vectors = [np.random.normal(scale=0.6, size=300).tolist()]  # UNK vector
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if vocab_limit and idx >= vocab_limit:
                break
            parts = line.rstrip().split(' ')
            word = parts[0]
            vector = list(map(float, parts[1:]))
            vocab[word] = len(vectors)
            vectors.append(vector)

    embedding_tensor = torch.tensor(vectors, dtype=torch.float)
    return vocab, embedding_tensor


class NewsDataset(Dataset):
    def __init__(self, dataset_type, data, labels, transform=None, max_length=50):

        self.dataset_type = dataset_type
        self.data = data
        self.labels = labels
        self.targets = self.labels

        # self.class_to_idx = {}
        # self.targets = []
        # self.data = []
        # self.labels = []

        # class_folders = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        # for idx, class_name in enumerate(class_folders):
        #     self.class_to_idx[class_name] = idx
        #     class_path = os.path.join(root_dir, class_name)
            
        #     for file_name in os.listdir(class_path):
        #         file_path = os.path.join(class_path, file_name)
        #         if not os.path.isfile(file_path):
        #             continue
        #         with open(file_path, 'r', encoding='latin1') as f:
        #             text = f.read()
        #         tokens = re.findall(r'\b\w+\b', text.lower())
        #         indices = [vocab.get(tok, 0) for tok in tokens[:max_length]]
        #         self.data.append(indices)
        #         self.labels.append(idx)

        # self.targets = self.labels

        # newsgroups = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'))
        # X_raw, y_raw = newsgroups.data, newsgroups.target

        # # Map 20 classes into 7 superclasses
        # label_map_7 = {
        #     0: 0, 1: 0, 2: 0, 3: 0,     # comp.*
        #     4: 1, 5: 1, 6: 1, 7: 1,     # rec.*
        #     8: 2, 9: 2, 10: 2,          # sci.*
        #     11: 3, 12: 3, 13: 3,        # talk.*
        #     14: 4,                      # misc.forsale
        #     15: 5,                      # soc.religion.christian
        #     16: 6,                      # alt.atheism
        #     17: 3, 18: 3, 19: 6         # talk.religion.misc -> talk, alt.atheism -> alt
        # }

        # for text, label in zip(X_raw, y_raw):
        #     tokens = re.findall(r'\b\w+\b', text.lower())
        #     indices = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens[:max_length]]
        #     self.data.append(indices)
        #     self.labels.append(label_map_7[label])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # indices = self.data[idx]
        # padded = indices[:self.max_length] + [self.vocab["<UNK>"]] * max(0, self.max_length - len(indices))
        # return torch.tensor(padded, dtype=torch.long), self.labels[idx]
        return self.data[idx], self.targets[idx]
        
# # Example usage
# glove_path = "data/glove.6B/glove.6B.100d.txt"  # make sure this file is downloaded
# vocab, weights = load_glove_embeddings(glove_path)

# embedding = nn.Embedding.from_pretrained(weights, freeze=False)
