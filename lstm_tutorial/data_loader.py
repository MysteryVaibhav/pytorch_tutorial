import numpy as np
import torch
import torch.utils.data
from collections import defaultdict


class DataLoader:
    def __init__(self, params):
        self.params = params

        # Dictionaries for word to index and vice versa
        w2i = defaultdict(lambda: len(w2i))
        t2i = defaultdict(lambda: len(t2i))
        # Adding unk token
        UNK = w2i["<unk>"]

        # Read in the data and store the dicts
        self.train = list(self.read_dataset("data/train.txt", w2i, t2i))
        w2i = defaultdict(lambda: UNK, w2i)
        self.dev = list(self.read_dataset("data/test.txt", w2i, t2i))
        self.t2i = t2i
        self.w2i = w2i
        self.nwords = len(w2i)
        self.ntags = len(t2i)

        # Setting pin memory and number of workers
        kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

        # Creating data loaders
        dataset_train = ClassificationDataSet(self.train)
        self.train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=params.batch_size,
                                                             collate_fn=dataset_train.collate, shuffle=True, **kwargs)

        dataset_dev = ClassificationDataSet(self.dev)
        self.dev_data_loader = torch.utils.data.DataLoader(dataset_dev, batch_size=params.batch_size,
                                                           collate_fn=dataset_dev.collate, shuffle=False, **kwargs)

    @staticmethod
    def read_dataset(filename, w2i, t2i):
        with open(filename, "r") as f:
            for line in f:
                tag, words = line.lower().strip().split(" ||| ")
                yield ([w2i[x] for x in words.split(" ")], t2i[tag])


class ClassificationDataSet(torch.utils.data.TensorDataset):
    def __init__(self, data):
        super(ClassificationDataSet, self).__init__()
        # data is a list of tuples (sent, label)
        self.sents = [x[0] for x in data]
        self.labels = [x[1] for x in data]
        self.num_of_samples = len(self.sents)

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        return self.sents[idx], len(self.sents[idx]), self.labels[idx]

    @staticmethod
    def collate(batch):
        sents = np.array([x[0] for x in batch])
        sent_lens = np.array([x[1] for x in batch])
        labels = np.array([x[2] for x in batch])

        # List of indices according to decreasing order of sentence lengths
        sorted_input_seq_len = np.flipud(np.argsort(sent_lens))
        # Sorting the elements od the batch in decreasing length order
        input_lens = sent_lens[sorted_input_seq_len]
        sents = sents[sorted_input_seq_len]
        labels = labels[sorted_input_seq_len]

        # Creating padded sentences
        sent_max_len = input_lens[0]
        padded_sents = np.zeros((len(batch), sent_max_len))
        for i, sent in enumerate(sents):
            padded_sents[i, :len(sent)] = sent

        return padded_sents, input_lens, labels