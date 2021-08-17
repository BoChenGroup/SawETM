import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import scipy.io as sio
import torch

class CustomDataset(Dataset):
    def __init__(self, data_file):
        # with open(data_file, 'rb') as f:
        #     data = pickle.load(f)
        # self.train_data = data['doc_bow'].toarray()
        # self.N, self.vocab_size = self.train_data.shape
        # self.voc = data['word2id']
        data = sio.loadmat('mnist_data/mnist')
        self.train_data = np.array(np.ceil(data['train_mnist'] * 5), order='C')  # 0-1
        self.test_data = np.array(np.ceil(data['test_mnist'] * 5), order='C')  # 0-1
        self.N, self.vocab_size = self.train_data.shape

    def __getitem__(self, index):
        topic_data = self.train_data[index, :]
        return np.squeeze(topic_data), 1

    def __len__(self):
        return self.N

def get_loader(topic_data_file, batch_size=200, shuffle=True, num_workers=0):
    dataset = CustomDataset(topic_data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.vocab_size

class CustomDataset_txt(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        data_all = data['data_2000'].toarray()
        # self.train_data = data_all[data['train_id']].astype("int32")
        self.train_data = data_all.astype("int32")
        self.voc = data['voc2000']
        train_label = [data['label'][i] for i in data['train_id']]
        test_label = [data['label'][i] for i in data['test_id']]
        self.N, self.vocab_size = self.train_data.shape

    def __getitem__(self, index):
        topic_data = self.train_data[index, :]
        return np.squeeze(topic_data), 1

    def __len__(self):
        return self.N

def get_loader_txt(topic_data_file, batch_size=200, shuffle=True, num_workers=0):
    dataset = CustomDataset_txt(topic_data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.vocab_size, dataset.voc


class CustomTestDataset_txt(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        data_all = data['data_2000'].toarray()
        self.test_data = data_all[data['test_id']].astype("int32")
        self.voc = data['voc2000']
        train_label = [data['label'][i] for i in data['train_id']]
        test_label = [data['label'][i] for i in data['test_id']]
        self.N, self.vocab_size = self.test_data.shape

    def __getitem__(self, index):
        topic_data = self.test_data[index, :]
        return np.squeeze(topic_data), 1

    def __len__(self):
        return self.N

def get_test_loader_txt(topic_data_file, batch_size=200, shuffle=True, num_workers=0):
    dataset = CustomTestDataset_txt(topic_data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.vocab_size, dataset.voc

def gen_ppl_doc(x, ratio=0.8):
    """
    inputs:
        x: N x V, np array,
        ratio: float or double,
    returns:
        x_1: N x V, np array, the first half docs whose length equals to ratio * doc length,
        x_2: N x V, np array, the second half docs whose length equals to (1 - ratio) * doc length,
    """
    import random
    x_1, x_2 = np.zeros_like(x), np.zeros_like(x)
    # indices_x, indices_y = np.nonzero(x)
    for doc_idx, doc in enumerate(x):
        indices_y = np.nonzero(doc)[0]
        l = []
        for i in range(len(indices_y)):
            value = doc[indices_y[i]]
            for _ in range(int(value)):
                l.append(indices_y[i])
        random.seed(2020)
        random.shuffle(l)
        l_1 = l[:int(len(l) * ratio)]
        l_2 = l[int(len(l) * ratio):]
        for l1_value in l_1:
            x_1[doc_idx][l1_value] += 1
        for l2_value in l_2:
            x_2[doc_idx][l2_value] += 1
    return x_1, x_2

class CustomDataset_txt_ppl(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        data_all = data['data_2000'].toarray()
        self.train_data, self.test_data = gen_ppl_doc(data_all.astype("int32"))
        self.voc = data['voc2000']
        self.N, self.vocab_size = self.train_data.shape

    def __getitem__(self, index):
        return torch.from_numpy(np.squeeze(self.train_data[index])).float(), torch.from_numpy(np.squeeze(self.test_data[index])).float()

    def __len__(self):
        return self.N

def get_loader_txt_ppl(topic_data_file, batch_size=200, shuffle=True, num_workers=0):
    dataset = CustomDataset_txt_ppl(topic_data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.vocab_size, dataset.voc
