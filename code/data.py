import torch
import pandas as pd


def read_data(root):
    # read entity dict and relation dict
    e_dict = {}
    r_dict = {}
    e_data = pd.read_csv(root + 'entities.dict', header=None, delimiter='\t').values
    r_data = pd.read_csv(root + 'relations.dict', header=None, delimiter='\t').values
    for record in r_data:
        r_dict[record[1]] = record[0]
    for record in e_data:
        e_dict[record[1]] = record[0]

    # read data and map to index
    train_data = pd.read_csv(root + 'train.txt', header=None, delimiter='\t')
    valid_data = pd.read_csv(root + 'valid.txt', header=None, delimiter='\t')
    test_data = pd.read_csv(root + 'test.txt', header=None, delimiter='\t')
    for data in [train_data, valid_data, test_data]:
        for column in range(3):
            if column != 1:
                data[column] = data[column].map(e_dict)
            else:
                data[column] = data[column].map(r_dict)
        data.columns = ['h', 'r', 't']

    # already existing heads or tails (for sampling and evaluation)
    already_ts_dict = {}
    already_hs_dict = {}
    already_ts = train_data.groupby(['h', 'r'])['t'].apply(list).reset_index(name='ts').values
    already_hs = train_data.groupby(['t', 'r'])['h'].apply(list).reset_index(name='hs').values
    for record in already_ts:
        already_ts_dict[(record[0], record[1])] = record[2]
    for record in already_hs:
        already_hs_dict[(record[0], record[1])] = record[2]
    return e_dict, r_dict, train_data, valid_data, test_data, already_ts_dict, already_hs_dict


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 e_dict, 
                 r_dict, 
                 train_data, 
                 already_ts_dict, 
                 already_hs_dict,
                 num_ng):
        super().__init__()
        self.e_dict = e_dict
        self.r_dict = r_dict
        self.data = torch.tensor(train_data.values)
        self.already_ts_dict = already_ts_dict
        self.already_hs_dict = already_hs_dict
        self.num_ng = num_ng
    
    def sampling(self, head, rel, tail):
        already_ts = torch.tensor(self.already_ts_dict[(head.item(), rel.item())])
        already_hs = torch.tensor(self.already_hs_dict[(tail.item(), rel.item())])
        neg_pool_t = torch.ones(len(self.e_dict))
        neg_pool_t[already_ts] = 0
        neg_pool_t = neg_pool_t.nonzero()
        neg_pool_h = torch.ones(len(self.e_dict))
        neg_pool_h[already_hs] = 0
        neg_pool_h = neg_pool_h.nonzero()
        neg_t = neg_pool_t[torch.randint(len(neg_pool_t), (self.num_ng//2,))]
        neg_h = neg_pool_h[torch.randint(len(neg_pool_h), (self.num_ng//2,))]
        return neg_t, neg_h
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        head, rel, tail = self.data[idx]
        neg_t, neg_h = self.sampling(head, rel, tail)
        neg_t = torch.cat([torch.tensor([head, rel]).expand(self.num_ng//2, -1), neg_t], dim=1)
        neg_h = torch.cat([neg_h, torch.tensor([rel, tail]).expand(self.num_ng//2, -1)], dim=1)
        sample = torch.cat([torch.tensor([head, rel, tail]).unsqueeze(0), neg_t, neg_h], dim=0)
        return sample


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 e_dict, 
                 r_dict, 
                 test_data):
        super().__init__()
        self.e_dict = e_dict
        self.r_dict = r_dict
        self.data = torch.tensor(test_data.values)
        self.all_e = torch.arange(len(e_dict)).unsqueeze(-1)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        head, rel, tail = self.data[idx]
        return self.data[idx], torch.cat([torch.tensor([head, rel]).expand(len(self.e_dict), -1), self.all_e], dim=1)
