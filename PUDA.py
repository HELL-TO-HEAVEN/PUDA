import torch
import argparse
import pandas as pd
import numpy as np 
import pdb
import random
import os
import tqdm


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
        
    all_data = pd.concat([train_data, valid_data, test_data])
    already_ts_dict_all = {}
    already_hs_dict_all = {}
    already_ts_all = all_data.groupby(['h', 'r'])['t'].apply(list).reset_index(name='ts').values
    already_hs_all = all_data.groupby(['t', 'r'])['h'].apply(list).reset_index(name='hs').values
    for record in already_ts_all:
        already_ts_dict_all[(record[0], record[1])] = record[2]
    for record in already_hs_all:
        already_hs_dict_all[(record[0], record[1])] = record[2]
    
    return e_dict, r_dict, train_data, valid_data, test_data, already_ts_dict, already_hs_dict, already_ts_dict_all, already_hs_dict_all


def get_gen_neg(h_emb, r_emb, t_emb, gen, bs, num_ng, emb_dim, device, gen_std, flag):
    z_tail = torch.normal(mean=0, std=gen_std, size=(bs, num_ng//2, emb_dim//8)).to(device)
    z_head = torch.normal(mean=0, std=gen_std, size=(bs, num_ng//2, emb_dim//8)).to(device)
    if flag == 'gen':
        neg_gen_tail = gen(z_tail)
        neg_gen_head = gen(z_head)
        h_emb, r_emb, t_emb = h_emb.detach(), r_emb.detach(), t_emb.detach()
    elif flag == 'dis':
        neg_gen_tail = gen(z_tail).detach()
        neg_gen_head = gen(z_head).detach()
    h_emb_dup = h_emb.view(bs, -1, h_emb.size(-1))[:, 0, :].unsqueeze(1).expand_as(neg_gen_head)
    r_emb_dup = r_emb.view(bs, -1, h_emb.size(-1))[:, 0, :].unsqueeze(1).expand_as(neg_gen_head)
    t_emb_dup = t_emb.view(bs, -1, h_emb.size(-1))[:, 0, :].unsqueeze(1).expand_as(neg_gen_head)
    h_emb = torch.cat([h_emb_dup, neg_gen_head], dim=1).view(-1, emb_dim)
    r_emb = torch.cat([r_emb_dup, r_emb_dup], dim=1).view(-1, emb_dim)
    t_emb = torch.cat([neg_gen_tail, t_emb_dup], dim=1).view(-1, emb_dim)
    return h_emb, r_emb, t_emb


def get_rank(pos, pred, already_dict, flag):
    if flag == 'tail':
        try:
            already = already_dict[(pos[0, 0].item(), pos[0, 1].item())]
        except:
            already = None
    elif flag == 'head':
        try:
            already = already_dict[(pos[0, 2].item(), pos[0, 1].item())]
        except:
            already = None
    else:
        raise ValueError
    ranking = torch.argsort(pred, descending=True)
    if flag == 'tail':
        rank = (ranking == pos[0, 2]).nonzero().item() + 1
    elif flag == 'head':
        rank = (ranking == pos[0, 0]).nonzero().item() + 1
    else:
        raise ValueError
    ranking_better = ranking[:rank - 1]
    if already != None:
        for e in already:
            if (ranking_better == e).sum() == 1:
                rank -= 1
    return rank


def evaluate(dataloader, already_dict, emb_model, dis, device, cfg, flag):
    r = []
    rr = []
    h1 = []
    h3 = []
    h10 = []
    with torch.no_grad():
        if cfg.verbose == 1:
            dataloader = tqdm.tqdm(dataloader)
        for pos, X in dataloader:
            X = X.to(device).squeeze(0)
            h_emb, r_emb, t_emb = emb_model(X)
            pred, _ = dis(h_emb, r_emb, t_emb)
            rank = get_rank(pos, pred, already_dict, flag)
            r.append(rank)
            rr.append(1/rank)
            if rank == 1:
                h1.append(1)
            else:
                h1.append(0)
            if rank <= 3:
                h3.append(1)
            else:
                h3.append(0)
            if rank <= 10:
                h10.append(1)
            else:
                h10.append(0)
    return [r, rr, h1, h3, h10]


def evaluate_wrapper(dataloader_tail, dataloader_head, \
                    already_ts_dict_all, already_hs_dict_all, emb_model, dis, device, cfg, require='tail'):
    tail_results = evaluate(dataloader_tail, already_ts_dict_all, emb_model, dis, device, cfg, flag='tail')
    head_results = evaluate(dataloader_head, already_hs_dict_all, emb_model, dis, device, cfg, flag='head')
    if require == 'head':
        r = int(sum(head_results[0])/len(head_results[0]))
        rr = round(sum(head_results[1])/len(head_results[1]), 3)
        h1 = round(sum(head_results[2])/len(head_results[2]), 3)
        h3 = round(sum(head_results[3])/len(head_results[3]), 3)
        h10 = round(sum(head_results[4])/len(head_results[4]), 3)
    elif require == 'tail':
        r = int(sum(tail_results[0])/len(tail_results[0]))
        rr = round(sum(tail_results[1])/len(tail_results[1]), 3)
        h1 = round(sum(tail_results[2])/len(tail_results[2]), 3)
        h3 = round(sum(tail_results[3])/len(tail_results[3]), 3)
        h10 = round(sum(tail_results[4])/len(tail_results[4]), 3)
    elif require == 'both':
        r = int((sum(tail_results[0]) + sum(head_results[0]))/(len(tail_results[0]) * 2))
        rr = round((sum(tail_results[1]) + sum(head_results[1]))/(len(tail_results[1]) * 2), 3)
        h1 = round((sum(tail_results[2]) + sum(head_results[2]))/(len(tail_results[2]) * 2), 3)
        h3 = round((sum(tail_results[3]) + sum(head_results[3]))/(len(tail_results[3]) * 2), 3)
        h10 = round((sum(tail_results[4]) + sum(head_results[4]))/(len(tail_results[4]) * 2), 3)
    else:
        raise ValueError
    print(r, flush=True)
    print(rr, flush=True)
    print(h1, flush=True)
    print(h3, flush=True)
    print(h10, flush=True)
    return rr

def pur_loss(pred, prior):
    p_above = - torch.nn.functional.logsigmoid(pred[:, 0]).mean()
    p_below = - torch.nn.functional.logsigmoid(-pred[:, 0]).mean()
    u = - torch.nn.functional.logsigmoid(pred[:, 0].unsqueeze(-1) - pred[:, 1:]).mean()
    if u > prior*p_below:
        return prior*p_above - prior*p_below + u
    else:
        return prior*p_above


def my_collate_fn(batch):
    return torch.cat(batch, dim=0)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

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
                test_data,
                flag):
        super().__init__()
        self.e_dict = e_dict
        self.r_dict = r_dict
        self.data = torch.tensor(test_data.values)
        self.all_e = torch.arange(len(e_dict)).unsqueeze(-1)
        self.flag = flag
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        head, rel, tail = self.data[idx]
        if self.flag == 'tail':
            return self.data[idx], torch.cat([torch.tensor([head, rel]).expand(len(self.e_dict), -1), self.all_e], dim=1)
        elif self.flag == 'head':
            return self.data[idx], torch.cat([self.all_e, torch.tensor([rel, tail]).expand(len(self.e_dict), -1)], dim=1)
        else:
            raise ValueError


class LookupEmbedding(torch.nn.Module):
    def __init__(self, e_dict, r_dict, emb_dim, bs):
        super().__init__()
        self.emb_dim = emb_dim
        self.bs = bs
        self.e_dict = e_dict
        self.r_dict = r_dict
        self.emb_e = torch.nn.Embedding(len(e_dict), self.emb_dim)
        self.emb_r = torch.nn.Embedding(len(r_dict), self.emb_dim)
        torch.nn.init.xavier_uniform_(self.emb_e.weight.data)
        torch.nn.init.xavier_uniform_(self.emb_r.weight.data)
    
    def forward(self, x):
        h, r, t = x[:, 0], x[:, 1], x[:, 2]
        h_emb, r_emb, t_emb = self.emb_e(h), self.emb_r(r), self.emb_e(t)
        return h_emb, r_emb, t_emb


class DistMult(torch.nn.Module):
    def forward(self, h_emb, r_emb, t_emb):
        score = (h_emb * r_emb * t_emb).sum(dim=1)
        l2_reg = torch.mean(h_emb ** 2) + torch.mean(t_emb ** 2) + torch.mean(r_emb ** 2)
        return score, l2_reg


class Generator(torch.nn.Module):
    def __init__(self, bs, emb_dim, gen_drop):
        super().__init__()
        self.bs = bs
        self.emb_dim = emb_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim//8, self.emb_dim//8),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(gen_drop),
            torch.nn.Linear(self.emb_dim//8, self.emb_dim),
            torch.nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z.view(-1, self.emb_dim//8)).view(self.bs, -1, self.emb_dim)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # Tunable
    parser.add_argument('--num_ng', default=2, type=int)
    parser.add_argument('--num_ng_gen', default=2, type=int)
    parser.add_argument('--bs', default=512, type=int)
    parser.add_argument('--emb_dim', default=128, type=int)
    parser.add_argument('--lrd', default=0.00001, type=float)
    parser.add_argument('--lrg', default=0.00001, type=float)
    parser.add_argument('--prior', default=0.00001, type=float)
    parser.add_argument('--reg', default=0, type=float)
    parser.add_argument('--gen_drop', default=0.5, type=float)
    parser.add_argument('--gen_std', default=1, type=float)
    # Misc
    parser.add_argument('--data_root', default='./data/FB15k-237', type=str)
    parser.add_argument('--save_path', default='./', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--max_epochs', default=5000, type=int)
    parser.add_argument('--tolerance', default=5, type=int)
    parser.add_argument('--valid_interval', default=100, type=int)
    return parser.parse_args(args)


if __name__ == '__main__':
    # preparation
    cfg = parse_args()
    print('Configurations:', flush=True)
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)
    seed_everything(cfg.seed)
    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')

    # load data
    e_dict, r_dict, train_data, valid_data, test_data, \
        already_ts_dict, already_hs_dict, already_ts_dict_all, already_hs_dict_all = read_data(cfg.data_root + '/')
    train_dataset = TrainDataset(e_dict, r_dict, train_data, already_ts_dict, already_hs_dict, cfg.num_ng)
    valid_dataset_tail = TestDataset(e_dict, r_dict, valid_data, flag='tail')
    valid_dataset_head = TestDataset(e_dict, r_dict, valid_data, flag='head')
    test_dataset_tail = TestDataset(e_dict, r_dict, test_data, flag='tail')
    test_dataset_head = TestDataset(e_dict, r_dict, test_data, flag='head')
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                    batch_size=cfg.bs,
                                                    num_workers=16,
                                                    shuffle=True,
                                                    drop_last=True,
                                                    collate_fn=my_collate_fn)
    valid_dataloader_tail = torch.utils.data.DataLoader(dataset=valid_dataset_tail, 
                                                        batch_size=1,
                                                        num_workers=16,
                                                        shuffle=False,
                                                        drop_last=False)
    valid_dataloader_head = torch.utils.data.DataLoader(dataset=valid_dataset_head, 
                                                        batch_size=1,
                                                        num_workers=16,
                                                        shuffle=False,
                                                        drop_last=False)
    test_dataloader_tail = torch.utils.data.DataLoader(dataset=test_dataset_tail, 
                                                        batch_size=1,
                                                        num_workers=16,
                                                        shuffle=False,
                                                        drop_last=False)
    test_dataloader_head = torch.utils.data.DataLoader(dataset=test_dataset_head, 
                                                        batch_size=1,
                                                        num_workers=16,
                                                        shuffle=False,
                                                        drop_last=False)

    # define model
    emb_model = LookupEmbedding(e_dict, r_dict, cfg.emb_dim, cfg.bs)
    dis = DistMult()
    gen = Generator(cfg.bs, cfg.emb_dim, cfg.gen_drop)
    emb_model = emb_model.to(device)
    dis = dis.to(device)
    gen = gen.to(device)

    # define optimizer
    optim_dis = torch.optim.Adam(list(emb_model.parameters()) + list(dis.parameters()), lr=cfg.lrd)
    optim_gen = torch.optim.Adam(gen.parameters(), lr=cfg.lrg)

    tolerance = cfg.tolerance
    max_mrr = 0
    for epoch in range(cfg.max_epochs):
        print(f'Epoch {epoch + 1}:', flush=True)
        emb_model.train()
        dis.train()
        gen.train()
        avg_loss_dis = []
        avg_loss_gen = []
        
        if cfg.verbose == 1:
            train_dataloader = tqdm.tqdm(train_dataloader)
        for X in train_dataloader:
            X = X.to(device)
            # ==========
            # Train G
            # ==========
            h_emb, r_emb, t_emb = emb_model(X)
            pred_real, _ = dis(h_emb, r_emb, t_emb)
            h_emb_gen, r_emb_gen, t_emb_gen = get_gen_neg(h_emb, r_emb, t_emb, gen, \
                cfg.bs, cfg.num_ng_gen, cfg.emb_dim, device, cfg.gen_std, flag='gen')
            pred_fake, _ = dis(h_emb_gen, r_emb_gen, t_emb_gen)
            pred = torch.cat([pred_real.view(cfg.bs, -1), pred_fake.view(cfg.bs, -1)], dim=-1)
            loss_gen = - pur_loss(pred, cfg.prior)
            optim_gen.zero_grad()
            loss_gen.backward()
            optim_gen.step()
            avg_loss_gen.append(loss_gen.item())
            # ==========
            # Train D
            # ==========
            h_emb, r_emb, t_emb = emb_model(X)
            pred_real, reg_real = dis(h_emb, r_emb, t_emb)
            h_emb_gen, r_emb_gen, t_emb_gen = get_gen_neg(h_emb, r_emb, t_emb, gen, \
                cfg.bs, cfg.num_ng_gen, cfg.emb_dim, device, cfg.gen_std, flag='dis')
            pred_fake, reg_fake = dis(h_emb_gen, r_emb_gen, t_emb_gen)
            pred = torch.cat([pred_real.view(cfg.bs, -1), pred_fake.view(cfg.bs, -1)], dim=-1)
            loss_dis = pur_loss(pred, cfg.prior) + cfg.reg * 0.5 * (reg_real + reg_fake)
            optim_dis.zero_grad()
            loss_dis.backward()
            optim_dis.step()
            avg_loss_dis.append(loss_dis.item())
        print(f'D Loss: {round(sum(avg_loss_dis)/len(avg_loss_dis), 4)}', flush=True)
        print(f'G Loss: {round(sum(avg_loss_gen)/len(avg_loss_gen), 4)}', flush=True)
        
        if (epoch + 1) % cfg.valid_interval == 0:
            emb_model.eval()
            dis.eval()
            rr = evaluate_wrapper(valid_dataloader_tail, valid_dataloader_head, \
                                already_ts_dict_all, already_hs_dict_all, emb_model, dis, device, cfg)
            if rr >= max_mrr:
                max_mrr = rr
                tolerance = cfg.tolerance
            else:
                tolerance -= 1
                
        if (tolerance == 0) or ((epoch + 1) == cfg.max_epochs):
            emb_model.eval()
            dis.eval()
            evaluate_wrapper(test_dataloader_tail, test_dataloader_head, \
                            already_ts_dict_all, already_hs_dict_all, emb_model, dis, device, cfg)
            break
        