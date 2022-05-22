import numpy as np 
import random
import os
import torch
import tqdm


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


def get_rank(pos, X, pred, already_ts_dict, e_dict):
    try:
        already_ts = already_ts_dict[(pos[0, 0].item(), pos[0, 1].item())]
    except:
        already_ts = None
    ranking = torch.argsort(pred, descending=True)
    rank = (ranking == pos[0, 2]).nonzero().item() + 1
    ranking_better = ranking[:rank - 1]
    if already_ts != None:
        for t in already_ts:
            if (ranking_better == t).sum() == 1:
                rank -= 1
    return rank


def evaluate(test_dataloader, already_ts_dict, e_dict, emb_model, dis, device):
    r = []
    rr = []
    h1 = []
    h3 = []
    h10 = []
    with torch.no_grad():
        for pos, X in tqdm.tqdm(test_dataloader):
            X = X.to(device).squeeze(0)
            h_emb, r_emb, t_emb = emb_model(X)
            pred, _ = dis(h_emb, r_emb, t_emb)
            rank = get_rank(pos, X, pred, already_ts_dict, e_dict)
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
    r = int(sum(r)/len(r))
    rr = round(sum(rr)/len(rr), 3)
    h1 = round(sum(h1)/len(h1), 3)
    h3 = round(sum(h3)/len(h3), 3)
    h10 = round(sum(h10)/len(h10), 3)
    print(r)
    print(rr)
    print(h1)
    print(h3)
    print(h10)
    return r, rr, h1, h3, h10


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
