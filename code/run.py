import torch
import argparse
from utils import seed_everything, get_gen_neg, evaluate, my_collate_fn, pur_loss
from data import read_data, TrainDataset, TestDataset
from model import LookupEmbedding, DistMult, Generator

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../data/FB15k-237', type=str)
    parser.add_argument('--save_path', default='./', type=str)
    parser.add_argument('--num_ng', default=8, type=int)
    parser.add_argument('--num_ng_gen', default=8, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--bs', default=512, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--emb_dim', default=1024, type=int)
    parser.add_argument('--lrd', default=0.00001, type=float)
    parser.add_argument('--lrg', default=0.00001, type=float)
    parser.add_argument('--prior', default=0.00001, type=float)
    parser.add_argument('--reg', default=0, type=float)
    parser.add_argument('--max_epochs', default=2600, type=int)
    parser.add_argument('--gen_drop', default=0.5, type=float)
    parser.add_argument('--gen_std', default=1, type=float)
    return parser.parse_args(args)


if __name__ == '__main__':
    # preparation
    cfg = parse_args()
    print('Configurations:')
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}')
    seed_everything(cfg.seed)
    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')

    # load data
    e_dict, r_dict, train_data, valid_data, test_data, already_ts_dict, already_hs_dict = read_data(cfg.data_root + '/')
    train_dataset = TrainDataset(e_dict, r_dict, train_data, already_ts_dict, already_hs_dict, cfg.num_ng)
    test_dataset = TestDataset(e_dict, r_dict, test_data)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                   batch_size=cfg.bs,
                                                   num_workers=16,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   collate_fn=my_collate_fn)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, 
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

    for epoch in range(cfg.max_epochs):
        print(f'Epoch {epoch + 1}:')
        emb_model.train()
        dis.train()
        gen.train()
        avg_loss_dis = []
        avg_loss_gen = []

        for X in train_dataloader:
            X = X.to(device)
            # ==========
            # Train G
            # ==========
            h_emb, r_emb, t_emb = emb_model(X)
            pred_real, _ = dis(h_emb, r_emb, t_emb)
            h_emb_gen, r_emb_gen, t_emb_gen = get_gen_neg(h_emb, r_emb, t_emb, gen, cfg.bs, cfg.num_ng_gen, cfg.emb_dim, device, cfg.gen_std, flag='gen')
            pred_fake, _ = dis(h_emb_gen, r_emb_gen, t_emb_gen)
            pred = torch.cat([pred_real.view(cfg.bs, -1), pred_fake.view(cfg.bs, -1)], dim=-1)
            loss_gen = - pur_loss(pred, cfg.prior)
            optim_gen.zero_grad()
def get_gen_neg(h_emb, r_emb, t_emb, gen, bs, num_ng, emb_dim, device, gen_std, flag):
    z_tail = torch.normal(mean=0, std=gen_std, size=(bs, num_ng//2, emb_dim//8)).to(device)
    z_head = torch.normal(mean=0, std=gen_std, size=(bs, num_ng//2, emb_dim//8)).to(device)
    if flag == 'gen':
        neg_gen_tail = gen(z_tail)
        neg_gen_head = gen(z_head)
        h_emb, r_emb, t_emb = h_emb.detach(), r_emb.detach(), t_emb.detach()
    elif flag == '
            # Train D
            # ==========
            h_emb, r_emb, t_emb = emb_model(X)
            pred_real, reg_real = dis(h_emb, r_emb, t_emb)
            h_emb_gen, r_emb_gen, t_emb_gen = get_gen_neg(h_emb, r_emb, t_emb, gen, cfg.bs, cfg.num_ng_gen, cfg.emb_dim, device, cfg.gen_std, flag='dis')
            pred_fake, reg_fake = dis(h_emb_gen, r_emb_gen, t_emb_gen)
            pred = torch.cat([pred_real.view(cfg.bs, -1), pred_fake.view(cfg.bs, -1)], dim=-1)
            loss_dis = pur_loss(pred, cfg.prior) + cfg.reg * 0.5 * (reg_real + reg_fake)
            optim_dis.zero_grad()
            loss_dis.backward()
            optim_dis.step()
            avg_loss_dis.append(loss_dis.item())

        print(f'D Loss: {round(sum(avg_loss_dis)/len(avg_loss_dis), 4)}')
        print(f'G Loss: {round(sum(avg_loss_gen)/len(avg_loss_gen), 4)}')

        if (epoch + 1) == cfg.max_epochs:
            emb_model.eval()
            dis.eval()
            r, rr, h1, h3, h10 = evaluate(test_dataloader, already_ts_dict, e_dict, emb_model, dis, device)