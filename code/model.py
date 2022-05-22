import torch


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
