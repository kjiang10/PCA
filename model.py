import torch
import torch.nn as nn
import torch.nn.functional as F
import backbone as bb


class CDDNet(nn.Module):
    def __init__(self, config=None, base='ResNet50', out_dim=256, n_cls=2):
        super(CDDNet, self).__init__()
        if config:
            base = config['base_net']
            out_dim = config['hidden_dim']
            n_cls = config['num_cls']
        self.n_cls = n_cls
        self.out_dim = out_dim
        self.base_network = bb.network_dict[base]()
        hidden_size = out_dim
        self.bn = nn.Sequential(
            nn.Linear(self.base_network.output_num(), hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Sigmoid(),
            # nn.Dropout(0.5),
            nn.Linear(hidden_size, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Sigmoid(),
        )
        # self.cls = nn.Linear(out_dim, n_cls)
        self.cls = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.Sigmoid(),
            # nn.Dropout(0.5),
            nn.Linear(out_dim, n_cls)
        )

    def forward(self, src_x, tgt_x):
        src_feat = self.base_network(src_x)
        tgt_feat = self.base_network(tgt_x)
        src_feat = self.bn(src_feat)
        tgt_feat = self.bn(tgt_feat)

        src_y = self.cls(src_feat)
        tgt_y = self.cls(tgt_feat)
        return src_feat, tgt_feat, src_y, tgt_y
    
    def evaluate(self, tgt_x):
        with torch.no_grad():
            tgt_feat = self.base_network(tgt_x)
            tgt_feat = self.bn(tgt_feat)
            tgt_y = self.cls(tgt_feat)
            tgt_y = tgt_y.softmax(dim=1)
            return tgt_y[:,1]

    # def predict(self, tgt_x):
    #     tgt_feat = self.base_network(tgt_x)
    #     tgt_feat = self.bn(tgt_feat)
    #     # tgt_y = self.cls(tgt_feat)
    #     # tgt_y = tgt_y.softmax(dim=1)
    #     return tgt_feat

if __name__ == '__main__':
    device = torch.device('cuda:3')
    x = torch.rand(10, 3, 256, 256).to(device)
    y = torch.rand(10, 3, 256, 256).to(device)
    nets = MultiHeadNet().to(device)
    s, t, sy, ty = nets(x, y)
    print(s.shape, sy.shape)

