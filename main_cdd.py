from tqdm import trange
import torch
import torch.nn as nn
import torch.optim as optim
from model import CDDNet
from prepare_loader import get_loader
from utils import measures_from_Yhat, combined_label
from CustomLoss import DDP_Loss, CDDLinear
from itertools import cycle

import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0,1,2,3'


device = 'cuda'
lr = 2e-4
lr_decay = 1.0 # Exponential decay factor of LR scheduler
lambda_ = 1

print('==> preparing data ... ...')
src_loader, tgt_loader = get_loader()

print('==> building model ... ...')
model = CDDNet(out_dim=256)
model = nn.DataParallel(model)
model = model.to(device)

# Set an optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
# optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-5, momentum=0.99)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

criteria = nn.CrossEntropyLoss()
CDD_Loss = CDDLinear(n_feat=256, n_cls=4, K=1)

sensitive_attrs = torch.LongTensor([0, 1]).to(device)

def train(e, da, fa):
    cls_loss_avg = 0
    intra_cdd_avg = 0
    inter_cdd_avg = 0
    ddp_avg = 0
    # acc_avg = 0

    with trange(len(src_loader)) as t:
        for i, (src, tgt) in enumerate(zip(src_loader, cycle(tgt_loader))):

            src_img, src_yz = src
            tgt_img, tgt_yz = tgt

            # print(src_yz)

            # print(src_img.shape, tgt_img.shape, src_yz.shape, tgt_yz.shape)
            src_y, src_z = src_yz[:,0], src_yz[:,1]
            tgt_y, tgt_z = tgt_yz[:,0], tgt_yz[:,1]
            src_img, tgt_img = src_img.to(device), tgt_img.to(device)

            print(src_img.shape, tgt_img.shape, src_yz.shape, tgt_yz.shape)
            src_y, tgt_y = src_y.to(device), tgt_y.to(device)
            src_z, tgt_z = src_z.to(device), tgt_z.to(device)

            src_feat, tgt_feat, src_logit, tgt_logit = model(src_img, tgt_img)


            cost = 0
            ddp = None
            cdd_intra = None
            cdd_inter = None

            # prediction loss
            cls_loss = criteria(src_logit, src_y)
            cost += cls_loss

            if fa:
                ddp = DDP_Loss(src_logit[:,1].squeeze(), src_z, sensitive_attrs, lambda_=lambda_)
                cost += ddp*1

            if da:
                # tgt_yhat = tgt_logit.argmax(dim=1)
                tgt_yhat = tgt_y
                # previous combined_lable is a typo
                target_src = combined_label(src_y, src_z)
                target_tgt = combined_label(tgt_yhat, tgt_z)
                cdd_intra, cdd_inter = CDD_Loss(src_feat.detach().cpu(), tgt_feat.detach().cpu(), target_src.cpu(), target_tgt.cpu())
                cost += (cdd_intra - cdd_inter)*.1

            optimizer.zero_grad()
            if (torch.isnan(cost)).any():
                continue
            cost.backward()
            optimizer.step()

            cls_loss_avg += cls_loss.item()
            if da:
                intra_cdd_avg += cdd_intra.item()
                inter_cdd_avg += cdd_inter.item()
            if fa:
                ddp_avg += ddp.item()
            metric = {'cls': cls_loss_avg/(i+1),
                        'cdd': '{:.3f}/{:.3f}'.format(intra_cdd_avg/(i+1), inter_cdd_avg/(i+1)),
                        'ddp': ddp_avg/(i+1)}
            t.set_postfix(metric)
            t.update()


def test(e):
    tgt_y_all = []
    tgt_z_all = []
    tgt_logit_all = []

    with trange(len(tgt_loader)) as t:
        with torch.no_grad():
            for i, (tgt_img, tgt_yz) in enumerate(tgt_loader):

                tgt_y, tgt_z = tgt_yz[:,0], tgt_yz[:,1]
                tgt_img= tgt_img.to(device)

                tgt_logit = model.module.evaluate(tgt_img)
                tgt_y_all.append(tgt_y)
                tgt_z_all.append(tgt_z)
                tgt_logit_all.append(tgt_logit.cpu())

                if i < len(tgt_loader)-1:
                    t.update()
                else:
                    tgt_y_all = torch.cat(tgt_y_all, dim=0)
                    tgt_z_all = torch.cat(tgt_z_all, dim=0)
                    tgt_yh_all = torch.cat(tgt_logit_all, dim=0)
                    acc, dp, eodd, eopp = measures_from_Yhat(tgt_y_all.numpy(), tgt_z_all.numpy(), Yhat=tgt_yh_all.numpy(), threshold=0.5)
                    metric = {'acc': acc, 'dp/eodd/eopp': '{:.3f}/{:.3f}/{:.3f}'.format(dp, eodd, eopp),}
                    t.set_postfix(metric)
                    t.update()

for e in range(200):
    train(e, da=False, fa=False)
    test(e)
    lr_scheduler.step()
