import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

tau = 0.5

# Approximation of Q-function given by López-Benítez & Casadevall (2011) based on a second-order exponential function & Q(x) = 1- Q(-x):
a = 0.4920
b = 0.2887
c = 1.1893
Q_function = lambda x: torch.exp(-a*x**2 - b*x - c) 

pi = torch.tensor(np.pi)#.to(device)
phi = lambda x: torch.exp(-0.5*x**2)/torch.sqrt(2*pi) #normal distribution

def CDF_tau(Yhat, h=0.01, tau=0.5):
    m = len(Yhat)
    Y_tilde = (tau-Yhat)/h
    sum_ = torch.sum(Q_function(Y_tilde[Y_tilde>0])) \
           + torch.sum(1-Q_function(torch.abs(Y_tilde[Y_tilde<0]))) \
           + 0.5*(len(Y_tilde[Y_tilde==0]))
    return sum_/m

def DDP_Loss(Yhat, z_batch, sen_att, lambda_=0.05, delta=1., h=0.1, tau=0.5):
    m = z_batch.shape[0]
    Pr_Ytilde1 = CDF_tau(Yhat.detach(),h,tau)
    for z in sen_att:
        Pr_Ytilde1_Z = CDF_tau(Yhat.detach()[z_batch==z],h,tau)
        m_z = z_batch[z_batch==z].shape[0]

        Delta_z = Pr_Ytilde1_Z-Pr_Ytilde1
        Delta_z_grad = torch.dot(phi((tau-Yhat.detach()[z_batch==z])/h).view(-1), 
                                    Yhat[z_batch==z].view(-1))/h/m_z
        Delta_z_grad -= torch.dot(phi((tau-Yhat.detach())/h).view(-1), 
                                    Yhat.view(-1))/h/m

        if Delta_z.abs() >= delta:
            if Delta_z > 0:
                Delta_z_grad *= lambda_*delta
            else:
                Delta_z_grad *= -lambda_*delta
        else:
            Delta_z_grad *= lambda_*Delta_z
        
        return Delta_z_grad

class CDDLinear(nn.Module):
    def __init__(self, config=None, K=500, n_cls=4, n_feat=256, device='cpu'):
        super(CDDLinear, self).__init__()
        if config is not None:
            self.K = config['buffer_size']
            self.n_cls = config['num_cls']
            self.n_feat = config['hidden_dim']
            self.device = config['device']
            # self.scale_down = config['scale_down']
        else:
            self.K = K
            self.n_cls = n_cls
            self.n_feat = n_feat
            self.device = device
            # self.scale_down = scale_down

        self.eps = 1e-8*torch.ones((self.n_cls,1), device=device)
        if self.K > 0:
            self.register_buffer('queue', torch.zeros((self.n_cls, self.n_feat, self.K), device=self.device))
            self.register_buffer('queue_pt', torch.zeros((self.n_cls,), dtype=torch.long, device=self.device))
            self.register_buffer('queue_size', torch.zeros((self.n_cls), device=self.device))

    @torch.no_grad()
    def _dequeue_enqueue(self, reps, src_y):
        src_y = src_y.argmax(dim=1)
        assert len(src_y.size()) == 1, 'The source label shape should be 1d.'
        cls_list = torch.unique(src_y)
        for c in cls_list:
            pointer = self.queue_pt[c]
            idx_c = src_y==c
            nc = idx_c.sum()
            if pointer+nc < self.K:
                self.queue[c, :, pointer:pointer+nc] = reps[idx_c].T
                self.queue_pt[c] = (pointer + nc) % self.K
                self.queue_size[c] = max(self.queue_size[c], self.queue_pt[c])
            else:
                self.queue[c, :, pointer:self.K] = reps[idx_c].T[:, :self.K-pointer]
                self.queue_pt[c] = 0
                self.queue_size[c] = max(self.queue_size[c], self.K)
    
    def src_centroid(self, cls_list=None):
        if cls_list is None:
            cls_list = list(range(self.n_cls))
            n_cls = self.n_cls
        else:
            n_cls = len(cls_list)
        centroid = torch.zeros((n_cls, self.n_feat), device=self.device)
        num_sample = torch.zeros((n_cls, 1), device=self.device)
        for i, c in enumerate(cls_list):
            num_sample[i] = self.queue_size[c]
            if num_sample[i] >0:
                centroid[i] = torch.mean(self.queue[c][:, :int(num_sample[i])], dim=1)
        return centroid, num_sample
    
   
    def intra_loss(self, src_x, tgt_x, src_y, tgt_y):
        
        batch_src = torch.mm(src_y.T, src_x)
        batch_tgt = torch.mm(tgt_y.T, tgt_x)

        buffer_src = self.src_center * self.src_num
        total_num = (self.src_num + torch.sum(src_y.T, dim=1, keepdim=True))  + 1e-8*torch.ones((self.n_cls,1), device=src_y.device)
        mean_src = (buffer_src + batch_src)/total_num

        ##TODO: divide by the probability or total number of samples
        batch_prob = (torch.sum(tgt_y.T, dim=1, keepdim=True)  
                    + 1e-8*torch.ones((self.n_cls,1), device=src_y.device))
        mean_tgt = batch_tgt/batch_prob
        dist = torch.cdist(mean_src, mean_tgt)
        
        return dist


    def inter_loss(self, src_x, src_y):

        batch_src = torch.mm(src_y.T, src_x)
        buffer_src = self.src_center * self.src_num
        total_num = ((self.src_num + torch.sum(src_y.T, dim=1, keepdim=True))
                      + 1e-8*torch.ones((self.n_cls,1), device=src_y.device))
        mean_src = (buffer_src + batch_src)/total_num

        dist = torch.cdist(mean_src, mean_src)
        return dist

    def forward(self, src_x, tgt_x, src_y, tgt_y, mask=None):
        """
        inputs:
            - src_x, tgt_x: (nc, ns, nf) or (ns, nf)
            - src_y: (ns,)
            - tgt_u: (ns, nc)
        """
        src_x, tgt_x, src_y, tgt_y = src_x.cpu(), tgt_x.cpu(), src_y.cpu(), tgt_y.cpu()
        # n_sample, n_cls = tgt_y.size()
        if len(src_y.size()) == 1:
            src_y = torch.eye(self.n_cls)[src_y].to(src_y.device)
        if len(tgt_y.size()) == 1:
            tgt_y = torch.eye(self.n_cls)[tgt_y].to(tgt_y.device)
        
        if self.K > 0:
            self.src_center, self.src_num = self.src_centroid()
        else:
            self.src_center = torch.zeros((self.n_cls, self.n_feat), device=src_x.device)
            self.src_num = torch.zeros((self.n_cls,1), device=src_y.device)
        intra = self.intra_loss(src_x, tgt_x, src_y, tgt_y)
        inter = self.inter_loss(src_x, src_y)
        if self.K > 0:
            self._dequeue_enqueue(src_x, src_y)
        
        if mask is not None:
            intra = intra[mask]
            inter = inter[mask]
            intra = intra[:,mask]
            inter = inter[:,mask]

        intra = torch.diag(intra).mean()
        inter = torch.mean(inter)
        return intra.cuda(), inter.cuda()


class CDDKernel(nn.Module):
    def __init__(self, kernel_num=5, kernel_mu=2, num_cls=12):
        super(CDDKernel, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mu = kernel_mu
        self.num_cls = num_cls
        # self.alpha = alpha

    def forward(self, src_x, tgt_x, src_y, tgt_y):
        if len(src_y.size()) == 1:
            src_y = torch.eye(self.num_cls)[src_y].to(src_y.device)
        
        intra = self.intra_loss(src_x, tgt_x, src_y, tgt_y)
        inter = self.inter_loss(src_x, src_y)
        return intra, inter
    
    def get_loss(self, src_x, tgt_x, src_y, tgt_y):
        if len(src_y.size()) == 1:
            src_y = torch.eye(self.num_cls)[src_y].to(src_y.device)
        edist = {}
        edist['ss'] = torch.cdist(src_x, src_x)
        edist['tt'] = torch.cdist(tgt_x, tgt_x)
        edist['st'] = torch.cdist(src_x, tgt_x)

        prob = {}
        prob['ss'], prob['st'], prob['tt'] = self.prob_mtx(src_y, tgt_y)

        pdist = {}
        for k in ['ss', 'st', 'tt']:
            pdist[k] = self.prob_dist(edist[k], prob[k])
        
        mmd_mtx = torch.zeros(self.num_cls, self.num_cls)
        for sc in range(self.num_cls):
            for tc in range(self.num_cls):
                gamma = self.gamma_estimation(pdist, prob, sc, tc)
                kernel_ss = self.kernel_dist(pdist['ss'][sc], gamma)
                kernel_tt = self.kernel_dist(pdist['tt'][tc], gamma)
                kernel_st = self.kernel_dist(pdist['st'][sc][tc], gamma)
                kernel_val = (self.prob_mean(kernel_ss, prob['ss'][sc]) + 
                                self.prob_mean(kernel_tt, prob['tt'][tc]) - 
                                2*self.prob_mean(kernel_st, prob['st'][sc][tc]))
                mmd_mtx[sc][tc] = kernel_val.sum()
        
        intra = (torch.diag(mmd_mtx).sum())/self.num_cls
        inter = (mmd_mtx.sum() - intra*self.num_cls)/(self.num_cls**2 - self.num_cls)

        return intra, inter
    
    def intra_loss(self, src_x, tgt_x, src_y, tgt_y):
        edist = {}
        edist['ss'] = torch.cdist(src_x, src_x)
        edist['tt'] = torch.cdist(tgt_x, tgt_x)
        edist['st'] = torch.cdist(src_x, tgt_x)

        prob = {}
        prob['ss'], prob['st'], prob['tt'] = self.prob_mtx(src_y, tgt_y)

        pdist = {}
        for k in ['ss', 'st', 'tt']:
            pdist[k] = self.prob_dist(edist[k], prob[k])
        
        mmd_mtx = torch.zeros(self.num_cls)
        for sc in range(self.num_cls):
            tc = sc
            gamma = self.gamma_estimation(pdist, prob, sc, tc)
            kernel_ss = self.kernel_dist(pdist['ss'][sc], gamma)
            kernel_tt = self.kernel_dist(pdist['tt'][tc], gamma)
            kernel_st = self.kernel_dist(pdist['st'][sc][tc], gamma)
            kernel_val = (self.prob_mean(kernel_ss, prob['ss'][sc]) + 
                            self.prob_mean(kernel_tt, prob['tt'][tc]) - 
                            2*self.prob_mean(kernel_st, prob['st'][sc][tc]))
            mmd_mtx[sc] = kernel_val.sum()
        
        intra = ((mmd_mtx).sum())/self.num_cls
        return intra
    
    def inter_loss(self, src_x, src_y):
        edist = {}
        edist['ss'] = self.euclidean_dist(src_x, src_x)
        edist['tt'] = edist['ss']
        edist['st'] = edist['ss']

        prob = {}
        prob['ss'], prob['st'], prob['tt'] = self.prob_mtx(src_y, src_y)

        pdist = {}
        for k in ['ss', 'st', 'tt']:
            pdist[k] = self.prob_dist(edist[k], prob[k])
        
        mmd_mtx = torch.zeros(self.num_cls, self.num_cls)
        for sc in range(self.num_cls):
            for tc in range(self.num_cls):
                if sc != tc:
                    gamma = self.gamma_estimation(pdist, prob, sc, tc)
                    kernel_ss = self.kernel_dist(pdist['ss'][sc], gamma)
                    kernel_tt = self.kernel_dist(pdist['tt'][tc], gamma)
                    kernel_st = self.kernel_dist(pdist['st'][sc][tc], gamma)
                    kernel_val = (self.prob_mean(kernel_ss, prob['ss'][sc]) + 
                                    self.prob_mean(kernel_tt, prob['tt'][tc]) - 
                                    2*self.prob_mean(kernel_st, prob['st'][sc][tc]))
                    mmd_mtx[sc][tc] = kernel_val.sum()
        
        inter = mmd_mtx.sum()/(self.num_cls**2 - self.num_cls)
        return inter

    def prob_mean(self, kernel, prob):
        return kernel*prob/(prob.sum()+0.)

    def kernel_dist(self, dist, gamma, kernel_num=5, kernel_mu=2):
        gamma = gamma.view(-1, 1)
        bandwidth = gamma / (kernel_mu ** (kernel_num//2))
        bandwidths = [bandwidth * (kernel_mu**i) for i in range(kernel_num)] 
        bandwidths = torch.stack(bandwidths, dim=0)

        eps = torch.ones_like(bandwidths)*1e-5
        bandwidths = torch.where(bandwidths > 1e-5, bandwidths, eps)

        for _ in range(len(bandwidths.size()) - len(dist.size())):
            dist = dist.unsqueeze(0)
        
        dist = dist/bandwidths
        ub = torch.ones_like(dist)*1e5
        lb = torch.ones_like(dist)*1e-5
        dist = torch.where(dist<1e5, dist, ub)
        dist = torch.where(dist>1e-5, dist, lb)
        kernel_val = torch.sum(torch.exp(-dist), dim=0)
        return kernel_val
    
    def gamma_estimation(self, pdist, prob, src_cls, tgt_cls):
        dist_sum = pdist['ss'][src_cls].sum() + pdist['tt'][tgt_cls].sum() + 2*pdist['st'][src_cls][tgt_cls].sum()
        prob_sum = prob['ss'][src_cls].sum() + prob['tt'][tgt_cls].sum() + 2*prob['st'][src_cls][tgt_cls].sum()

        gamma = dist_sum.item()/prob_sum
        return gamma.detach()


    def prob_mtx(self, src_y, tgt_y):
        ns, scls = src_y.size()
        nt, tcls = src_y.size()
        prob_ss = torch.empty(scls, ns, ns)
        prob_st = torch.empty(scls, tcls, ns, nt)
        prob_tt = torch.empty(tcls, nt, nt)
        for sc in range(scls):
            probs = src_y[:, sc]
            prob_ss[sc] = (torch.outer(probs, probs))
        for tc in range(tcls):
            probt = tgt_y[:, tc]
            prob_tt[tc] = (torch.outer(probt, probt))
        for sc in range(scls):
            for tc in range(tcls):
                probs = src_y[:, sc]
                probt = tgt_y[:, tc]
                prob_st[sc, tc] = (torch.outer(probs, probt))
        
        return prob_ss.to(src_y.device), prob_st.to(src_y.device), prob_tt.to(src_y.device)
    
    def prob_dist(self, edist, prob):
        for _ in range(len(prob.size()) - len(edist.size())):
            edist = edist.unsqueeze(0)
        return edist*prob

############## MMD ##########################

def RBF_full(source, target, kernel_mu=2.0, kernel_num=5, fix_sigma=None):
    n_sample = source.size(0) + target.size(0)
    all_sample = torch.cat([source, target], dim=0)
    all_sample0 = all_sample.unsqueeze(0).expand(n_sample, n_sample, all_sample.size(1))
    all_sample1 = all_sample.unsqueeze(1).expand(n_sample, n_sample, all_sample.size(1))
    l2_distance = ((all_sample0 - all_sample1)**2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distance.data) / ((n_sample - 1)*n_sample)

    bandwidth /= kernel_mu**(kernel_num//2)
    bandwidths = [bandwidth * (kernel_mu**i) for i in range(kernel_num)]
    rbf_val = [torch.exp(-l2_distance / bandwidth_i) for bandwidth_i in bandwidths]
    return sum(rbf_val)

def MMDLinear(source, target, kernel_mu=2.0, kernel_num=5, fix_sigma=None):
    batch_size = source.size(0)
    kernels = RBF_full(source, target, kernel_mu=kernel_mu, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

if __name__ == '__main__':
    cdd = CDDLinear(K=0, n_cls=3, n_feat=4)
    for _ in range(1):
        src_x = torch.randn(3, 20, 4)
        tgt_x = src_x
        src_y = torch.randint(0,3,(20,))
        tgt_y = torch.eye(3)[src_y]
        # tgt_y = src_y
        loss = cdd(src_x, tgt_x, src_y, tgt_y)
        print(loss)
        src_x = torch.randn(20, 4)
        tgt_x = src_x
        src_y = torch.randint(0,3,(20,))
        tgt_y = torch.eye(3)[src_y]
        # tgt_y = src_y
        loss = cdd(src_x, tgt_x, src_y, tgt_y)
        print(loss)

    ncls = 10
    nsample = 100
    src_x = torch.rand(nsample, 40)
    tgt_x = torch.rand(nsample, 40)
    src_y = torch.randint(0,ncls,(nsample,))
    tgt_y = torch.randint(0,ncls,(nsample,))

    sy = torch.eye(ncls)[src_y]
    ty = torch.eye(ncls)[tgt_y]

    cdd_loss = CDDKernel(num_cls=ncls)
    cdd = cdd_loss.get_loss(src_x, tgt_x, sy, ty)
    cdd2 = cdd_loss(src_x, tgt_x, sy, ty)
    print(cdd)
    print(cdd2)