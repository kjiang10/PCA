import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA
import pdb

from prepare_data import vectorize_image

from sklearn.neighbors import KernelDensity
from KDEpy import NaiveKDE
import matplotlib.pyplot as plt
import prepare_data
import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LogisticRegression
import sys, os 
from matplotlib.ticker import StrMethodFormatter

def create_shift(data, sampling, src_split = .4, mean_a = .5, std_b = 1.5, feature_ind = 0, threshold = 0.5, p_s = 0.7, p_t = 0.5, kdebw = 'ISJ'):
    '''
    create shift according to the sampling
    sampling = 'pca' , based on PCA and 1d gaussian
    sampling = 'feature' , based on sensitive feature, a (using \mathbb{E}[p(a)] = 0.7 as example)

    data: [m, n+1] with label at the last dimension
    sampling: "pca" or "feature"
    mean_a, std_b: the parameter that distorts the gaussian used in sampling
                   according to the first principle n_components
    feature_ind, threshold, p_s, p_t: feature_ind for sensitive features, threshold and 
                                      source and target probabilities, respectively.
                                      For example, default setting is, 
                                      \mathbb{E}_{p_s}[p(a_0 > 0.5)] = 0.7 
                                      \mathbb{E}_{p_t}[p(a_0 > 0.5)] = 0.5 

    output: if PCA, [m/2, n+1] as source, [m/2, n+1] as target
    '''
    #features = data[:, 0:-2]
    #labels = data[:, -1]
    m = np.shape(data)[0]
    source_size = int(m * src_split)
    #target_size = m - source_size
    target_size = source_size
    if sampling == 'pca':
    #PCA
        pca = PCA(n_components=2)
        pc2 = pca.fit_transform(data)
        pc = pc2[:,0]
        #pc2 = pc2[:,:2]
        pc = pc.reshape(-1,1)
        #pc = pc_[:,0]
        # or use certain feature dimension to sample
        #pc = data[:,0]
        #print(pc)
        pc_mean = np.mean(pc)
        pc_std = np.std(pc)

        #sample_mean = pc_mean/mean_a
        sample_mean = pc_mean + mean_a
        sample_std = pc_std/std_b

        print(sample_mean)
        print(pc_mean)
 
        # sample according to the probs
        prob_s = norm.pdf(pc, loc = sample_mean, scale = sample_std)
        sum_s = np.sum(prob_s)
        prob_s = prob_s/sum_s
        prob_t = norm.pdf(pc, loc = pc_mean, scale = pc_std)
        sum_t = np.sum(prob_t)
        prob_t = prob_t/sum_t
        # test = np.random.choice(range(m), 2, replace = False, p = prob_s)
        #print(prob_s)
        source_ind = np.random.choice(range(m), size = source_size, replace = False, p = np.reshape(prob_s, (m)) )
        pt_proxy = np.copy(prob_t)
        pt_proxy[source_ind] = 0
        pt_proxy = pt_proxy/np.sum(pt_proxy)
        target_ind = np.random.choice(range(m), size = target_size, replace = False, p = np.reshape(pt_proxy, (m)) )
         
        #target_ind = np.random.choice(range(m), size = target_size, replace = False, p = np.reshape(prob_t, (m)) )
        #ps_proxy = np.copy(prob_s)
        #ps_proxy[target_ind] = 0
        #sum_proxy = sum(ps_proxy)
        #ps_proxy = ps_proxy/sum_proxy
        #source_ind = np.random.choice(range(m), size = source_size, replace = False, p = np.reshape(ps_proxy, (m)) )
        source_data = data[source_ind, :]
        target_data = data[target_ind, :]
        assert(np.all(np.sort(source_ind) != np.sort(target_ind)))
        src_kde = KDEAdapter(kde = NaiveKDE(kernel='gaussian', bw=kdebw)).fit(pc2[source_ind,:])
        trg_kde = KDEAdapter(kde = NaiveKDE(kernel='gaussian', bw=kdebw)).fit(pc2[target_ind,:])

        #src_kde = KDEAdapter(kde = KernelDensity(kernel='gaussian', bandwidth=kdebw)).fit(pc[source_ind])
        #trg_kde = KDEAdapter(kde = KernelDensity(kernel='gaussian', bandwidth=kdebw)).fit(pc[target_ind])
         
        ### original
        ratios = src_kde.p(pc2) / trg_kde.p(pc2)
        print("min ratios {:.5f}, max ratios {:.5f}".format(np.min(ratios), np.max(ratios)))
        ratios = np.maximum(np.minimum(ratios, 10), .1)
        ratios = ratios.reshape(-1,1)

        source_ratios = ratios[source_ind]
        target_ratios = ratios[target_ind]
        #pdb.set_trace()
    else:
        source_ind = []
        target_ind = []
        prob_s = np.zeros(m)
        prob_t = np.zeros(m)
        # sample according to a's value
        a_value = data[:, feature_ind]
        for i in range(m):
            # if equality, change to "=="
            if a_value[i] > threshold:
                rand = np.random.uniform(0, 1, 1)
                prob_s[i] = p_s
                if rand < p_s:
                    source_ind.append(i)
                     
            else:
                rand = np.random.uniform(0, 1, 1)
                prob_s[i] = 1 - p_s
                if rand > p_s:
                    source_ind.append(i)
                    

        for i in range(m):
            # if equality, change to "=="
            if a_value[i] > threshold:
                rand = np.random.uniform(0, 1, 1)
                prob_t[i] = p_t
                if rand < p_t:
                    target_ind.append(i)
                    
            else:
                rand = np.random.uniform(0, 1, 1)
                prob_t[i] = 1 - p_t
                if rand > p_t:
                    target_ind.append(i)
                    

        source_data = data[source_ind, :]
        target_data = data[target_ind, :]
        #print(np.shape(source_data))
        #print(np.shape(target_data))

        

        ratios = prob_s/prob_t

        source_ratios = ratios[source_ind]
        target_ratios = ratios[target_ind]

        # second uniform sampling according to size
         
        #source_size_ind_unif = np.random.choice(range(source_data.shape[0]), size = source_size, replace = False)
        #target_size_ind_unif = np.random.choice(range(target_data.shape[0]), size = target_size, replace = False)

        #source_data = source_data[source_size_ind_unif, :]
        #target_data = target_data[target_size_ind_unif, :]

        #source_ratios = ratios[source_size_ind_unif]
        #target_ratios = ratios[target_size_ind_unif]
         
        
    print(np.shape(source_data))
    print(np.shape(target_data))
    #print(source_ratios)
    #print(target_ratios)
    #return source_data, target_data, source_ratios, target_ratios, source_t_prob, target_t_prob, source_ind, target_ind
    return source_ind, target_ind, np.squeeze(ratios)




def load_data_xay(dataset):
    data = vectorize_image()
    #dataX = pd.concat([dataA,dataX],axis = 1).values
    return data
    

def test_shift(dataset, **kwargs):
    data = load_data_xay(dataset)
    
    src_ind, trg_ind, ratios = create_shift(data, src_split = kwargs['split'], sampling = kwargs['sampling'], mean_a = kwargs['param1'], std_b = kwargs['param2'], p_s=kwargs['param1'], p_t=kwargs['param2'], kdebw = kwargs['kdebw'])
    pca = PCA(n_components=2)
    # print(data)
    pc = pca.fit_transform(data)[:,0]
    f = data[:,4]
    pc_mean = np.mean(pc)
    pc_std = np.std(pc)

    sample_mean = pc_mean + kwargs['param1']
    sample_std = pc_std/ kwargs['param2']
    
    #lr = LogisticRegression(solver='liblinear',fit_intercept=True, C = .01)
    #lr.fit(data,dataY.values)
    #c_xy = np.histogram2d(pc, lr.predict_proba(data)[:,1] , bins=20)[0]
    #mi = mutual_info_score(None, None, contingency=c_xy)
    #print('MI = {:.4f}'.format(mi))
    #print(data[src_ind].shape)
    plt.subplot(3,1,1)
    #plt.subplot(1,1,1)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig = plt.gcf()
    fig.set_size_inches(8,10)
    font_options={'family' : 'sans-serif','size' : '24'}
    plt.rc('font', **font_options)
    
    _, src_bins, _ = plt.hist(pc[src_ind],bins=50, alpha = .5, label = 'src', density= True)
    _, trg_bins, _ = plt.hist(pc[trg_ind],bins=50, alpha =.5, label = 'trg', density= True)
    #plt.title(' m = {:.2f} , s = {:.2f}'.format(kwargs['param1'], kwargs['param2']))
    plt.title(r'src ($\bar x\!=\!${:.2f}, $s\!=\!${:.2f}), trg ($\bar x\!=\!${:.2f}, $s\!=\!${:.2f})'.format(np.mean(pc[src_ind]), np.std(pc[src_ind]), np.mean(pc[trg_ind]), np.std(pc[trg_ind])), fontdict = {'fontsize' : 24})
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    #plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) 
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    plt.legend()
    #src_ind_srt = np.argsort(pc[src_ind])
    #trg_ind_srt = np.argsort(pc[trg_ind])
    values = np.arange(-3,3,.1).reshape(-1,1) 
    #src_kde = KDEAdapter(KernelDensity(kernel='gaussian', bandwidth=.2)).fit(pc[src_ind].reshape(-1,1))
    #trg_kde = KDEAdapter(KernelDensity(kernel='gaussian', bandwidth=.2)).fit(pc[trg_ind].reshape(-1,1))
    src_kde = KDEAdapter(NaiveKDE(kernel='gaussian', bw=kwargs['kdebw'])).fit(pc[src_ind].reshape(-1,1))
    trg_kde = KDEAdapter(NaiveKDE(kernel='gaussian', bw=kwargs['kdebw'])).fit(pc[trg_ind].reshape(-1,1))
     
    #trg_kde = density_estimator(pc[trg_ind].reshape(-1,1),kwargs['kdebw'])
    #src_ps = np.exp(src_kde.score_samples(pc[src_ind].reshape(-1,1)))
    #trg_ps = np.exp(trg_kde.score_samples(pc[trg_ind].reshape(-1,1)))
    #src_pc = src_kde.pdf(pc[src_ind].reshape(-1,1))
    #trg_pc = trg_kde.pdf(pc[trg_ind].reshape(-1,1))
    
    src_bins = 0.5*(src_bins[1:] + src_bins[:-1])
    trg_bins = 0.5*(trg_bins[1:] + trg_bins[:-1])

    #plt.plot(src_bins, src_kde.evaluate(src_bins), color ='blue')
    #plt.plot(trg_bins, trg_kde.evaluate(trg_bins), color = 'orange')
    plt.plot(src_bins, src_kde.pdf(src_bins), color ='blue')
    plt.plot(trg_bins, trg_kde.pdf(trg_bins), color = 'orange')

    outfile = "{}_histogram_{:.2f}_{:.2f}_{:.2f}".format(dataset,kwargs['param1'],kwargs['param2'],kwargs['kdebw']).replace('.','-') #+ "_err"
    plot_path = "deo_plot"
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    plt.savefig(os.path.join(plot_path,outfile + '.pdf'), dpi=300,bbox_inches='tight')
    # plt.savefig(os.path.join(plot_path,outfile+ '.png'),figsize=(.2,.2),dpi=300,bbox_inches='tight')
    #plt.clf()

    prob_s = norm.pdf(src_bins, loc = sample_mean, scale = sample_std)
    #prob_s = prob_s/ np.sum(prob_s)
    prob_t = norm.pdf(trg_bins, loc = pc_mean, scale = pc_std)
    #prob_t = prob_t/ np.sum(prob_t)

    #plt.plot(src_bins, prob_s, color ='blue')
    #plt.plot(trg_bins, prob_t, color = 'orange')

     
     
    ax = plt.subplot(3,1,2)
    #plt.plot(pc[src_ind_srt], src_ps, label='src')
    #plt.plot(pc[trg_ind_srt], trg_ps, label='trg')
    pc2 = pca.fit_transform(data)
    src_kde = KDEAdapter(NaiveKDE(kernel='gaussian', bw=kwargs['kdebw'])).fit(pc2[src_ind])
    trg_kde = KDEAdapter(NaiveKDE(kernel='gaussian', bw=kwargs['kdebw'])).fit(pc2[trg_ind])
    #src_pc = src_kde.p(pc[src_ind].reshape(-1,1))
    #trg_pc = trg_kde.p(pc[trg_ind].reshape(-1,1))
    
    src_kde2 = NaiveKDE(kernel='gaussian', bw=kwargs['kdebw']).fit(pc2[src_ind,:])
    trg_kde2 = NaiveKDE(kernel='gaussian', bw=kwargs['kdebw']).fit(pc2[trg_ind,:])
     
    #src_mean = np.dot(src_pc, pc[src_ind])
    #src_mean2 = np.dot(src_pc, np.power(pc[src_ind],2))
    #src_std = np.sqrt(src_mean2 - np.power(src_mean,2))
    print("src posterior mean: {:.4f}".format(np.mean(pc[src_ind])))
    print("src posterior std: {:.4f}".format(np.std(pc[src_ind])))
    #trg_mean = np.dot(trg_pc, pc[trg_ind])
    #trg_mean2 = np.dot(trg_pc, np.power(pc[trg_ind],2))
    #trg_std = np.sqrt(trg_mean2 - np.power(trg_mean,2))
    print("trg posterior mean: {:.4f}".format(np.mean(pc[trg_ind])))
    print("trg posterior std: {:.4f}".format(np.std(pc[trg_ind])))
     
    grid_points = 256
    grid_s, points_s = src_kde2.evaluate(grid_points)
    grid_t, points_t = trg_kde2.evaluate(grid_points)

    # The grid is of shape (obs, dims), points are of shape (obs, 1)
    x_s, y_s = np.unique(grid_s[:, 0]), np.unique(grid_s[:, 1])
    x_t, y_t = np.unique(grid_t[:, 0]), np.unique(grid_t[:, 1])
    
    z_s = points_s.reshape(grid_points, grid_points).T
    z_t = points_t.reshape(grid_points, grid_points).T
    N = 16
    # Plot the kernel density estimate
    #ax.contourf(x_s, y_s, z_s, N, cmap="Blues", extend = 'neither')
    ax.contourf(x_t, y_t, z_t, N, cmap="Oranges", extend= 'neither')
    
    ax.plot(pc2[src_ind, 0], pc2[src_ind, 1], 'ok', ms=1, color = 'blue', alpha = .2, label = 'src')
    ax.plot(pc2[trg_ind, 0], pc2[trg_ind, 1], 'ok', ms=1, color = 'orange', alpha = .2, label = 'trg')
    ax.contour(x_s, y_s, z_s, N, linewidths=0.8, colors='blue', label = 'src')
    ax.contour(x_t, y_t, z_t, N, linewidths=0.8, colors='orange', label = 'trg')

    outfile = '2d_countor'
    plt.show()

class KDEAdapter():
    def __init__(self, kde = KernelDensity(kernel = 'gaussian', bandwidth = .3)):
        self._kde = kde

    def fit(self, sample):
        self._kde.fit(sample)
        return self

    def pdf(self, sample):
        if isinstance(self._kde , KernelDensity):
            return np.exp(self.logp(sample))
        elif isinstance(self._kde , NaiveKDE):
            density = self._kde.evaluate(sample)
            return density 
    
    def p(self, sample):
        if isinstance(self._kde , KernelDensity):
            return np.exp(self.logp(sample))
        elif isinstance(self._kde , NaiveKDE):
            density = self._kde.evaluate(sample)
            return density / np.sum(density)

     
    def logp(self,sample):
        if isinstance(self._kde , KernelDensity):
            return self._kde.score_samples(sample)
        elif isinstance(self._kde , NaiveKDE):
            return np.log(self.p(sample))
        

def main():
    # load data
    # data = np.genfromtxt("ecoli.csv", delimiter=",")
    # print(data)
    # print(np.shape(data))

    data = load_data_xay('celebA')


    # print(create_shift(data, src_split = .4, sampling = 'pca', mean_a = 0 , std_b = 1, kdebw = .3))



if __name__ == '__main__':
    main()
    # test_shift('celebA', split = .4, sampling = 'pca', param1 = 0, param2 = 1, kdebw = .3)
    # test_pykliep(sys.argv[1], split = .4, sampling = 'pca', param1 = 1.5, param2 = 3, kdebw = .3)