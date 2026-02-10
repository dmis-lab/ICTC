import scipy.sparse as sp
import numpy as np
import os
import pickle
from sklearn.decomposition import PCA

from ictc import config as args
from ictc.data.preprocessing import get_data
from ictc.evaluation.metrics import get_scores, get_aa_scores, get_cpa_scores, get_jc_scores, get_cn_scores
from ictc.models.srnmf import getXY

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

########################################
# choose similarity measure to be used #
########################################
args.similarity = 'srnmf_cn'
# args.similarity = 'srnmf_jc'
# args.similarity = 'srnmf_cpa'

if __name__ == '__main__':
    test_ap_list = []
    test_roc_list = []
    for i in range(10):
        adj, features,\
                adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all, edges_false_all = get_data(args.dataset)

        with open('data/bipartite/id2name/' +args.dataset +'u2id.pkl', 'rb') as f:
            u2id = pickle.load(f)
        with open('data/bipartite/id2name/' +args.dataset +'v2id.pkl', 'rb') as f:
            v2id = pickle.load(f)

        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()
        adj_train += adj_train.T

        pca = PCA().fit(adj_train.toarray())
        cumulative_contribution_rate = list(np.cumsum(pca.explained_variance_ratio_))
        val = min(cumulative_contribution_rate, key=lambda x:abs(x-0.95))
        k = cumulative_contribution_rate.index(val)+1
        print(k)

        if args.similarity == 'srnmf_aa':
            S = get_aa_scores(adj_train,u2id,v2id)
        if args.similarity == 'srnmf_cpa':
            S = get_cpa_scores(adj_train,u2id,v2id)
        if args.similarity == 'srnmf_jc':
            S = get_jc_scores(adj_train,u2id,v2id)
        if args.similarity == 'srnmf_cn':
            S = get_cn_scores(adj_train,u2id,v2id)
        print('finished computing S')

        X, Y = getXY(S, adj_train, k)
        print('finished computing updating XY')

        B_hat = np.nan_to_num(X@Y)

        test_roc, test_ap = get_scores(test_edges, test_edges_false, B_hat, adj_orig)
        print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
                  "test_ap=", "{:.5f}".format(test_ap))
        # exit()

        test_roc_list.append(test_roc)
        test_ap_list.append(test_ap)
        # break

    mean_roc, ste_roc = np.mean(test_roc_list), np.std(test_roc_list)/(args.numexp**(1/2))
    mean_ap, ste_ap = np.mean(test_ap_list), np.std(test_ap_list)/(args.numexp**(1/2))

    print(args.similarity)
    print('mean_roc=','{:.5f}'.format(mean_roc),', ste_roc=','{:.5f}'.format(ste_roc))
    print('mean_ap=','{:.5f}'.format(mean_ap),', ste_ap=','{:.5f}'.format(ste_ap))

    roc = '{:.1f}'.format(mean_roc*100.0)+'+'+'{:.2f}'.format(ste_roc*100.0).strip(' ')
    ap = '{:.1f}'.format(mean_ap*100.0)+'+'+'{:.2f}'.format(ste_ap*100.0).strip(' ')

    print(roc)
    print(ap)
