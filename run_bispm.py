import scipy.sparse as sp
import numpy as np
import os
import pickle

from ictc import config as args
from ictc.data.preprocessing import get_data
from ictc.evaluation.metrics import get_scores
from ictc.models.bispm import getBrAndBtriangle, getBiSPM

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

if __name__ == '__main__':
    test_ap_list = []
    test_roc_list = []
    for i in range(10):

        adj, features,\
                adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, edges_all, false_edges = get_data(args.dataset)

        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()

        with open('data/bipartite/id2name/'+ str(args.dataset)  +'u2id.pkl', 'rb') as f:
            u2id = pickle.load(f)
        with open('data/bipartite/id2name/'+ str(args.dataset) +'v2id.pkl', 'rb') as f:
            v2id = pickle.load(f)

        adj_train = adj_train[:len(u2id),len(u2id):]

        Bi_adjacency = np.zeros(adj.shape)
        for i in range(1):
            B_r, B_triangle = getBrAndBtriangle(adj_train)
            Bi_adjacency += getBiSPM(B_r,B_triangle)
        Bi_adjacency /= 1.0

        test_roc, test_ap = get_scores(test_edges, test_edges_false, Bi_adjacency, adj_orig)
        print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
                  "test_ap=", "{:.5f}".format(test_ap))
        test_roc_list.append(test_roc)
        test_ap_list.append(test_ap)
        # break

    mean_roc, ste_roc = np.mean(test_roc_list), np.std(test_roc_list)/(args.numexp**(1/2))
    mean_ap, ste_ap = np.mean(test_ap_list), np.std(test_ap_list)/(args.numexp**(1/2))

    print('BiSPM')

    roc = '{:.1f}'.format(mean_roc*100.0)+'+'+'{:.2f}'.format(ste_roc*100.0).strip(' ')
    ap = '{:.1f}'.format(mean_ap*100.0)+'+'+'{:.2f}'.format(ste_ap*100.0).strip(' ')

    print(roc)
    print(ap)
