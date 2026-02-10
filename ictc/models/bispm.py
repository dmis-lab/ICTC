import numpy as np
import scipy.sparse as sp
from scipy import linalg

from ictc import config as args
from ictc.data.preprocessing import sparse_to_tuple

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def getBrAndBtriangle(adj_train):
    adj_tuple = sparse_to_tuple(adj_train)
    edges = adj_tuple[0]
    percentage = 0.9
    num_train = int(np.floor(edges.shape[0] * percentage)) # 10%

    all_edge_idx = list(range(edges.shape[0]))
    np.random.seed(args.edge_idx_seed)
    np.random.shuffle(all_edge_idx)
    B_r_idx = all_edge_idx[:num_train] # 90% of training edges
    B_triangle_idx = all_edge_idx[num_train:] # 10% of training edges
    B_r_edges = edges[B_r_idx]
    B_triangle_edges = edges[B_triangle_idx]

    data = np.ones(B_r_edges.shape[0])
    data2 = np.ones(B_triangle_edges.shape[0])
    # Re-build adj matrix
    B_r = sp.csr_matrix((data, (B_r_edges[:, 0], B_r_edges[:, 1])), shape=adj_train.shape)
    B_triangle = sp.csr_matrix((data2, (B_triangle_edges[:, 0], B_triangle_edges[:, 1])), shape=adj_train.shape)

    return B_r, B_triangle
def getBiSPM(B_r,B_triangle):
    B_r = B_r.toarray()
    B_triangle = B_triangle.toarray()
    np.random.seed(0)
    rank = np.linalg.matrix_rank(B_r)

    np.random.seed(0)
    U, s, Vh = linalg.svd(B_r, full_matrices=False)
    S = np.diag(s)

    # print('b_r.shape:',B_r.shape)
    # val = np.zeros((B_r.shape))
    # for i in range(0, rank):
    #     val += np.multiply(S[i,i] , np.outer(U[:, i].reshape(-1,1), Vh[i,:].reshape(1,-1)))

    middle_val = (B_r.T @ B_triangle) + (B_triangle.T @ B_r) # 877 * 877

    result = np.zeros((B_r.shape), dtype='float64')
    for i in range(0, rank):
        left = np.matmul(Vh[i,:].reshape(1,-1),middle_val)
        right = Vh[i,:].reshape(-1,1)

        numerator = np.matmul(left,right)
        coefficient = 2.0 * S[i,i]
        denominator = np.multiply( coefficient, np.matmul(Vh[i,:].reshape(1,-1),right ))

        delta_sigma = numerator/denominator
        delta_sigma = delta_sigma[0][0]

        result += (S[i,i] + delta_sigma) *  np.matmul(U[:, i].reshape(-1,1), Vh[i,:].reshape(1,-1))

    # print('new:',np.allclose(B_r.T @ B_r @ Vh[0,:].reshape(-1,1),S[i,i]**2.0 * Vh[0,:].reshape(-1,1)))
    # B_hat = sigmoid(result)
    B_hat = result

    Bi_adjacency_left = np.concatenate((np.zeros((B_hat.shape[0],B_hat.shape[0])), np.transpose(B_hat)), axis=0)
    Bi_adjacency_right = np.concatenate((B_hat, np.zeros((B_hat.shape[1],B_hat.shape[1]))), axis=0)
    Bi_adjacency = np.concatenate((Bi_adjacency_left, Bi_adjacency_right), axis=1)
    return Bi_adjacency
