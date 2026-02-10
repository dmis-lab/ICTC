import numpy as np
from sklearn.decomposition import NMF

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def getXY(S, adj_train, k):
    model = NMF(n_components=k, init='nndsvda',solver='mu',max_iter=400)
    X = model.fit_transform(adj_train.toarray())
    Y = model.components_
    print('finished computing factorizing XY')

    # print('X.shape:',X.shape)
    # print('Y.shape:',Y.shape)
    adj_train = adj_train.toarray()

    gamma = 0.5
    lamda = 2.0
    for i in range(200):
        new_X_num = adj_train@Y.T+ gamma*(S*adj_train) @ Y.T
        new_X_den = X@Y@Y.T + gamma * (S * (X@Y)) @ Y.T + lamda * X
        # new_X_den = np.where(new_X_den != 0, new_X_den, 0.0001 )
        new_X_den = np.where(new_X_den==0, 0.0001, new_X_den)
        new_X = new_X_num / new_X_den

        new_Y_num = X.T@adj_train + gamma * X.T @(S*adj_train)
        new_Y_den = X.T@X@Y+gamma*X.T@(S*(X@Y))+lamda*Y
        # new_Y_den = np.where(new_Y_den != 0, new_Y_den, 0.0001 )
        new_Y_den = np.where(new_Y_den==0, 0.0001, new_Y_den)
        new_Y = new_Y_num / new_Y_den

        X_old = X
        Y_old = Y

        X = np.multiply(X,new_X)
        Y = np.multiply(Y,new_Y)

    #     diff1 = np.linalg.norm(X_old-X, 'fro')
    #     diff2 = np.linalg.norm(Y_old-Y, 'fro')
    #     print(diff1)
    #     print(diff2)
    # exit()
    return X,Y
