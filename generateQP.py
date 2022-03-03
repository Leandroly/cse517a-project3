"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
C : regularization constant

Output:
Q,p,G,h,A,b as defined in qpsolvers.solve_qp

A call of qpsolvers.solve_qp(Q, p, G, h, A, b) should return the optimal nx1 vector of alphas
of the SVM specified by K, yTr, C. Just make these variables np arrays.

"""
import numpy as np

def generateQP(K, yTr, C):
    yTr = yTr.astype(np.double)
    n = yTr.shape[0]
    
    # YOUR CODE HERE
    a = np.zeros((n,n))
    np.fill_diagonal(a,1)

    A = yTr.T
    A = np.matrix(A)

    b = np.double(0)
    b = np.matrix(b)

    Q = np.dot(yTr,yTr.T) * K
    Q = np.matrix(Q)

    p = (-1) * np.ones([n,1])
    p = np.matrix(p)

    G = np.vstack((a, (-1) * a))
    G = np.matrix(G)

    h = np.vstack((C * np.ones([n,1]), np.zeros([n,1])))
    h = np.matrix(h)
            
    return Q, p, G, h, A, b

