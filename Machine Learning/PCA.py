import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Part a
def PCA(d):
    X = scipy.io.loadmat('USPS.mat')["A"]
    y = scipy.io.loadmat('USPS.mat')["L"]
    x = X[0, :]


    # Normalize data to have zero-mean and unit-variance
    X_nomal = StandardScaler().fit_transform(X)

    # Covariance matrix
    cov_x = np.cov(X_nomal.T)

    # Compute Eigen value and Eigen vector
    egvau, egvec = np.linalg.eig(cov_x)
    e_ind_order = np.flip(egvau.argsort())
    e_val = egvau[e_ind_order]
    e_vec = egvec[:,e_ind_order ]

    # d = 0:10 10, 50, 100, 200
    vec = e_vec[:, 0:d]

    # Reconstructiuon matrix and show the picture
    recon_matrix = vec@(vec.T@x)
    recon_matrix_show = np.reshape(np.array(recon_matrix), (16, 16))
    plt.imshow(recon_matrix_show)
    plt.title("Reconstruct images using principal components d = " + str(d))
    plt.show()


    # Entire Reconstructiuon
    recon_data = vec@(vec.T@X.T)

    # Calculate error
    err_tmp = (X.T - recon_data)
    error = np.linalg.norm(err_tmp, ord='fro')

    print("Dataset d = ", d," principal components", error)




def main():

    # Part b
    d = [10, 50, 100, 200]
    for i in d:
        PCA(i)

if __name__ == '__main__':
    main()
