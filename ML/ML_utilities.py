import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD


def apply_compression(input_data, err_tolerance, alg):

    """ Compresses the input data based on the specified algorithm
        
        Inputs -

        input_data (np.array) - the input data 
        err_tolerance (float) - the threshold for selecting optimal no. of components
        alg (str) - to specify the type of compression algorithm (PCA or TruncatedSVD)
        
        Outputs -
        
        n_comp_opt (int) - the no. of optimal components at which the difference of successive NMSEs b/w 
                           orignal & the reconstructed signal falls below the specified threshold
        compressed_data (np.array) - the compressed data 
        Method - an instance of the fitted algorithm used for compression """

    prev_err = 0

    for n_comp in range(10, input_data.shape[1]):

        if alg.lower()=='pca':
            Method = PCA(n_comp)
        elif alg.lower()=='svd':
            Method = TruncatedSVD(n_comp)
        
        compressed_data = Method.fit_transform(input_data)

        err = nmse(input_data, Method.inverse_transform(compressed_data))
        n_comp_opt = n_comp

        if np.abs(err - prev_err) < err_tolerance:
            break

        prev_err = err
    
    return n_comp_opt, compressed_data, Method


def load_pkl_dat(filename):

    """ Reads the contents of a .pkl file & returns the data as a numpy array """

    # Load data from pickle file
    data = []

    with open(filename, 'rb') as f:
        try:
            while True:
                data.append(pickle.load(f))
        except EOFError:
            pass

    return np.array(data)


def nmse(signal1, signal2):

    """ Returns the Normalized Mean Squared Error b/w two signals """

    return np.linalg.norm(signal2 - signal1)**2 / np.linalg.norm(signal1)**2


def plot_pred_dat(test_y, y_pred_final, batch_err, nmse):

    """ Utility function for plotting ML predictions """

    for i in range(0,test_y.shape[0],250):
        
        orig_A_scan = test_y[i,:]
        pred_A_scan = y_pred_final[i,:]
        err = nmse(orig_A_scan, pred_A_scan)
        batch_err.append(err)

        plt.subplots(1,1, figsize=(10,5))
        plt.title("{}:  NMSE = {}".format(i, np.around(err, decimals=10)))
        plt.plot(test_y[i,:], '-b', label='Orignal A-scan')
        plt.plot(pred_A_scan , '--r', label='Predicted A-scan')
        plt.legend()

    print("Batch Error =", np.mean(batch_err))

