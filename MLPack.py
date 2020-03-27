import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def KMeansClusterer1D(values, n, init_mu, n_iter = 10 ):
    
    '''
	
	Performs K-Means Clustering on uni-dimensional data.

    values : 1D Vector containing values to cluster
    n : Number of clusters to make
    init_mu : list containing 'n' values for initial centres
    
    RETURNS:
    new_mu : list of new clusters location
    radius : list of cluster radius
    '''
    
    assert len(init_mu) == n
    
    
    for i in range(n_iter):
        
        val_mu_diff = np.reshape(np.power(values - init_mu[0], 2), [-1,1])
        #To calculate x_i - mu_c 
        ## and
        #To calculate cluster which minimizes square of val_mu_diff
        for j in range(1,n):
            val_mu_diff = np.concatenate([ val_mu_diff, np.reshape(np.power(values - init_mu[j], 2), [-1,1]) ], axis = 1)
            
        c = np.argmin(val_mu_diff, axis = 1)
        
        #Updating Centroids
        new_mu = []
        for j in range(n):
            
            temp = values[c == j]
            new_mu.append(np.sum(temp) / len(temp))
        
        init_mu = new_mu
        
    
    #To calculate Radius of each cluster
    radius = []
    for j in range(n):
        temp = values[c == j]
        radius.append(np.abs(np.max(temp) - new_mu[j]))        
        
    return new_mu, radius