import numpy as np
from functools import reduce

euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))

def possibilistic(data, no_of_clusters, ita, centroids_initial = None, q = 1.25):
    ''' An implementation of the possibilistic clustering algorithm
    
    Parameters:
        data((m x n) 2-d numpy array): a data set of m instances and n features
        no_of_clusters(integer): the number of clusters
        ita(list): contains the values of the ita parameter for every cluster
        centroids_initial(): the optional initial values for the centroids
        q(float): fuzzifier parameter
    
    Returns:
        data((m x (n + 1)) 2-d numpy array): the data set with one more column that contains the vector's cluster
        centroids_new((k x n)2-d numpy array): contains the k = no_of_clusters centroids with n features
        centroids_history((l x 2) 2-d numpy array): an array to keep the previous positions of the centroids for 
                                                    better visualisation of the result. 
        typicality_matrix ((n x 2) 2-d numpy array): the matrix containing the weights which depict the typicality
                                                     of a vector i to the cluster j 
    
    '''
    # Initializations
    initial_shape = list(data.shape)
    N = reduce(lambda x, y: x * y, data.shape[:-1]) 
    m = data.shape[-1] 
    typicality_matrix = np.zeros((N, no_of_clusters))
    # Conversion to 2-D array
    data = data.reshape(N, m)

    
    if centroids_initial is None:
        centroids_old = np.random.choice(np.arange(np.min(data), np.max(data), 0.1), size = (no_of_clusters, m), replace = False)
    else:
        centroids_old = centroids_initial
    centroids_new = np.zeros(centroids_old.shape) 
    centroids_history = np.copy(centroids_old) # this array stacks the old positions of the centroids
    
    # A do - while loop implementation in Python, as the loop needs to run at least once
    condition = True
    counter = 1
    
    
    while condition and counter < 50:
        counter += 1
        print(counter)
        # Update the U matrix

        for i in range(no_of_clusters):
            #precalculate euclidean distances for the instance. instance is the point, centroids are the data
            eucl_dist = euclidean_distance(data, centroids_old[i, :])
            typicality_matrix[:, [i]] = 1 / (1 + np.power(eucl_dist/ita[i], (1/(q-1))))
        
        ''' The loop below was replaced by the loop above for optimization reasons
        for i in range(N):
            eucl_dist = euclidean_distance(centroids_old, data[i,:])
            for j in range(no_of_clusters):
                typicality_matrix[i][j] = 1 / (1 + np.power(eucl_dist[j][0]/ita[j], (1/(q-1))))
        '''
            
        #update the centroids
        for i, centroid in enumerate(centroids_old):
            centroids_new[i] = np.sum(np.power(typicality_matrix[:,[i]], q) * data,axis = 0) / np.sum(np.power(typicality_matrix[:,i], q))
        
        # Update the termination criterion where e = 0.00001
        criterion_array = np.absolute(centroids_new - centroids_old) < 0.00001
        if np.all(criterion_array) :
            condition = False
        
        centroids_old = np.copy(centroids_new)
        centroids_history = np.vstack((centroids_history, centroids_old))
    
    
    
    # Assign each vector to a cluster taking the greatest u
    assigned_cluster = np.argmax(typicality_matrix, axis = 1) 
    data = np.hstack((data, assigned_cluster.reshape(N, 1)))
    
    initial_shape[-1] += 1
    data = data.reshape(initial_shape)
    
    return data, centroids_new, centroids_history, typicality_matrix
    
    
    
    
    

    