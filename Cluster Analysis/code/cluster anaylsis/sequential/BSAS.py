import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sys import maxsize as max_integer
from functools import reduce
from tqdm import tqdm

euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))

def basic_sequential_scheme(data, max_number_of_clusters = 10, threshold = max_integer):
    ''' An implementation of the basic sequential scheme clustering algorithm.
    
    Parameters:
        data((m x n) 2-d numpy array): a data set of m instances and n features
        max_number_of_clusters(integer): the maximum allowable number of clusters
    
    Returns:
        clustered_data((m x (n + 1)) 2-d numpy array): the data set with one more column that contains the vector's cluster
        centroids((k x n)2-d numpy array): contains the k = no_of_clusters centroids with n features
        no_of_clusters(integer): the final number of clusters created
    
    '''
    # Initializations
    initial_shape = list(data.shape)
    N = reduce(lambda x, y: x * y, data.shape[:-1]) 
    m = data.shape[-1] 
    
    # No matter what is the dimensions of the input data array, we convert it to 2-D array, we implement the algorithm and then we turn it back to its
    # original dimensions 
    data = data.reshape(N, m)

    # Automatically calculating threshold by peaks and valleys technique
    if threshold == max_integer:
        threshold, _ = thresholding_BSAS(data)
    print('threshold value: ', threshold)
    
    # If it remained max_integer
    if threshold == max_integer: # in the rare case when it fails to find a unique threshold, if there aren't two peaks
        return None, 0, 0
    
    # We keep two copies of the data matrix, one with the cluster column
    clusters = np.zeros((N, 1))
    clusters.fill(-1)
    clustered_data = np.concatenate((data, clusters), axis = 1)
    
    # Assign the first point to this cluster 
    cluster_index = 0
    clustered_data[0, m] = 0
    centroids = np.array([data[0]])
    
    vectors_in_clusters_array = np.zeros(max_number_of_clusters)
    vectors_in_clusters_array[0] = 1
    
    for i, vector in tqdm(enumerate(data[1:], start = 1)):
        distance_from_centroids = euclidean_distance(centroids, vector)
        nearest_cluster_distance = np.min(distance_from_centroids)
        nearest_cluster = np.argmin(distance_from_centroids)
        if nearest_cluster_distance > threshold and cluster_index < max_number_of_clusters - 1:
            cluster_index += 1
            clustered_data[i, m] = cluster_index
            # Add new Centroid
            centroids = np.concatenate((centroids, [data[i]]), axis = 0)
            vectors_in_clusters_array[cluster_index] = 1
        else:
            clustered_data[i, m] = nearest_cluster
            # Update Centroids
            
            #vectors_in_cluster = len(np.where(clustered_data[:, m] == nearest_cluster)[0])
            vectors_in_clusters_array[nearest_cluster] += 1
            vectors_in_cluster = vectors_in_clusters_array[nearest_cluster]
            centroids[nearest_cluster] = ((vectors_in_cluster - 1) * centroids[nearest_cluster] + vector) /vectors_in_cluster
    
    # Reassignment procedure
    for i, d in enumerate(data):
        distance_from_centroids = euclidean_distance(centroids, d)
        nearest_cluster_distance = np.min(distance_from_centroids)
        nearest_cluster = np.argmin(distance_from_centroids)
        clustered_data[i, m] = nearest_cluster
    
    
    final_clusters = np.unique(clustered_data[:, m])
    for j in final_clusters:
        # Update Centroids
        indices_of_current_centroid = np.where(clustered_data[:, m] == j)[0]
        centroids[j] = np.mean(data[indices_of_current_centroid, :], axis = 0)

    # Return data matrix back to its original dimensions taking under consideration the one extra column for the cluster id
    initial_shape[-1] += 1
    clustered_data = clustered_data.reshape(initial_shape)

    return clustered_data, centroids, cluster_index + 1



def thresholding_BSAS(data):
    ''' A function to calculate the value of the threshold by using the peaks and valleys technique
    
    Parameters:
        data((m x n) 2-d numpy array): a data set of m instances and n features

    Returns:
        deepest_valley(float): the height of the histogram at the point of the deepest valley
                               between the two highest peeks. It is actually the threshold value
    
    '''
    # Construct the dissimilarity matrix
    N = len(data)
    m = len(data[0])
    
    # If the dataset is less than 5000 vectors, calculate the threshold on the whole dataset. Otherwise,
    # calculate it only on the 1% of it.
    if N > 5000:
        uniformingly_random_data = np.random.randint(N, size = (int(N/100)))
        n = len(uniformingly_random_data)
        
        dissimilarity_matrix = np.empty((n, n)) 
        summary_array = data[uniformingly_random_data]
        
        for i, point in enumerate(summary_array):
            dissimilarity_matrix[:, [i]] = euclidean_distance(summary_array[:, :m],point[:m])
        distances = np.zeros((n * (n - 1)/2)) #number of pairs
    
    else:
        dissimilarity_matrix = np.empty((N, N)) 
        for i, point in enumerate(data):
            dissimilarity_matrix[:, [i]] =  euclidean_distance(data[:, :m],point[:m])
        
        distances = np.zeros((N * (N - 1)/2)) #number of pairs
    
    
    
    k = 0
    for i, row in enumerate(dissimilarity_matrix):
        temp_data = row[(i + 1):]
        distances[k: k + len(temp_data)] = temp_data
        k += len(temp_data)
    
    n, bins  = np.histogram(distances, bins = 50) # calculating, not plotting
    
    # Peak and valley seeking
    all_peaks_indices = argrelextrema(n, np.greater)[0]
    all_peaks_values = n[all_peaks_indices]
    # Sorting an array with regards to another
    sorted_list_of_peaks_indices = [index for value, index in sorted(zip(all_peaks_values, all_peaks_indices))]
    two_largest_peaks = sorted_list_of_peaks_indices[-2:]
    temp = sorted(two_largest_peaks)
    
    # If we were not able to find two peaks exit and let the calling functions decide what to do
    if len(temp) < 2:
        return max_integer, 0

    deepest_valley_bin = np.argmin(n[temp[0]:temp[1] + 1]) + temp[0]
    deepest_valley = bins[deepest_valley_bin]
  
    return deepest_valley, bins
    
    
    
    
    
    