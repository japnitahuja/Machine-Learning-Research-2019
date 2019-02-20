import numpy as np
from tqdm import tqdm
from cost_function_optimization import fuzzy_clustering, possibilistic_clustering, kmeans_clustering
from sys import maxsize as max_integer
import matplotlib.pyplot as plt
from utility.plotting_functions import *
from sequential import BSAS, TTSS
from graph_theory import MST
from functools import reduce


euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))

def relative_validity_hard_sequential(X):
    ''' Defines the several values of the BSAS parameter. Then conducts successive executions of the algorithm by passing to it 
        those values and calculates all the proper relative indices.
        
        Parameters:
            X((N x m) numpy array): a data set of N instances and m features
        
        Returns:
            no_of_threshold_values: the different values of the threshold parameter 
            DI, DB, SI: the arrays holding the values of the relative indices
    '''
    # Initialization
    threshold, bins = BSAS.thresholding_BSAS(X)
    threshold_index = np.where(bins == threshold)[0][0]
    
    # Finds the threshold values against which to run the BSAS algorithm
    number_of_threshold_values = 10
    if threshold_index >= number_of_threshold_values:
        no_of_threshold_values = [bins[i] for i in range(threshold_index - number_of_threshold_values, min(threshold_index + number_of_threshold_values, len(bins) - 1))]
    else:
        no_of_threshold_values = [bins[i] for i in range(0, threshold_index + threshold_index)]
    
    DI = np.zeros(len(no_of_threshold_values))
    DB = np.zeros(len(no_of_threshold_values))
    SI = np.zeros(len(no_of_threshold_values))


    for i, threshold_values in tqdm(enumerate(no_of_threshold_values)): # no_of_clusters
    
        X_, centroids_BSAS, total_clusters_ = BSAS.basic_sequential_scheme(X, threshold = threshold_values)
        
        DI[i] = Dunn_index(X_)
        DB[i] = Davies_Bouldin(X_, centroids_BSAS)
        SI[i] = silhouette_index(X_)
    
    return no_of_threshold_values, DI, DB, SI





def relative_validity_TTSS(X):
    ''' Defines the several values of the TTSS parameters. Then conducts successive executions of the algorithm by passing to it 
        those values and calculates all the proper relative indices.
        
        Parameters:
            X((N x m) numpy array): a data set of N instances and m features
        
        Returns:
            no_of_threshold_values1: the different values of the threshold1 parameter 
            no_of_threshold_values2: the different values of the threshold2 parameter 
            DI, DB, SI: the arrays holding the values of the relative indices
    '''
    # Initialization
    threshold1_value, threshold2_value, bins = TTSS.thresholding_TTSS(X)
    threshold_index1 = np.where(bins == threshold1_value)[0][0]
    threshold_index2 = np.where(bins == threshold2_value)[0][0]
    
    # Finds the threshold values against which to run the BSAS algorithm
    range_of_threshold_values = 4
    if threshold_index1 >= range_of_threshold_values:
        no_of_threshold_values1 = [bins[i] for i in range(threshold_index1 - range_of_threshold_values, min(threshold_index1 + range_of_threshold_values, len(bins) - 1))]
    else:
        range_of_threshold_values = threshold_index1 - 1
        no_of_threshold_values1 = [bins[i] for i in range(threshold_index1 - range_of_threshold_values, min(threshold_index1 + range_of_threshold_values, len(bins) - 1))]
    
    range_of_threshold_values = 6
    if threshold_index2 >= range_of_threshold_values:
        no_of_threshold_values2 = [bins[i] for i in range(threshold_index2 - range_of_threshold_values, min(threshold_index2 + range_of_threshold_values, len(bins) - 1))]
    else:
        range_of_threshold_values = threshold_index1 - 1
        no_of_threshold_values2 = [bins[i] for i in range(threshold_index2 - range_of_threshold_values, min(threshold_index2 + range_of_threshold_values, len(bins) - 1))]
    
    
    N = reduce(lambda x, y: x * y, X.shape[:-1]) 
    m = X.shape[-1] 

    # Conversion to 2-D array
    X = X.reshape(N, m)
    
    # Initialize arrays to hold the indices. We use separate arrays for easier modification of the code if needed.
    # If we wanted to use one array then this would be a 3 - dimensional array.
    DI = np.zeros((len(no_of_threshold_values1), len(no_of_threshold_values2)))
    DB = np.zeros((len(no_of_threshold_values1), len(no_of_threshold_values2)))
    SI = np.zeros((len(no_of_threshold_values1), len(no_of_threshold_values2)))
    
    for i, threshold_v1 in tqdm(enumerate(no_of_threshold_values1)): # no_of_clusters
        for j, threshold_v2 in enumerate(no_of_threshold_values2): 
            
            if threshold_v2 <= threshold_v1:
                DI[i, j] = np.nan
                DB[i, j] = np.nan
                SI[i, j] = np.nan
            else:
                # When X returns it has one more column that needs to be erased
                X_, centroids, no_of_clusters = TTSS.two_threshold_sequential_scheme(X, threshold1 = threshold_v1, threshold2 = threshold_v2)
                
                #plot_data(X_, no_of_clusters, centroids)
                
                #plt.show()
                
                # Calculate indices
                DI[i, j] = Dunn_index(X_)
                DB[i, j] = Davies_Bouldin(X_, centroids)
                SI[i, j] = silhouette_index(X_)
            
    return no_of_threshold_values1, no_of_threshold_values2, DI, DB, SI




def relative_validity_hard_graph(X):
    ''' Defines the several values of the MST parameters. Then conducts successive executions of the algorithm by passing to it 
        those values and calculates all the proper relative indices.
        
        Parameters:
            X((N x m) numpy array): a data set of N instances and m features
        
        Returns:
            no_of_k_list: the different values of the k parameter 
            no_of_f_list: the different values of the f parameter 
            DI, SI: the arrays holding the values of the relative indices
    '''
    # Initialization
    no_of_k_list = [i for i in np.linspace(2, 10, 9)]
    no_of_f_list = [i for i in np.linspace(1.5, 3.5, 6)]
    
    DI = np.zeros((len(no_of_k_list), len(no_of_f_list)))
    SI = np.zeros((len(no_of_k_list), len(no_of_f_list)))

    for i, k_value in tqdm(enumerate(no_of_k_list)):
        for j, f_value in enumerate(no_of_f_list): 
            
            # When X returns it has one more column that needs to be erased
            X_, no_of_clusters = MST.minimum_spanning_tree(X, k = k_value, f = f_value)
            
            # Calculate indices
            DI[i, j] = Dunn_index(X_)
            SI[i, j] = silhouette_index(X_)

    return no_of_k_list, no_of_f_list, DI, SI



def relative_validity_hard_large_data(X):
    # Initialization
    no_of_clusters_list = [i for i in range(2, 11)]
    
    DB = np.zeros(len(no_of_clusters_list))
    
    # Centroids must remain the same. The only parameter that should change is the number of clusters 
    clustered_data, centroids_BSAS, total_clusters_ = BSAS.basic_sequential_scheme(X)

    for i, total_clusters in tqdm(enumerate(no_of_clusters_list)): # no_of_clusters
        
        if len(centroids_BSAS) < total_clusters:
            centroids = np.zeros((total_clusters, len(X[0])))
            # First centroids values
            centroids[:len(centroids_BSAS),:] = centroids_BSAS 
            # Last centroids values
            random_indices = np.random.randint(len(X),size = total_clusters - len(centroids_BSAS))
            centroids[len(centroids_BSAS):,:] = X[random_indices, :]
        elif len(centroids_BSAS) > total_clusters:
            centroids = centroids_BSAS[:total_clusters, :]
        elif len(centroids_BSAS) == total_clusters:
            centroids = centroids_BSAS
        
        X_, centroids, centroids_history = kmeans_clustering.kmeans(X, total_clusters, centroids_initial = centroids)
        
        DB[i] = Davies_Bouldin(X_, centroids)

    return no_of_clusters_list, DB










def relative_validity_hard(X):
    ''' Defines the several values of the kmeans parameter. Then conducts successive executions of the algorithm by passing to it 
        those values and calculates all the proper relative indices.
        
        Parameters:
            X((N x m) numpy array): a data set of N instances and m features
        
        Returns:
            no_of_clusters_list: the different values of the clusters number
            DI, DB, SI, GI: the arrays holding the values of the relative indices
    '''
    # Initialization
    no_of_clusters_list = [i for i in range(2, 11)]
    
    DI = np.zeros(len(no_of_clusters_list))
    DB = np.zeros(len(no_of_clusters_list))
    SI = np.zeros(len(no_of_clusters_list))
    GI = np.zeros(len(no_of_clusters_list))
    
    # Centroids must remain the same. The only parameter that should change is the number of clusters 
    clustered_data, centroids_BSAS, total_clusters_ = BSAS.basic_sequential_scheme(X)

    for i, total_clusters in tqdm(enumerate(no_of_clusters_list)): # no_of_clusters
        
        if len(centroids_BSAS) < total_clusters:
            centroids = np.zeros((total_clusters, len(X[0])))
            # First centroids values
            centroids[:len(centroids_BSAS),:] = centroids_BSAS 
            # Last centroids values
            random_indices = np.random.randint(len(X),size = total_clusters - len(centroids_BSAS))
            centroids[len(centroids_BSAS):,:] = X[random_indices, :]
        elif len(centroids_BSAS) > total_clusters:
            centroids = centroids_BSAS[:total_clusters, :]
        elif len(centroids_BSAS) == total_clusters:
            centroids = centroids_BSAS
        
        X_, centroids, centroids_history = kmeans_clustering.kmeans(X, total_clusters, centroids_initial = centroids)
        
        
        DI[i] = Dunn_index(X_)
        DB[i] = Davies_Bouldin(X_, centroids)
        SI[i] = silhouette_index(X_)
        GI[i] = gap_index(X_, total_clusters, kmeans_clustering.kmeans)
    
    return no_of_clusters_list, DI, DB, SI, GI


def relative_validity_fuzzy(X):
    ''' Defines the several values of the fuzzy parameter. Then conducts successive executions of the algorithm by passing to it 
        those values and calculates all the proper relative indices.
        
        Parameters:
            X((N x m) numpy array): a data set of N instances and m features
        
        Returns:
            no_of_clusters_list: the different values of the clusters number
            values_of_q: the different values of the q parameter
            PC, PE, XB, FS: the arrays holding the values of the relative indices
    '''
    # Initialization
    no_of_clusters_list = [i for i in range(2, 11)]
    values_of_q = [1.25, 1.5, 1.75]
    
    N = reduce(lambda x, y: x * y, X.shape[:-1]) 
    m = X.shape[-1] 

    # Conversion to 2-D array
    X = X.reshape(N, m)
    
    # Initialize arrays to hold the indices. We use separate arrays for easier modification of the code if needed.
    # If we wanted to use one array then this would be a 3 - dimensional array.
    PC = np.zeros((len(no_of_clusters_list), len(values_of_q)))
    PE = np.zeros((len(no_of_clusters_list), len(values_of_q)))
    XB = np.zeros((len(no_of_clusters_list), len(values_of_q)))
    FS = np.zeros((len(no_of_clusters_list), len(values_of_q)))
    #cost = np.zeros((len(no_of_clusters_list), len(values_of_q)))
    
    for i, total_clusters in tqdm(enumerate(no_of_clusters_list)): # no_of_clusters
        # IMPORTANT: The centroids must remain the same for every run of the algorithm with the same no_of_clusters
        centroids_initial = np.random.choice(np.arange(np.min(X), np.max(X), 0.1), size = (total_clusters, len(X[0])), replace = False)
        
        for j, q_value in enumerate(values_of_q): #edw vazw to q

            # When X returns it has one more column that needs to be erased
            X_, centroids, ita, centroids_history, partition_matrix = fuzzy_clustering.fuzzy(X, total_clusters, centroids_initial, q = q_value)

            # Calculate indices
            PC[i, j] = partition_coefficient(X, partition_matrix)
            PE[i, j] = partition_entropy(X, partition_matrix)
            XB[i, j] = Xie_Beni(X, centroids, partition_matrix)
            FS[i, j] = fukuyama_sugeno(X, centroids, partition_matrix, q = 2)   
            #cost[i, j] = cost_calc(X, partition_matrix, centroids)
            
    return no_of_clusters_list, values_of_q, PC, PE, XB, FS


# measuring the total cost
def cost_calc(X, partition_matrix, centroids):
    cost = 0
    for i, d in enumerate(X):
        for j, k in enumerate(centroids):
            cost += partition_matrix[i, j] * euclidean_distance(d[:2].reshape(1,2), k) 
    return cost


def relative_validity_possibilistic(X):
    ''' Defines the several values of the possibilistic parameter. Then conducts successive executions of the algorithm by passing to it 
        those values and calculates all the proper relative indices.
        
        Parameters:
            X((N x m) numpy array): a data set of N instances and m features
        
        Returns:
            no_of_clusters_list: the different values of the clusters number
            values_of_q: the different values of the q parameter
            PC, PE, XB, FS: the arrays holding the values of the relative indices
    '''
    # Initialization
    no_of_clusters_list = [i for i in range(2, 11)]
    values_of_q = [1.25, 1.5, 1.75]
    
    N = reduce(lambda x, y: x * y, X.shape[:-1]) 
    m = X.shape[-1] 

    # Conversion to 2-D array
    X = X.reshape(N, m)
    
    # Initialize arrays to hold the indices. We use separate arrays for easier modification of the code if needed.
    # If we wanted to use one array then this would be a 3 - dimensional array.
    PC = np.zeros((len(no_of_clusters_list), len(values_of_q)))
    PE = np.zeros((len(no_of_clusters_list), len(values_of_q)))
    XB = np.zeros((len(no_of_clusters_list), len(values_of_q)))
    FS = np.zeros((len(no_of_clusters_list), len(values_of_q)))
    
    for i, total_clusters in tqdm(enumerate(no_of_clusters_list)): # no_of_clusters
        # IMPORTANT: The centroids must remain the same for every run of the algorithm with the same no_of_clusters
        centroids_initial = np.random.choice(np.arange(np.min(X), np.max(X), 0.1), size = (total_clusters, len(X[0])), replace = False)
        
        for j, q_value in enumerate(values_of_q): #edw vazw to q

            # When X returns it has one more column that needs to be erased
            X_, centroids, ita, centroids_history, partition_matrix = fuzzy_clustering.fuzzy(X, total_clusters, centroids_initial, q = q_value)
            X_, centroids, centroids_history, typicality_matrix = possibilistic_clustering.possibilistic(X, total_clusters, ita, centroids_initial = centroids, q = q_value)


            # Calculate indices
            # In order to calculate indices we pass them the normalized typicality matrix, see Yang, Wu, Unsupervised possibilistic learning
            
            typicality_matrix_norm = typicality_matrix / np.sum(typicality_matrix, axis = 1).reshape(len(X), 1)
            
            PC[i, j] = partition_coefficient(X, typicality_matrix_norm)
            PE[i, j] = partition_entropy(X, typicality_matrix_norm)
            XB[i, j] = Xie_Beni(X, centroids, typicality_matrix_norm)
            FS[i, j] = fukuyama_sugeno(X, centroids, typicality_matrix_norm, q = 2)   
            
    return no_of_clusters_list, values_of_q, PC, PE, XB, FS





###############################################################################################

def Dunn_index(X):
    ''' Calculates the Dunn index of a clustered dataset.
    
        Parameters: 
        X((N x m + 1) numpy array): a clustered data set of N instances, m features and the cluster id at the last column of each vector
        
        Returns:
        The Dunn index
    '''
    

    N = len(X)
    m = len(X[0]) - 1
    clusters = np.unique(X[:, m])
    if len(clusters) <= 1: 
        return np.nan

    min_cluster_distance = max_integer
    max_cluster_diameter = -max_integer - 1
    
    # Construct the dissimilarity matrix
    dissimilarity_matrix = np.empty((N, N)) 
    for j, point in enumerate(X):
        dissimilarity_matrix[:, [j]] = euclidean_distance(X[:, :m], point[:m])
    
    for i, cluster1 in enumerate(clusters):
        # Calculate the diameter of each cluster
        first_cluster_indices = np.where(X[:, m] == cluster1)[0]
        # Number of vectors in cluster
        n = len(first_cluster_indices)
        temp1 = np.max(dissimilarity_matrix[first_cluster_indices.reshape(n, 1), first_cluster_indices])
        if max_cluster_diameter < temp1:
            max_cluster_diameter = temp1
        
        for j, cluster2 in enumerate(clusters[(i+1):], start = i + 1):
            # Calculate the distances between the clusters
            second_cluster_indices = np.where(X[:, m] == cluster2)[0]
            # The reshape creates a n x 1 2-d array which is very important for the indexing of the dissimilarity matrix
            temp2 = np.min(dissimilarity_matrix[first_cluster_indices.reshape(n, 1), second_cluster_indices])
            if min_cluster_distance > temp2:
                min_cluster_distance = temp2
    # Dunn index is the minimum distance between clusters divided by the maximum diameter
    return min_cluster_distance/max_cluster_diameter


def Davies_Bouldin(X, centroids):
    ''' Calculates the Davies Bouldin index of a clustered dataset. Whereas in Dunn index the distance between clusters is 
        the distance between the closest vectors of the clusters, in Davies Bouldin the same distance is the 
        distance between the centroids.
    
        Parameters: 
        X((N x m + 1) numpy array): a clustered data set of N instances, m features and the cluster id at the last column of each vector
        centroids: The centroids returned from the clustering algorithm
        
        Returns:
        The Davies Bouldin index
    '''
    # If a centroids has not been used, the index is implemented in such a way that it is skipped
    
    # Initializations

    N = reduce(lambda x, y: x * y, X.shape[:-1]) 
    m = X.shape[-1] 
    
    # No matter what is the dimensions of the input data array, we convert it to 2-D array, we implement the algorithm and then we turn it back to its
    # original dimensions 
    X = X.reshape(N, m)
    
    clusters = np.unique(X[:, m - 1])
    if len(clusters) <= 1: return np.nan #TODO: change with np.nan
    
    no_of_clusters = len(clusters)
    # Casting the clusters array to int as we are going to use it later for indexing
    clusters = clusters.astype(int)
    
    # Create a 1-D matrix to hold the values of each cluster's dispersion
    cluster_dispersion = np.zeros((no_of_clusters))
    # Create a dissimilarity matrix to hold the distances between the clusters' centroids
    cluster_distances = np.zeros((no_of_clusters, no_of_clusters))
    
    # Calculate dispersion values and clusters' distances in one loop
    for i, cluster in enumerate(clusters): 
        temp = np.sum(np.power(X[np.where(X[:, m - 1] == cluster)[0], :m - 1] - centroids[cluster], 2))
        cluster_dispersion[i] = np.sqrt(1/N * temp)
        # Calculate clusters' distances
        cluster_distances[i, :] = euclidean_distance(centroids[clusters, :], centroids[cluster]).reshape(1, len(clusters))
    
    # Create a matrix to hold the similarity indices between clusters
    R = np.zeros((len(clusters), len(clusters)))
    for i, _ in enumerate(clusters):
        for j, _ in enumerate(clusters[(i+1):], start = i + 1):
            R[i, j] =  (cluster_dispersion[i] + cluster_dispersion[j]) / cluster_distances[i, j] 
    
    DB = np.average(np.amax(R, axis = 1))
    
    return DB

        

def silhouette_index(X):
    ''' Calculates the silhouette index of a clustered dataset. 
    
        Parameters: 
        X((N x m + 1) numpy array): a clustered data set of N instances, m features and the cluster id at the last column of each vector
        
        Returns:
        The silhouette index
    '''
    N= len(X)
    m = len(X[0]) - 1
    clusters = np.unique(X[:, m])
    if len(clusters) <= 1: return np.nan #TODO: change with np.nan
    # Construct the dissimilarity matrix
    dissimilarity_matrix = np.empty((N, N)) 
    for j, point in enumerate(X):
        dissimilarity_matrix[:, [j]] = euclidean_distance(X[:, :m], point[:m])
    
    
    # a: average_distance_in_same_cluster. Average distance only for the vectors belonging to the same clusters
    a = np.zeros((N))
    # Calculates the silhouettes of all clusters as the average silhouettes of their vectors
    for i, cluster in enumerate(clusters):
        cluster_indices = np.where(X[:, m] == cluster)[0]
        # Number of vectors in the cluster
        n = len(cluster_indices)
        cluster_dissimmilarity_matrix = dissimilarity_matrix[cluster_indices.reshape(n, 1), cluster_indices]
            
        for j, vector_index in enumerate(cluster_indices):
            if n != 1:
                a[vector_index] = np.sum(cluster_dissimmilarity_matrix[j, :], axis = 0)/(n - 1) #average. not counting distance 0 to itself
            else:
                a[vector_index] = np.sum(cluster_dissimmilarity_matrix[j, :], axis = 0)
    
    
    #  b: average_distance_in_closest_cluster. Average distance for vectors belonging to the closest cluster
    b = np.zeros((N))
    b.fill(max_integer)
    # Calculates b
    for i, cluster1 in enumerate(clusters):
        cluster_indices1 = np.where(X[:, m] == cluster1)[0]
        # Number of vectors in the cluster
        n = len(cluster_indices1)
        for j, cluster2 in enumerate(clusters):
            if cluster1 != cluster2:
                cluster_indices2 = np.where(X[:, m] == cluster2)[0]
                k = len(cluster_indices2)
                different_cluster_dissimmilarity_matrix = dissimilarity_matrix[cluster_indices1.reshape(n, 1), cluster_indices2]
                
                for j, vector_index in enumerate(cluster_indices1):
                    if b[vector_index] > np.average(different_cluster_dissimmilarity_matrix[j, :], axis = 0):
                        b[vector_index] = np.average(different_cluster_dissimmilarity_matrix[j, :], axis = 0)
        
                
    # Calculates the silhouette width of every vector
    vector_silhouette_width = (b - a)/np.amax((b,a), axis = 0)
    
    # Calculates the silhouette width of every cluster
    cluster_silhouette_width = np.zeros(len(clusters))
    for i, cluster in enumerate(clusters):
        cluster_indices = np.where(X[:, m] == cluster)[0]
        cluster_silhouette_width[i] = np.average(vector_silhouette_width[cluster_indices])
    
    # Calculates the global silhouette index
    global_silhouette_index = np.average(cluster_silhouette_width)
    
    return global_silhouette_index


def _gap_index_calculation(X):
    ''' Calculates the log(W) for the provided dataset.
    
        Parameters: 
        X((N x m + 1) numpy array): a clustered data set of N instances, m features and the cluster id at the last column of each vector
        
        Returns:
        The log(W)
    '''
    N =len(X)
    m = len(X[0]) - 1
    # Construct the dissimilarity matrix
    dissimilarity_matrix = np.empty((N, N)) 
    for j, point in enumerate(X):
        dissimilarity_matrix[:, [j]] = euclidean_distance(X[:, :m], point[:m])
    
    clusters = np.unique(X[:, m])
    # Calculate the sum of the distances between all pairs for each cluster
    D = np.zeros((len(clusters)))

    W = 0.
    for i, cluster in enumerate(clusters):
        cluster_indices = np.where(X[:, m] == cluster)[0]
        n = len(cluster_indices)
        cluster_dissimmilarity_matrix = dissimilarity_matrix[cluster_indices.reshape(n, 1), cluster_indices]
        
        D[i] = np.sum(cluster_dissimmilarity_matrix)
        
        W += 1/(2 * n) * D[i]
    
    return np.log(W)


def gap_index(X, no_of_clusters, algorithm):
    ''' Calculates the Gap index of a clustered dataset.
    
        Parameters: 
        X((N x m + 1) numpy array): a clustered data set of N instances, m features and the cluster id at the last column of each vector
        no_of_clusters: the number of clusters
        algorithm: the function object representing the algorithm that called the function
        
        Returns:
        The Gap index
    '''
    log_W = _gap_index_calculation(X)
    # Create an array to hold the logW values of the 100 monte carlo simulations
    log_W_sample = np.zeros((100))
    
    N =len(X)
    m = len(X[0]) - 1
    # Monte Carlo simulation - create the datasets (random position hypothesis)
    for i in range(100):
        random_data = np.empty((N, m))
        
        for j in range(m):
            max_value = np.amax(X[:, j])
            min_value = np.min(X[:, j])
            temp = (max_value - min_value) * np.random.random(size = (N, 1)) + min_value
            random_data[:, [j]] = temp
            
        if algorithm == kmeans_clustering.kmeans:
            X_, centroids, centroids_history = kmeans_clustering.kmeans(random_data, no_of_clusters)

            
        log_W_sample[i] = _gap_index_calculation(X_)

            
    Gap = np.average(log_W_sample) - log_W
    
    return Gap        
            
################################ fuzzy indices #######################    

# Lambda functions in order to calculate the same name indices
partition_coefficient = lambda X, partition_matrix: np.round(1/len(X) * np.sum(np.power(partition_matrix, 2)), 5)
partition_entropy = lambda X, partition_matrix: - 1/len(X) * np.sum(partition_matrix * np.log(partition_matrix)) 




def Xie_Beni(X, centroids, partition_matrix):
    ''' Calculates the Xie Beni index.
    
    Parameters:
        X((N x m + 1) numpy array): a clustered data set of N instances, m features and the cluster id at the last column of each vector
        centroids: the value of the centroids after running a clustering algorihtm on the data set
        partition_matrix: the partition matrix
    
    Returns:
        Xie_Beni(float): the value of the Xie Beni index
        
    Reference: Pattern Recognition, S. Theodoridis, K. Koutroumbas
    '''
    total_variation = 0.
    for k, centroid in enumerate(centroids):
        temp = X - centroid
        distances = np.sum(np.power(temp, 2), axis = 1).reshape(len(X), 1)
        # alternative way
        # distances = np.diagonal(np.dot(temp, temp.T)).reshape(len(X), 1) na dw kai trace
        cluster_variation = np.sum(np.power(partition_matrix[:, [k]], 2) * distances) # 2 here is the q value
        total_variation += cluster_variation
                
    min_distance = max_integer
    for k, centroid1 in enumerate(centroids):
        for l, centroid2 in enumerate(centroids):
            if k < l:
                temp = centroid1 - centroid2
                distance = np.sum(np.power(temp, 2)) # it will always be 1 x 1, euclidean distance without the root
                if min_distance > distance:
                    min_distance = distance 
                
    Xie_Beni = total_variation/(min_distance * len(X))
    return Xie_Beni


def fukuyama_sugeno(X, centroids, partition_matrix, q = 2):
    ''' Calculates the fukuyama sugeno index.
    
    Parameters:
        X((N x m + 1) numpy array): a clustered data set of N instances, m features and the cluster id at the last column of each vector
        centroids: the value of the centroids after running a clustering algorihtm on the data set
        partition_matrix: the partition matrix
    
    Returns:
        total_sum(float): the value of the fukuyama sugeno index
        
    Reference: Pattern Recognition, S. Theodoridis, K. Koutroumbas
    '''
    w = np.mean(X, axis = 0)
    total_sum = 0.
    for k, centroid in enumerate(centroids):
        term1 = X - centroid
        distances1 = np.sum(np.power(term1, 2), axis = 1).reshape(len(X), 1) 
        
        term2 = centroid - w
        distances2 = np.sum(np.power(term2, 2))
        
        temp = distances1 - distances2
        total_sum += np.sum(np.power(partition_matrix[:, [k]], q) * temp)
    
    return total_sum

















