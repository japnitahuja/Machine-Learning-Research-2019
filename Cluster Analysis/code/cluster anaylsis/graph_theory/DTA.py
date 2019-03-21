import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sys import maxsize as max_integer
from scipy.spatial import Delaunay


euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))


def minimum_spanning_tree_variation(data):
    ''' An implementation of a graph algorithm based on Delaunay triangulation graph.
    
    Parameters:
        data((m x n) 2-d numpy array): a data set of m instances and n features
        
    Returns:
        clustered_data((m x (n + 1)) 2-d numpy array): the data set with one more column that contains the vector's cluster
        
    '''
    # Initialization
    N = len(data)
    m = len(data[0])
    G = np.zeros((N, N), np.float32)
    p = max_integer
    #inconsistent = np.zeros(N - 1) #follows x_list, y_list
    
    tri = Delaunay(data)
    
    '''
    # Debug: visualise delaunay triangulisation
    plt.triplot(data[:,0], data[:,1], tri.simplices.copy())
    plt.plot(data[:,0], data[:,1], 'o')
    for i, d in enumerate(data):
        plt.text(d[0], d[1], i)
    plt.show()
    '''
    
    # Build the graph after Delaunay triangulation
    for i, point in enumerate(data):
        all_triangles = np.where(tri.simplices[:,] == i) #all triangles that contain the point
        all_points_in_triangles = tri.simplices[all_triangles[0]]
        neighborhood_points = np.unique(all_points_in_triangles)
        #debug
        #print(neighborhood_points)
        neighborhood_distance_matrix = euclidean_distance(data[neighborhood_points, :], point)
        #debug
        #print(neighborhood_distance_matrix)
        for k, j in enumerate(neighborhood_points):
            if i !=j and neighborhood_distance_matrix[k] < p:
                G[i, j] = neighborhood_distance_matrix[k]

   
    # Build list of edges. the edges are duplicates since G[i,j] = G[j,i], so keep only
    # those for which x_list element < y_list element 
    x_list, y_list = np.nonzero(G)
    all_edges = np.vstack((x_list, y_list))
    all_edges = all_edges[:, np.where(all_edges[0, :] < all_edges[1, :])[0]]
    
    
    #in order to find p we first need to define the delaunay triangulation graph
    edges_weights = G[all_edges[0, :], all_edges[1, :]]
    
    edges_weights = np.sort(edges_weights)
    
    #make 20 partitions
    step = int(np.ceil(len(edges_weights)/20))
    #min_total = max_integer
    p_minimizing_total = 0
    
    max_diff = -max_integer - 1
    for i in range(0, len(edges_weights), step):
        first_partition = edges_weights[:i+step]
        second_partition = edges_weights[i+step:] 
        
        # The writers' solution for cutting edges. Did not work.
        '''first_sum = np.sum(np.power((np.log(first_partition) - np.mean(np.log(first_partition))), 2))/(len(first_partition) - 1)
        second_sum = np.sum(np.power((np.log(second_partition) - np.mean(np.log(second_partition))), 2))/(len(second_partition) - 1)
        print(first_sum + second_sum)
        if min_total > (first_sum + second_sum):
            min_total = first_sum + second_sum
            print(min_total)
            p_minimizing_total = first_partition[-1]
        '''
        
        # My proposal for cutting edges
        if max_diff < np.mean(second_partition) - np.mean(first_partition):
            max_diff = np.mean(second_partition) - np.mean(first_partition)
            p_minimizing_total = first_partition[-1]

    # Erase long, 'inconsistent' edges by setting them to 0
    G[np.where(G > p_minimizing_total)] = 0
    
    clustered_data = np.hstack((data, np.zeros((len(data), 1))))
    visited_nodes = np.zeros(N)
    
    # Depth First Search to create a forest of trees
    cluster_id = 1
    for s in range(N):
        if(visited_nodes[s] == 0):
            visited_nodes[s] = 1
            clustered_data[s, m] = cluster_id
            _dfs_util(G, s, visited_nodes, cluster_id, clustered_data)
            cluster_id += 1
    
    no_of_clusters = len(np.unique(clustered_data[:, 2]))
    
    return clustered_data, no_of_clusters

    
def _dfs_util(G, s, visited_nodes, cluster_id, data):
    ''' A utility depth first search algorithm used in order to find the forest of trees the dataset is 
        consisted of.
        
        Parameters:
        G: the Delaunay graph
        s: the current node index
        visited_nodes: the list of nodes that have been visited
        cluster_id: the id of the cluster to be assigned to a node
        data: the data matrix
    '''
    adj_nodes =  np.nonzero(G[s, :])
    for node in adj_nodes[0]:
        if visited_nodes[node] == 0:
            visited_nodes[node] = 1
            data[node, -1] = cluster_id
            _dfs_util(G, node, visited_nodes, cluster_id, data)
    
        

