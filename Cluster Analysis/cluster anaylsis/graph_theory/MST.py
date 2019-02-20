import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sys import maxsize as max_integer
from functools import reduce as reduce
from tqdm import tqdm

euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))

def minimum_spanning_tree(data, k = 3, q = 1.5, f = 3):
    ''' An implementation of the Minimum Spanning tree clustering algorithm.
    
    Parameters:
        data((m x n) 2-d numpy array): a data set of m instances and n features
        k: the user defined step to define the depth of the neighborhoud of a function
        q: the user defined number of standard deviations of the weights mean above which an edge is considered inconsistent
        f: the user defined threshold of the ratio of the weight of the edge under investigation and the neighborhoud average weights 
    
    Returns:
        clustered_data((m x (n + 1)) 2-d numpy array): the data set with one more column that contains the vector's cluster
        no_of_clusters(integer): the number of clusters

    '''
    N = reduce(lambda x, y: x * y, data.shape[:-1]) # number of vectors CHANGE : N = len(data)
    m = data.shape[-1] #CHANGE: m = len(data[0])
    
    data = data.reshape(N, m) #CHANGE: erase this line
    
    # Construct the complete Graph G
    G = np.empty((N, N))  #o pinakas twn apostasewn tha einai panta disdiastatos. ola ta simeia ston x, ola ta simeia ston y
                           # to N pali tha einai to plithos twn simeion
                           
    ###############################
    
    for i, point in enumerate(data): #use the data array as a 2-D array for this loop only
        G[:, [i]] = euclidean_distance(data,point)
    
    # Construct the MST using the Kruskal's algorithm
    MST = np.zeros(G.shape)
    
    # We initialize three structures for the execution of the algorithm, key, parent, visited
    key = np.zeros(N)
    key.fill(max_integer)
    key[0] = 0
    
    parent = np.zeros(N)
    parent[0] = -1
    
    visited = np.zeros(N)
    
    for i in range(N - 1): #edges
        # Find the next minimun edge
        current_value = np.min(key[np.where(visited == 0)]) #take the min key where visited  = 0
        current_node = np.where(key == current_value)[0][0]
        visited[current_node] = 1
        
        for adj_node in range(N): #all nodes are adjacent
            if  G[current_node, adj_node] < key[adj_node] and visited[adj_node] == 0 and G[current_node][adj_node] !=0:
                key[adj_node] = G[current_node, adj_node]
                parent[adj_node] = current_node
                
    # Fill in the MST array
    x_list = np.empty(N - 1) # x_list and y_list contain two adjacent nodes in the same position i
    y_list = np.empty(N - 1)
    for i, j in enumerate(zip(key[1:], parent[1:]), start = 1):
        MST[j[1], i] = MST[i, j[1]] = j[0]
        x_list[i - 1] = j[1]
        y_list[i - 1] = i

    # Find the inconsistent edges
    inconsistent = np.zeros(N - 1) #follows x_list, y_list
    
    # Find all pairs of nodes of edges
    for i, nodes in tqdm(enumerate(zip(x_list, y_list))):
        weight = MST[nodes[0], nodes[1]]
        list_of_weights_N1 = np.empty((0, 0))
        list_of_weights_N2 = np.empty((0, 0))
        
        list_of_weights_N1 = _recursion_util(nodes, k, list_of_weights_N1, MST)
        list_of_weights_N2 = _recursion_util(nodes[::-1], k, list_of_weights_N2, MST)
        
        #inconsistency criterion
        weight_mean_N1 = np.mean(list_of_weights_N1)
        weight_mean_N2 = np.mean(list_of_weights_N2)
        weight_std_N1 = np.std(list_of_weights_N1)
        weight_std_N2 = np.std(list_of_weights_N2)
        
        if weight > max(q * weight_std_N1 + weight_mean_N1, q * weight_std_N2 + weight_mean_N2) and \
           weight / max(weight_mean_N1, weight_mean_N2) > f: 
            inconsistent[i] = 1 
    
    # Cut the inconsistent edges
    clustered_data = np.hstack((data, np.zeros((N, 1))))
    
    inc_edges_indices = np.nonzero(inconsistent)
    
    for index in inc_edges_indices[0]:
        MST[x_list[index], y_list[index]] = MST[y_list[index], x_list[index]] = 0
    
    visited_nodes = np.zeros(N)
    print('here')
    # Implementing Dfs in order to find a forest of trees
    cluster_id = 1
    for s in range(N):
        if(visited_nodes[s] == 0):
            visited_nodes[s] = 1
            clustered_data[s, 2] = cluster_id
            _dfs_util(MST, s, visited_nodes, cluster_id, clustered_data)
            cluster_id += 1
    
    #Merge
    for i, cluster in enumerate(np.unique(clustered_data[:, m])):
        cluster_indices = np.where(clustered_data[:, m] == cluster)[0] 
        if len(cluster_indices) < 3 :
            # Make the distances of the noisy vectors max_integer, so that to choose the minimum between them and other vectors
            for index in cluster_indices:
                G[index, cluster_indices] = max_integer
            closest_indices = np.argmin(G[cluster_indices, :], axis = 1)
            clustered_data[cluster_indices, m] = clustered_data[closest_indices, m]
    
    no_of_clusters = len(np.unique(clustered_data[:, m]))
    
    # Visual debugging
    #plot_MST_for_debug(clustered_data, MST, inconsistent, x_list, y_list)
    return clustered_data, no_of_clusters

    
def _dfs_util(MST, s, visited_nodes, cluster_id, data):
    ''' A utility depth first search algorithm used in order to find the forest of trees the dataset is 
        consisted of.
        
        Parameters:
        MST: the minimum spanning tree matrix
        s: the current node index
        visited_nodes: the list of nodes that have been visited
        cluster_id: the id of the cluster to be assigned to a node
        data: the data matrix
        
    '''
    adj_nodes =  np.nonzero(MST[s, :])
    for node in adj_nodes[0]:
        if visited_nodes[node] == 0:
            visited_nodes[node] = 1
            data[node, -1] = cluster_id
            _dfs_util(MST, node, visited_nodes, cluster_id, data)
    
        
def _recursion_util(nodes, k, list_of_weights, MST):
    ''' A utility recursive method used in order to gather the weights of an edge's 
        neighborhoud edges.
        
        Parameters:
        nodes: the two nodes of the edge
        k: the step defining the depth of the neighbourhoud edges
        list_of_weights: a list to fill in the weights of the neighborhoud edges
        MST: the minimum spanning tree matrix
        
        Returns: 
        list_of_weights: a list to fill in the weights of the neighborhoud edges
        
    '''
    if k == 0: return list_of_weights
    
    current_node1 = nodes[0]
    current_node2 = nodes[1]
    
    # Find the adjacent nodes of a node excluding its pair
    adj_nodes = np.nonzero(MST[current_node1, :])
    adj_nodes = np.delete(adj_nodes, np.where(adj_nodes[0] == current_node2))
    
    if  len(adj_nodes) == 0: return list_of_weights
    
    for node in adj_nodes:
        k = k - 1
        list_of_weights = np.append(list_of_weights, MST[node, current_node1])
        list_of_weights = _recursion_util((node, current_node1),k, list_of_weights, MST)
        k += 1
    
    return list_of_weights 
    
def plot_MST_for_debug(data, MST, inconsistent, x_list, y_list):
    #debug: visualize MST
    x, y= np.nonzero(MST)
    for ole in zip(x, y):    
        plt.plot((data[ole[0], 0], data[ole[1], 0]), (data[ole[0], 1], data[ole[1], 1]), color = 'm')    

    
    #debug: show in graph inconsistent edges
    for i, inc in enumerate(inconsistent):
        if inc == 1:
            plt.text((data[x_list[i],0] + data[y_list[i],0])/2, (data[x_list[i],1]  + data[y_list[i],1])/2 , "inc")
            plt.plot((data[x_list[i],0], data[y_list[i], 0] ), (data[x_list[i],1], data[y_list[i], 1]))
    
    #debug: show in graph weights
    for ind, g in enumerate(MST):
        for oe, weight in enumerate(g):
            if weight !=0:
                plt.text((data[ind,0] + data[oe, 0])/2, (data[ind, 1]  + data[oe, 1])/2 , round(weight,2), fontsize=7)
    
