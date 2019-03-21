import numpy as np
from scipy import ndimage
import os, inspect
from tqdm import tqdm
from functools import reduce
from sys import maxsize as max_integer
import matplotlib.pyplot as plt
from utility.plotting_functions import *
import time

'''
Module of utility functions set to work with the image repository of 
The Berkeley Segmentation Dataset and Benchmark.
'''



def insert_clusters(original_image, seg_file):
    ''' A function that takes the seg format files along with the original image 
        and returns the image as a numpy array, ALONG with the externaly provided 
        clusters (called segments in the seg file).
        
    Parameters:
        original_image(string): the name of the original image. It must be placed 
                                into the images folder
        
        seg_file(string): the name of the .seg file with the human segmented results.
                          It must be placed into the images folder
    
    Returns:
        clustered_image(numpy 3-d array): the .seg file in the form of a 3-d numpy 
                                         array which contains the segment id for each pixel.
        
    '''
    # Find the project's directory and get the files
    this_file = os.path.abspath(inspect.getfile(inspect.currentframe()))
    project_dir = os.path.dirname(os.path.dirname(this_file))
    path_to_images = os.path.join(project_dir,'images\\')
    
    labels = np.loadtxt(path_to_images + seg_file, skiprows = 11)
    
    image = ndimage.imread(path_to_images + original_image)
    
    # Add one extra column to image array in order to keep the cluster ids
    temp = np.zeros((image.shape[0], image.shape[1], 1))
    clustered_image = np.dstack((image, temp))
    
    for row in labels:
        clustered_image[row[1], row[2]: row[3], 3] = row[0]

    return clustered_image
    
    
    
def rand_index_calculation(X_, external_info):
    ''' Calculates the rand index of two different clustering results. The matrices provided as arguments 
        must be contain at least 5000 elements. Instead of comparing all elements, the function chooses 
        5000 element uniformingly distributed in the input matrices and perform its calculations solely on them.
        
    Parameters:
        X_(numpy array): the first clustering result as a numpy array
        external_info(numpy array): the second clustering result as a numpy array
    
    Returns:
        rand index(floag): the value of the rand index
        
    '''
    initial_shape = list(X_.shape)
    N = reduce(lambda x, y: x * y, X_.shape[:-1])
    m = X_.shape[-1] 
    X_ = X_.reshape(N, m)
        
    external_info = external_info.reshape(N, m)
        
    random_size = 5000
    indices = np.random.choice(len(X_), size = random_size, replace = False)
    X_random = X_[indices,:]
    external_info_random = external_info[indices, :]
        
    nominator = 0
    for i, vector1 in tqdm(enumerate(X_random)):
        temp = external_info_random[i + 1:, :]
        for j, vector2 in enumerate(temp, start = i + 1):
            if ((vector1[3] == X_random[j, 3] and vector2[3] == external_info_random[i, 3]) or 
                (vector1[3] != X_random[j, 3] and vector2[3] != external_info_random[i, 3])):
                nominator += 1
  
    X_ = X_.reshape(initial_shape)
    rand_index = nominator/(random_size*(random_size-1)/2)
    return rand_index
        






def merging_procedure(image, threshold):
    ''' Takes a clustered image as a numpy 3-D array, containing the cluster id for each pixel, and transforms it
        in such a way that small clusters are merged into their neighbourhood ones.
        
    Parameters:
        image(3-D numpy array): array containing the cluster ids before the merge procedure
        threshold(integer): the user defined threshold for the maximum number of pixels allowed in a recursion
    
    Returns:
        image(3-D numpy array): array containing the cluster ids after the merge procedure
        
    '''

    N = image.shape[0]
    m = image.shape[1]
    visited = np.zeros((N, m))
    
    no_of_clusters = len(np.unique(image[:,:,3]))
    

    for index in tqdm(np.ndindex(N, m)):
        
        if(visited[index[0], index[1]] == 0):
            dominant_cluster_list = np.zeros(no_of_clusters + 1) # Plus one in case some algorithm starts numbering the clusters fron 1 instead of zero
            counter = 1
            visited[index[0], index[1]] = 3 
            counter, dominant_cluster_list = _dfs_util(image, index[0], index[1], N, m, visited, image[index[0], index[1], 3], counter, dominant_cluster_list, threshold)
            # Reset visited array
            indices_of_dominant_clusters = np.where(visited==2)
            visited[indices_of_dominant_clusters[0], indices_of_dominant_clusters[1]] = 0
            
            indices_of_previously_visited = np.where(visited == 3)
            visited[indices_of_previously_visited[0], indices_of_previously_visited[1]] = 1
            
            if counter < threshold:
                dom_cluster = np.argmax(dominant_cluster_list)
                
                # Change all pixels of previous island to the dominant cluster
                image[indices_of_previously_visited[0], indices_of_previously_visited[1], 3] = dom_cluster
                #draw_clustered_image(image, (321, 481, 3), 5)
             
        
        
    return image


def _moves(y, x):
    ''' Private function that takes the coordinates of the current position as arguments and calculates 
        the next positions.  
    
    Parameters:
        y(integer): the 'vertical coordinate' of the current pixel of the image 
        x(integer): the 'horizontal coordinate' of the current pixel of the image 
    
    Returns:
        list_of_new_positions(list): a list of tuples of length 2 containing all the next possible pixels on 
                                     the image, either eligible or not
    '''
    
    moves = [(0,1), (1,0), (1,1), (-1, -1), (-1, 0),(-1, 1),(0,-1),(1,-1) ]
    list_of_new_positions = []
    for move in moves:
        list_of_new_positions.append((y + move[0], x + move[1]))

    return list_of_new_positions

def _constraints(y, x, N, m): 
    ''' Private function that takes the coordinates of a position as arguments and calculates 
        whether it is eligible or not. Please note that in order to process an image we reshape it to 2 dimensions
    
    Parameters:
        y(integer): the 'vertical coordinate' of the position of the image 
        x(integer): the 'horizontal coordinate' of the position of the image 
        N(integer): the length of the second dimension of the image
        m(integer): the length of the first dimension of the image
    
    Returns:
        list_of_new_positions(list): a list of tuples of length 2 containing all the next possible pixels on 
                                     the image, either eligible or not
    '''
    if y >= N or x >= m:
        return False 
    elif y < 0 or x < 0:
        return False
    else:
        return True

def _dfs_util(image, y, x, N, m, visited, pixels_cluster, counter, dominant_cluster_list, threshold):
    ''' Private function that implements the depth first search algorithms on the image by visiting pixels that
        belong to the same cluster. It also returns the cluster that appears most often in the neighborhood pixels.
        
    Parameters:
        image(numpy array): the 2-D image array
        y(integer): the 'vertical coordinate' of the current position of the image 
        x(integer): the 'horizontal coordinate' of the current position of the image 
        N(integer): the length of the second dimension of the image
        m(integer): the length of the first dimension of the image
        visited(numpy array): a 2-D array to hold the several stages of a pixel
        pixels_cluster(integer): the cluster id of the current pixel 
        counter(integer): a counter to measure the recursion depth
        dominant_cluster_list(numpy array): a list to count the prevailing cluster of the neighbourhood pixels
        threshold(integer): the user defined threshold for the maximum number of pixels allowed in a recursion
    
    Returns:
        list_of_new_positions(list): a list of tuples of length 2 containing all the next possible pixels on 
                                     the image, either eligible or not
    '''
    if counter > threshold:
        return max_integer, []
    
    for move in _moves(y, x):
        y = move[0]
        x = move[1]
        if _constraints(y,x, N, m) == True:
            #print('y', y)
            #print('x', x)
            if visited[y, x] == 0 or visited[y, x] == 2:
                if image[y, x, 3] == pixels_cluster:
                    visited[y, x] = 3
                    counter += 1
                    counter, dominant_cluster_list = _dfs_util(image, y, x, N, m, visited, image[y, x, 3], counter, dominant_cluster_list, threshold)
                    if counter == max_integer:
                        return counter, dominant_cluster_list
                else:
                    if visited[y, x] != 2:
                        #print('boundary y ', y)
                        #print('boundary x ', x)
                        visited[y, x] = 2 # 2 means not visited by used for . it's a workaround in order not to use another array
                        dominant_cluster_list[image[y, x, 3]] += 1
            else: # when we reach here, the pixel is external. External means boundary to other cluster not the image limits
                if visited[y, x] != 2 and visited[y, x] != 3:
                    #print('boundary y ', y)
                    #print('boundary x ', x)
                    visited[y, x] = 2 # 2 means not visited by used for . it's a workaround in order not to use another array
                    dominant_cluster_list[image[y, x, 3]] += 1
    return counter, dominant_cluster_list
                
    























