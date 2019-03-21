from sklearn.datasets import *
import numpy as np
from cost_function_optimization import fuzzy_clustering
from validity_scripts import internal_criteria, external_criteria, relative_criteria
from utility.plotting_functions import *
import matplotlib.pyplot as plt
from scipy import  ndimage
from utility import image_segm_utility
from sequential import BSAS


import unittest

plt.style.use('ggplot')

euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))

class Test(unittest.TestCase):

    @unittest.skip("no")
    def testBlobs(self):
        no_of_clusters = 4
        
        # Create the dataset
        X, y = make_blobs(n_samples = 500, centers= no_of_clusters, n_features=2,random_state=46)
        
        # Run the clustering algorithm
        clusters_number_to_execute = 4
        X, centroids, ita, centroids_history, partition_matrix = fuzzy_clustering.fuzzy(X, no_of_clusters = clusters_number_to_execute)
        
        # Plotting
        plot_data(X, clusters_number_to_execute, centroids, centroids_history)
        
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, clusters_number_to_execute, fuzzy_clustering.fuzzy)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, clusters_number_to_execute, y, fuzzy_clustering.fuzzy)
        
        # Histogram of gammas from internal criteria 
        hist_internal_criteria(initial_gamma, list_of_gammas, result)
        hist_external_criteria(initial_indices, list_of_indices, result_list)
        
        plt.show()
    
    #@unittest.skip("no")
    def testCircles(self):

        # Create the dataset
        X, y = make_circles(n_samples=500, shuffle = True, noise = 0.05, factor = 0.5, random_state = 10)
        
        # Run the clustering Algorithm
        clusters_number_to_execute = 3
        X, centroids, ita, centroids_history, partition_matrix = fuzzy_clustering.fuzzy(X, no_of_clusters = clusters_number_to_execute)
        
        # Plotting
        plot_data(X, clusters_number_to_execute, centroids, centroids_history)
        
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, clusters_number_to_execute , fuzzy_clustering.fuzzy)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, clusters_number_to_execute, y, fuzzy_clustering.fuzzy)
        
        # Histogram of gammas from internal and external criteria 
        hist_internal_criteria(initial_gamma, list_of_gammas, result)
        hist_external_criteria(initial_indices, list_of_indices, result_list)
        
        plt.show()
        
    @unittest.skip("no")
    def testMoons(self):
        # Create the dataset
        X, y = make_moons(n_samples=300, shuffle = True, noise = 0.05, random_state = 10)
        
        # Run the clustering algorithm
        clusters_number_to_execute = 2
        X, centroids, ita, centroids_history, partition_matrix = fuzzy_clustering.fuzzy(X, no_of_clusters = clusters_number_to_execute)
        
        # Plotting
        plot_data(X, clusters_number_to_execute, centroids, centroids_history)
        
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, clusters_number_to_execute, fuzzy_clustering.fuzzy)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, clusters_number_to_execute, y, fuzzy_clustering.fuzzy)
        
        # Histogram of gammas from internal and external criteria 
        hist_internal_criteria(initial_gamma, list_of_gammas, result)
        hist_external_criteria(initial_indices, list_of_indices, result_list)
        
        plt.show()
    
    
    ######################### Relative Criteria Clustering #########################
    
    @unittest.skip('no')
    def testRelativeBlobs(self):
        no_of_clusters= 4
        
        # Create the dataset
        X, y = make_blobs(n_samples=500, centers= no_of_clusters, n_features=2,random_state=46)
        
        # Successive executions of the clustering algorithm
        no_of_clusters_list, values_of_q, PC, PE, XB, FS = relative_criteria.relative_validity_fuzzy(X)
        
        # Plot the indices
        plot_relative_criteria_fuzzy(no_of_clusters_list, values_of_q, PC, PE, XB, FS)
        plt.show()        
    
    
    @unittest.skip('no')
    def testRelativeCircles(self):
        # Create the dataset
        X, y = make_circles(n_samples=500, shuffle = True, noise = 0.05, factor = 0.5, random_state = 10)
        
        # Successive executions of the clustering algorithm
        no_of_clusters_list, values_of_q, PC, PE, XB, FS = relative_criteria.relative_validity_fuzzy(X)
        
        # Plot the indices
        plot_relative_criteria_fuzzy(no_of_clusters_list, values_of_q, PC, PE, XB, FS)
        plt.show()      
    
    
    @unittest.skip('no')
    def testRelativeMoons(self):
        # Create the dataset
        X, y = make_moons(n_samples=500, shuffle = True, noise = 0.01, random_state = 10)
        
        # Successive executions of the clustering algorithm
        no_of_clusters_list, values_of_q, PC, PE, XB, FS = relative_criteria.relative_validity_fuzzy(X)
        
        # Plot the indices
        plot_relative_criteria_fuzzy(no_of_clusters_list, values_of_q, PC, PE, XB, FS)
        plt.show()      
    
    
                    
    ################################################## Image Segmentation #########################
    
    
    @unittest.skip('no')
    def testRelativeImageSegmentation(self):
        image = ndimage.imread('..//..//images//113044.jpg')
        
        # Successive executions of the clustering algorithm
        no_of_clusters_list, values_of_q, PC, PE, XB, FS = relative_criteria.relative_validity_fuzzy(image)
        
        # Plot the indices
        plot_relative_criteria_fuzzy(no_of_clusters_list, values_of_q, PC, PE, XB, FS)
        plt.show()  
    
    
    @unittest.skip('no')
    def testImageSegmentation(self):
        image = ndimage.imread('..//..//images//181091.jpg')
        image = image.astype(np.int32, copy = False)
        
        # Algorithm execution.
        clusters_number_to_execute = 28
        clustered_data, centroids, total_clusters = BSAS.basic_sequential_scheme(image)
        X_, centroids, ita, centroids_history, partition_matrix = fuzzy_clustering.fuzzy(image, no_of_clusters = clusters_number_to_execute)
        
        ###################################################################
        # Merging procedure
        
        X_  = image_segm_utility.merging_procedure(X_, 500)
        
        
        
        # Calculate the Rand Index to test similarity to external data
        original_image = '181091.jpg'
        seg_file = '181091.seg'
        external_info = image_segm_utility.insert_clusters(original_image, seg_file)
        rand_index = image_segm_utility.rand_index_calculation(X_, external_info)
        print(rand_index)
        
        # Draw the clustered image
        draw_clustered_image(X_, image.shape, rand_index)
        plt.show()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()