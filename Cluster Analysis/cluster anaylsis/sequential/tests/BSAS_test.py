from sklearn.datasets import *
import numpy as np
import matplotlib.pyplot as plt
from sequential import BSAS
from validity_scripts import internal_criteria, external_criteria, relative_criteria
from scipy.stats import norm
from tqdm import tqdm
from sys import maxsize as max_integer
from utility.plotting_functions import *
from scipy import ndimage
from utility import image_segm_utility

import unittest

plt.style.use('ggplot')

class Test(unittest.TestCase):


    @unittest.skip("no")
    def testBlobs(self):
        no_of_clusters = 4
        
        # Create the dataset
        X, y = make_blobs(n_samples = 500, centers= no_of_clusters, n_features=2, random_state = 121)
        
        # Run the clustering algorithm
        X, centroids, no_of_clusters = BSAS.basic_sequential_scheme(X, threshold = 9)

        # Plotting
        plot_data(X, no_of_clusters, centroids)
        
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, no_of_clusters, BSAS.basic_sequential_scheme)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, no_of_clusters, y, BSAS.basic_sequential_scheme)
        
        # Histogram of gammas from internal criteria 
        hist_internal_criteria(initial_gamma, list_of_gammas, result)
        hist_external_criteria(initial_indices, list_of_indices, result_list)
        
        plt.show()
    
    #@unittest.skip("no")
    def testCircles(self):
        # Create the dataset
        X, y = make_circles(n_samples=500, shuffle = True, noise = 0.05, factor = 0.5, random_state = 121)
        
        # Run the clustering Algorithm
        X, centroids, no_of_clusters = BSAS.basic_sequential_scheme(X, threshold = 1.1)
        
        # Plotting
        plot_data(X, no_of_clusters, centroids)
        
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, no_of_clusters , BSAS.basic_sequential_scheme)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, no_of_clusters, y, BSAS.basic_sequential_scheme)
        
        # Histogram of gammas from internal and external criteria 
        hist_internal_criteria(initial_gamma, list_of_gammas, result)
        hist_external_criteria(initial_indices, list_of_indices, result_list)
        
        plt.show()
        
    @unittest.skip("no")
    def testMoons(self):
        # Create the dataset
        X, y = make_moons(n_samples=500, shuffle = True, noise = 0.1, random_state = 121)
        
        # Run the clustering algorithm
        X, centroids, no_of_clusters = BSAS.basic_sequential_scheme(X, threshold = 1)
        
        # Plotting
        plot_data(X, no_of_clusters, centroids)
        
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, no_of_clusters, BSAS.basic_sequential_scheme)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, no_of_clusters, y, BSAS.basic_sequential_scheme)
        
        # Histogram of gammas from internal and external criteria 
        hist_internal_criteria(initial_gamma, list_of_gammas, result)
        hist_external_criteria(initial_indices, list_of_indices, result_list)
        
        plt.show()

################################################## Relative Criteria Clustering #########################
    
    @unittest.skip('no')
    def testRelativeBlobs(self):
        no_of_clusters= 4
        
        # Create the dataset
        X, y = make_blobs(n_samples=500, centers= no_of_clusters, n_features=2,random_state=121)
        
        # Successive executions of the clustering algorithm
        no_of_clusters_list, DI, DB, SI= relative_criteria.relative_validity_hard_sequential(X)

        # Plot the indices
        plot_relative_criteria_sequential(no_of_clusters_list, DI, DB, SI)
        plt.show()         
    
    @unittest.skip('no')
    def testRelativeCircles(self):

        # Create the dataset
        X, y = make_circles(n_samples=500, shuffle = True, noise = 0.05, factor = 0.5, random_state = 121)
        
        # Successive executions of the clustering algorithm
        no_of_clusters_list, DI, DB, SI= relative_criteria.relative_validity_hard_sequential(X)

        # Plot the indices
        plot_relative_criteria_sequential(no_of_clusters_list, DI, DB, SI)
        plt.show()     
    
    
    @unittest.skip('no')
    def testRelativeMoons(self):

        # Create the dataset
        X, y = make_moons(n_samples=500, shuffle = True, noise = 0.07, random_state = 121)
        
        # Successive executions of the clustering algorithm
        no_of_clusters_list, DI, DB, SI = relative_criteria.relative_validity_hard_sequential(X)

        # Plot the indices
        plot_relative_criteria_sequential(no_of_clusters_list, DI, DB, SI)
        plt.show()  
                
                
                
    @unittest.skip('no')
    def testImageSegmentation(self):
        image = ndimage.imread('..//..//images//113044.jpg')
        image = image.astype(np.int32, copy = False)
        
        # Algorithm execution. We run BSAS first to get estimates for the centroids
        X_, centroids, total_clusters = BSAS.basic_sequential_scheme(image, max_number_of_clusters = 1000, threshold = 185)
        
        # Calculate the Rand Index to test similarity to external data
        original_image = '113044.jpg'
        seg_file = '113044.seg'
        external_info = image_segm_utility.insert_clusters(original_image, seg_file)
        rand_index = image_segm_utility.rand_index_calculation(X_, external_info)
        print(rand_index)
        
        # Draw the clustered image
        draw_clustered_image(X_, image.shape, total_clusters, rand_index)
        plt.show()



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()