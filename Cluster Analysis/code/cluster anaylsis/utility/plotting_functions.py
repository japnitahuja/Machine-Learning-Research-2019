import matplotlib.pyplot as plt
import numpy as np
    
    
def plot_data(X, no_of_clusters, centroids = None, centroids_history = None ):
    
    # Initialization
    m = len(X[0])
    np.random.seed(seed = None)
    clusters = np.unique(X[:, m - 1])
    
    # Initialize plots
    f, initDataPlot = plt.subplots(2, sharex=True,  figsize = (12,8))
    f.canvas.set_window_title('Unclustered and Clustered Data')
    #plt.tight_layout()

    initDataPlot[0].set_title('Initial Data')
    initDataPlot[0].set_xlabel('Feature 1')
    initDataPlot[0].set_ylabel('Feature 2')
    
    initDataPlot[1].set_title('Clustered Data')
    initDataPlot[1].set_xlabel('Feature 1')
    initDataPlot[1].set_ylabel('Feature 2')
    
    
    # Plot initial data set without clustering
    initDataPlot[0].scatter(X[:, 0], X[:, 1])
    
    # Plot data after clustering
    for i, cluster in enumerate(clusters):
        initDataPlot[1].scatter(X[ X[:,2] == cluster, 0], X[ X[:, 2] == cluster, 1], c=np.random.rand(3,1), s = 30)
    
    # Plots the centroids history
    if centroids_history is not None:
        colors= ['k', 'b', 'g', 'y', 'm', 'c']
        for alpha_counter, i in enumerate(range(0, len(centroids_history),  no_of_clusters)):
            for j in range(i, i + no_of_clusters):
                initDataPlot[1].plot(centroids_history[j, 0], centroids_history[j, 1], c = colors[j % len(colors)], marker = 'x', mew =  1, ms = 15, alpha = 0.2 + alpha_counter * 0.8/(len(centroids_history)/no_of_clusters))
    
    # Plots the centroids
    if centroids is not None:
        for i, c in enumerate(centroids):
            initDataPlot[1].plot(centroids[i, 0], centroids[i, 1], c = 'r', marker = 'x', mew=2, ms = 10)
    
    
   

def hist_internal_criteria(initial_gamma, list_of_gammas, result):
    f, ax = plt.subplots(figsize = (12,8))
    f.canvas.set_window_title('Internal Criteria')
    n, bins, patches = plt.hist(list_of_gammas, bins = 50, color = 'g')
    ax.hist(initial_gamma, bins = 50, color = 'r')
    
    #bincenters = 0.5*(bins[1:]+bins[:-1])
    #y = norm.pdf(bincenters, np.mean(list_of_gammas), np.std(list_of_gammas))
    #ax.plot(bincenters, y, 'r--', linewidth=1)
    
    ax.set_title(result)
    #f.suptitle(result)
    ax.set_xlabel('Hubert\'s Gamma Values')
    ax.set_ylabel('Probability')
    
    plt.tight_layout()



def hist_external_criteria(initial_indices, list_of_indices, result_list):
    
    f1,  ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,  figsize = (12,8))
    f1.canvas.set_window_title('External Criteria')

    figure_subplots = (ax1, ax2, ax3, ax4)
    names_of_indices = ['Rand statistic', 'Jaccard coefficient', 'Fowlkes and Mallows', 'Hubert\'s Gamma']
    for i in range(len(result_list)):
        n, bins, patches = figure_subplots[i].hist(list_of_indices[i, :], bins = 50, color = 'g',histtype='stepfilled')
        figure_subplots[i].hist(initial_indices[i], bins = [initial_indices[i], initial_indices[i] + 5*(bins[1] - bins[0])], color = 'r')
        # in case we need the initial_indice histogram to have the same length. Not visible enough though
        #figure_subplots[i].hist(initial_indices[i], bins = [initial_indices[i], initial_indices[i] + bins[1] - bins[0]], color = 'r')
        figure_subplots[i].set_title(result_list[i], fontsize=8, wrap = True, ha = 'center')
        figure_subplots[i].set_xlabel(names_of_indices[i])
        figure_subplots[i].set_ylabel('Probability')
        # important addition for good visualization
        figure_subplots[i].set_xlim(min(np.min(list_of_indices[i]), initial_indices[i]) - 0.05, 
                                    max(np.max(list_of_indices[i]), initial_indices[i]) + 0.05) 
        # Fit the normal distribution to the data
        #bincenters = 0.5*(bins[1:]+bins[:-1])
        #y = norm.pdf(bincenters, np.mean(list_of_indices[i, :]), np.std(list_of_indices[i, :]))
        #figure_subplots[i].plot(bincenters, y, 'r--', linewidth=1)
    
    #ax1.set_title('External indices')
    plt.tight_layout()
    

def plot_relative_criteria_fuzzy(no_of_clusters_list, values_of_q, PC, PE, XB, FS):
    
    # row and column sharing
    figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (12,9))
    
    subplots_list = (ax1, ax2, ax3, ax4)
    
    
    
    # Plot PC
    for j, q_value in enumerate(values_of_q):
        ax1.plot(no_of_clusters_list, PC[:, j], label =  q_value)
        
    # Plot PE
    for j, q_value in enumerate(values_of_q):
        ax2.plot(no_of_clusters_list, PE[:, j], label = q_value)
        
    # Plot XB
    for j, q_value in enumerate(values_of_q):
        ax3.plot(no_of_clusters_list, XB[:, j], label = q_value)
        
    # Plot FS
    for j, q_value in enumerate(values_of_q):
        ax4.plot(no_of_clusters_list, FS[:, j], label = q_value)
        
    #plt.tight_layout()
    
    ax1.set_title('Partition Coefficient')
    ax2.set_title('Partition Entropy Coefficient')
    ax3.set_title('Xien Ben index')
    ax4.set_title('Fukuyama Sugeno index')
    figure.canvas.set_window_title('Relative Indices')
    
    for subplot in subplots_list:
        subplot.set_xlabel('Number of clusters')
        subplot.set_ylabel('Index value')
        subplot.legend(title = 'q values',framealpha= 0.7)

def plot_relative_criteria_hard(no_of_clusters_list, DI, DB, SI, GI):
    # row and column sharing
    figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (12,9))
    
    subplots_list = (ax1, ax2, ax3, ax4)
    
    # Plot DI
    ax1.plot(no_of_clusters_list, DI)
    
    # Plot DB
    ax2.plot(no_of_clusters_list, DB)
    
    # Plot SI
    ax3.plot(no_of_clusters_list, SI)
    
    # Plot GI
    ax4.plot(no_of_clusters_list, GI)
    
    #plt.tight_layout()
    
    ax1.set_title('Dunn Index(maximum)')
    ax2.set_title('Davies - Bouldin(minimum)')
    ax3.set_title('Silhouette Index(maximum)')
    ax4.set_title('Gap Index(maximum)')
    figure.canvas.set_window_title('Relative Indices')
    
    for subplot in subplots_list:
        subplot.set_xlabel('Parameter value')
        subplot.set_ylabel('Index value')

def plot_relative_criteria_hard_large_data(no_of_clusters_list, DB):

    # Plot DB
    plt.plot(no_of_clusters_list, DB)
    
    
    plt.title('Davies - Bouldin(minimum)')
    
    fig = plt.gcf()
    fig.canvas.set_window_title('Relative Indices')
    
    plt.xlabel('Parameter value')
    plt.ylabel('Index value')
        
        

def plot_relative_criteria_graph(no_of_k_list, no_of_f_list, DI, SI):
    # row and column sharing
    figure, (ax1, ax2) = plt.subplots(2, 1, figsize = (12,9))
    
    subplots_list = (ax1, ax2)
    
    # Plot PC
    for j, q_value in enumerate(no_of_f_list):
        ax1.plot(no_of_k_list, DI[:, j], label =  q_value)
        
    # Plot PE
    for j, q_value in enumerate(no_of_f_list):
        ax2.plot(no_of_k_list, SI[:, j], label = q_value)
        
    ax1.set_title('Dunn Index(maximum)')
    ax2.set_title('Silhouette Index(maximum)')


    figure.canvas.set_window_title('Relative Indices')
    
    for subplot in subplots_list:
        subplot.set_xlabel('k values')
        subplot.set_ylabel('Index value')
        subplot.legend(title = 'f values',framealpha= 0.7)
        
def plot_relative_criteria_sequential(no_of_clusters_list, DI, DB, SI):
    # row and column sharing
    figure, ((ax1, ax2, ax3)) = plt.subplots(3, 1, figsize = (12,9))
    
    subplots_list = (ax1, ax2, ax3)
    
    # Plot DI
    ax1.plot(no_of_clusters_list, DI)
    
    # Plot DB
    ax2.plot(no_of_clusters_list, DB)
    
    # Plot SI
    ax3.plot(no_of_clusters_list, SI)

    
    #plt.tight_layout()
    
    ax1.set_title('Dunn Index(maximum)')
    ax2.set_title('Davies - Bouldin(minimum)')
    ax3.set_title('Silhouette Index(maximum)')

    figure.canvas.set_window_title('Relative Indices')
    
    for subplot in subplots_list:
        subplot.set_xlabel('Parameter value')
        subplot.set_ylabel('Index value')




def draw_clustered_image(X, shape_of_image, rand_index):
    ''' A utility function used to re-draw the clustered image by using one single colour for each cluster
    '''
    # Builds an empty image numpy array with the same dimensions as our image
    picture = np.empty(shape_of_image)
    
    clusters = np.unique(X[:, :, 3])
            
    np.random.seed(14)
    for i, cluster_ in enumerate(clusters):
        x, y = np.where(X[:, :, 3] == cluster_)
        random_color = np.random.randint(256, size = (1,3))
        picture[x, y] = random_color
        #print(random_color)
    
    title = 'Number of clusters: ', len(clusters), ' rand:', rand_index
    plt.title(title)
    plt.grid(False)
    
    plt.imshow(picture)


def plot_relative_criteria_TTSS(no_of_threshold_values1, no_of_threshold_values2, DI, DB, SI):
    
    # row and column sharing
    figure, ((ax1, ax2, ax3)) = plt.subplots(3, 1, figsize = (12,9))
    
    subplots_list = (ax1, ax2, ax3)
    
    # Plot DI
    for j, t2 in enumerate(no_of_threshold_values2):
        ax1.plot(no_of_threshold_values1, DI[:, j], label = t2)
        
    # Plot DB
    for j, t2 in enumerate(no_of_threshold_values2):
        ax2.plot(no_of_threshold_values1, DB[:, j], label = t2)
        
    # Plot SI
    for j, t2 in enumerate(no_of_threshold_values2):
        ax3.plot(no_of_threshold_values1, SI[:, j], label = t2)
    
    ax1.set_title('Dunn Index(maximum)')
    ax2.set_title('Davies - Bouldin(minimum)')
    ax3.set_title('Silhouette Index(maximum)')

    figure.canvas.set_window_title('Relative Indices')
    
    for subplot in subplots_list:
        subplot.set_xlabel('Parameter value')
        subplot.set_ylabel('Index value')
        subplot.legend(title = 'Threshold2 values',framealpha= 0.7)






