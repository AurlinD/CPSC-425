 #Starter code prepared by Borna Ghotbi for computer vision
 #based on MATLAB code by James Hay
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import svm

'''This function will predict the category for every test image by finding
the training image with most similar features. Instead of 1 nearest
neighbor, you can vote based on k nearest neighbors which will increase
performance (although you need to pick a reasonable value for k). '''

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):

    '''
    Parameters
        ----------
        train_image_feats:  is an N x d matrix, where d is the dimensionality of the feature representation.
        train_labels: is an N x l cell array, where each entry is a string 
        			  indicating the ground truth one-hot vector for each training image.
    	test_image_feats: is an M x d matrix, where d is the dimensionality of the
    					  feature representation. You can assume M = N unless you've modified the starter code.
        
    Returns
        -------
    	is an M x l cell array, where each row is a one-hot vector 
        indicating the predicted category for each test image.

    Usefull funtion:
    	
    	# You can use knn from sci-kit learn.
        # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    '''
    k_number =  50
    # initialize the classifier
    neigh = KNeighborsClassifier(n_neighbors=k_number)
    # using train_image_feats value to fit model of target value train_labels
    neigh.fit(train_image_feats,train_labels) 
    # predict test features into the fitted training labels 
    predicted_labels = neigh.predict(test_image_feats)
    labels = len(set(train_labels))
    
    #kmeans = KMeans(n_clusters=3, random_state=0).fit(train_image_feats)
    # print "predicted_labels"
    # print predicted_labels
    
    # initialize the M*1 hot array
    
    ml_array = np.zeros((len(predicted_labels), labels))  
    for binNumber in range(len(predicted_labels)):
        binValue = predicted_labels[binNumber]        
        for index in range(labels):
            if(binValue == index):
                ml_array[binNumber][index] = 1
                
                
        
        
        
    # print "ml array is-----------"   
    # print ml_array   
    return ml_array



'''This function will train a linear SVM for every category (i.e. one vs all)
and then use the learned linear classifiers to predict the category of
very test image. Every test feature will be evaluated with all 15 SVMs
and the most confident SVM will "win". Confidence, or distance from the
margin, is W*X + B where '*' is the inner product or dot product and W and
B are the learned hyperplane parameters. '''

def svm_classify(train_image_feats, train_labels, test_image_feats):

    '''
    Parameters
        ----------
        train_image_feats:  is an N x d matrix, where d is the dimensionality of the feature representation.
        train_labels: is an N x l cell array, where each entry is a string 
        			  indicating the ground truth one-hot vector for each training image.
    	test_image_feats: is an M x d matrix, where d is the dimensionality of the
    					  feature representation. You can assume M = N unless you've modified the starter code.
        
    Returns
        -------
    	is an M x l cell array, where each row is a one-hot vector 
        indicating the predicted category for each test image.
        
        
        

    Usefull funtion:
    	
    	# You can use svm from sci-kit learn.
        # Reference: https://scikit-learn.org/stable/modules/svm.html

    '''
    c = 2400
    # initialize the svm 
    instance = svm.LinearSVC(C = c)
    # using train_image_feats value to fit model of target value train_labels
    instance.fit(train_image_feats, train_labels)
    # predict test features into the fitted training labels 
    predicted_labels1 = instance.predict(test_image_feats)  
    labels1 = len(set(train_labels))
    
    # initialize the M*1 hot array
    ml_array1 = np.zeros((len(predicted_labels1), labels1))  
    for binNumber in range(len(predicted_labels1)):
        binValue1 = predicted_labels1[binNumber]        
        for index in range(labels1):
            if(binValue1 == index):
                ml_array1[binNumber][index] = 1
    
 

    return ml_array1

