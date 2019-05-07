#Starter code prepared by Borna Ghotbi, Polina Zablotskaia, and Ariel Shann for Computer Vision
#based on a MATLAB code by James Hays and Sam Birch 
from __future__ import division
import numpy as np
from util import sample_images, build_vocabulary, get_bags_of_sifts
from classifiers import nearest_neighbor_classify, svm_classify
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd



#For this assignment, you will need to report performance for sift features on two different classifiers:
# 1) Bag of sift features and nearest neighbor classifier
# 2) Bag of sift features and linear SVM classifier

#For simplicity you can define a "num_train_per_cat" vairable, limiting the number of
#examples per category. num_train_per_cat = 100 for intance.

#Sample images from the training/testing dataset. 
#You can limit number of samples by using the n_sample parameter.

print('Getting paths and labels for all train and test data\n')
train_image_paths, train_labels = sample_images("/Users/aurlin/Desktop/425/hw5/sift/train", n_sample=300)
test_image_paths, test_labels = sample_images("/Users/aurlin/Desktop/425/hw5/sift/test", n_sample=100)
       

''' Step 1: Represent each image with the appropriate feature
 Each function to construct features should return an N x d matrix, where
 N is the number of paths passed to the function and d is the 
 dimensionality of each image representation. See the starter code for
 each function for more details. '''

        
print('Extracting SIFT features\n')
#TODO: You code build_vocabulary function in util.py
kmeans = build_vocabulary(train_image_paths, vocab_size=200)

#TODO: You code get_bags_of_sifts function in util.py 
train_image_feats = get_bags_of_sifts(train_image_paths, kmeans)
test_image_feats = get_bags_of_sifts(test_image_paths, kmeans)
        
#If you want to avoid recomputing the features while debugging the
#classifiers, you can either 'save' and 'load' the extracted features
#to/from a file.

# created bins to contain an array that consists only of the bin#. for example bin0 contains only 0's  in test_labels
bin0 = []
bin1 = []
bin2 = []
bin3 = []
bin4 = []
bin5 = []
bin6 = []
bin7 = []
bin8 = []
bin9 = []
bin10 = []
bin11 = []
bin12 = []
bin13 = []
bin14 = []

# organize label by creating bins containing all the images that contain bin category number
for labelValue in range(len(test_labels)):
    for index in range(len(test_labels)):
        if labelValue == test_labels[index]:
            if labelValue == 0:
                bin0.append(index)
            if labelValue == 1:
                bin1.append(index)
            if labelValue == 2:
                bin2.append(index)
            if labelValue == 3:
                bin3.append(index)
            if labelValue == 4:
                bin4.append(index) 
            if labelValue == 5:
                bin5.append(index)
            if labelValue == 6:
                bin6.append(index)
            if labelValue == 7:
                bin7.append(index)
            if labelValue == 8:
                bin8.append(index)
            if labelValue == 9:
                bin9.append(index)
            if labelValue == 10:
                bin10.append(index)
            if labelValue == 11:
                bin11.append(index)
            if labelValue == 12:
                bin12.append(index)
            if labelValue == 13:
                bin13.append(index)
            if labelValue == 14:
                bin14.append(index) 

#go through bin# array to find the clusters that are associated with the bin #
# go through train_image_feats, images, clusters to add them all to a single array containing the sum of all clusters at each position
# then average them with the total length of the bin # array length 
hist0 = [0] * 200
for image in bin0:
      for cluster in range(200):
          hist0[cluster] = hist0[cluster] + train_image_feats[image][cluster]

hist_d0 = len(bin0)
for cluster in range(len(hist0)):
    if hist0[cluster] != 0:
        hist0[cluster] = hist0[cluster]/hist_d0


hist1 = [0] * 200
for image in bin1:
      for cluster in range(200):
          hist1[cluster] = hist1[cluster] + train_image_feats[image][cluster]
                
hist_d1 = len(bin1)
for cluster in range(len(hist1)):
     if hist1[cluster] != 0:
        hist1[cluster] = hist1[cluster]/hist_d1

hist2 = [0] * 200
for image in bin2:
      for cluster in range(200):
          hist2[cluster] = hist2[cluster] + train_image_feats[image][cluster]
#print hist2
          
hist_d2 = len(bin2)
for cluster in range(len(hist2)):
    if hist2[cluster] != 0:
        hist2[cluster] = hist2[cluster]/hist_d2
    
hist3 = [0] * 200
for image in bin3:
      for cluster in range(200):
          hist3[cluster] = hist3[cluster] + train_image_feats[image][cluster]
          
hist_d3 = len(bin3)
for cluster in range(len(hist3)):
    if hist3[cluster] != 0:
        hist3[cluster] = hist3[cluster]/hist_d3
     
hist4 = [0] * 200         
for image in bin4:
      for cluster in range(200):
          hist4[cluster] = hist4[cluster] + train_image_feats[image][cluster]
          
hist_d4 = len(bin4)
for cluster in range(len(hist4)):
     if hist4[cluster] != 0:
        hist4[cluster] = hist4[cluster]/hist_d4
    
hist5 = [0] * 200
for image in bin5:
      for cluster in range(200):
          hist5[cluster] = hist5[cluster] + train_image_feats[image][cluster]
          
hist_d5 = len(bin5)
for cluster in range(len(hist5)):
     if hist5[cluster] != 0:
        hist5[cluster] = hist5[cluster]/hist_d5
    
hist6 = [0] * 200
for image in bin6:
      for cluster in range(200):
          hist6[cluster] = hist6[cluster] + train_image_feats[image][cluster]
          
hist_d6 = len(bin6)
for cluster in range(len(hist6)):
     if hist6[cluster] != 0:
        hist6[cluster] = hist6[cluster]/hist_d6
    
hist7 = [0] * 200
for image in bin7:
      for cluster in range(200):
          hist7[cluster] = hist7[cluster] + train_image_feats[image][cluster]
          
hist_d7 = len(bin7)
for cluster in range(len(hist7)):
     if hist7[cluster] != 0:
        hist7[cluster] = hist7[cluster]/hist_d7
    
hist8 = [0] * 200          
for image in bin8:
      for cluster in range(200):
          hist8[cluster] = hist8[cluster] + train_image_feats[image][cluster]
          
hist_d8 = len(bin8)
for cluster in range(len(hist8)):
     if hist8[cluster] != 0:
        hist8[cluster] = hist8[cluster]/hist_d8
    
hist9 = [0] * 200          
for image in bin9:
      for cluster in range(200):
          hist9[cluster] = hist9[cluster] + train_image_feats[image][cluster]
          
hist_d9 = len(bin9)
for cluster in range(len(hist9)):
     if hist9[cluster] != 0:
        hist9[cluster] = hist9[cluster]/hist_d9
     
hist10 = [0] * 200         
for image in bin10:
      for cluster in range(200):
          hist10[cluster] = hist10[cluster] + train_image_feats[image][cluster]
          
hist_d10 = len(bin10)
for cluster in range(len(hist10)):
     if hist10[cluster] != 0:
        hist10[cluster] = hist10[cluster]/hist_d10
    
hist11 = [0] * 200         
for image in bin11:
      for cluster in range(200):
          hist11[cluster] = hist11[cluster] + train_image_feats[image][cluster]
          
hist_d11 = len(bin11)
for cluster in range(len(hist11)):
     if hist11[cluster] != 0:
        hist11[cluster] = hist11[cluster]/hist_d11
    
hist12 = [0] * 200
for image in bin12:
      for cluster in range(200):
          hist12[cluster] = hist12[cluster] + train_image_feats[image][cluster]
          
hist_d12 = len(bin12)
for cluster in range(len(hist12)):
     if hist12[cluster] != 0:
        hist12[cluster] = hist12[cluster]/hist_d12
    
hist13 = [0] * 200
for image in bin13:
      for cluster in range(200):
          hist13[cluster] = hist13[cluster] + train_image_feats[image][cluster]
          
hist_d13 = len(bin13)
for cluster in range(len(hist13)):
     if hist13[cluster] != 0:
        hist13[cluster] = hist13[cluster]/hist_d13
              
hist14 = [0] * 200          
for image in bin14:     
      for cluster in range(200):
          hist14[cluster] = hist14[cluster] + train_image_feats[image][cluster]

hist_d14 = len(bin14)
for cluster in range(len(hist14)):
     if hist14[cluster] != 0:
        hist14[cluster] = hist14[cluster]/hist_d14

# create all_histograms array that contains all of the hist#'s to be later printed on to visualize
all_histograms = []
all_histograms.append(hist0)
all_histograms.append(hist1)
all_histograms.append(hist2)
all_histograms.append(hist3)
all_histograms.append(hist4)
all_histograms.append(hist5)
all_histograms.append(hist6)
all_histograms.append(hist7)
all_histograms.append(hist8)
all_histograms.append(hist9)
all_histograms.append(hist10)
all_histograms.append(hist11)
all_histograms.append(hist12)
all_histograms.append(hist13)
all_histograms.append(hist14)

# for loop through all the histogram values to be visualized using plot function that is imported
acc = 0
for histogram in range(len(all_histograms)):
    height = all_histograms[histogram]

    plt.bar(range(200),height,align='center', alpha= 0.5)
    plt.ylabel('Number of frequency')
    string = str(acc)
    plt.title ('Histogram Number' + ' ' + string)
    plt.savefig('/Users/aurlin/Desktop/425/' + string + '.png')
    acc = acc + 1    
    plt.clf()
#     
''' Step 2: Classify each test image by training and using the appropriate classifier
 Each function to classify test features will return an N x l cell array,
 where N is the number of test cases and each entry is a string indicating
 the predicted one-hot vector for each test image. See the starter code for each function
 for more details. '''

print('Using nearest neighbor classifier to predict test set categories\n')
#TODO: YOU CODE nearest_neighbor_classify function from classifers.py
pred_labels_knn = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)


# create simple_pred 1 dimensional array of the pred_labels_knn which is a nested array
simple_pred = []
total_accumulator =0
total_cat = 15
for image in range(len(pred_labels_knn)):
    for predic in range(len(pred_labels_knn[image])):
        if pred_labels_knn[image][predic] == 1:
            simple_pred.append(predic)
           # if predic == train_labels[image]:
               # total_accumulator = total_accumulator + 1
                
# divide total_accumulator with the # of categories
# this accumulator is not used 
# total_accumulator = total_accumulator/total_cat
# print "correlation % is FOR PART 1"
# print total_accumulator

# convert values to its corresponding string words of simple_pred
string_predic = []
for index in simple_pred:
    if index == 0:
        string_predic.append('Bedroom')
    if index == 1:
        string_predic.append('Coast')
    if index == 2:
        string_predic.append('Forest')
    if index == 3:
        string_predic.append('Highway')
    if index == 4:
        string_predic.append('Industrial')
    if index == 5:
        string_predic.append('InsideCity')
    if index == 6:
        string_predic.append('Kitchen')
    if index == 7:
        string_predic.append('LivingRoom')
    if index == 8:
        string_predic.append('Mountain')
    if index == 9:
        string_predic.append('Office')
    if index == 10:
        string_predic.append('OpenCountry')
    if index == 11:
        string_predic.append('Store')
    if index == 12:
        string_predic.append('Street')
    if index == 13:
        string_predic.append('Suburb')
    if index == 14:
        string_predic.append('TallBuilding')

#convert test_labels values into its string counterparts
string_actual = []
for index in test_labels:
    if index == 0:
        string_actual.append('Bedroom')
    if index == 1:
        string_actual.append('Coast')
    if index == 2:
        string_actual.append('Forest')
    if index == 3:
        string_actual.append('Highway')
    if index == 4:
        string_actual.append('Industrial')
    if index == 5:
        string_actual.append('InsideCity')
    if index == 6:
        string_actual.append('Kitchen')
    if index == 7:
        string_actual.append('LivingRoom')
    if index == 8:
        string_actual.append('Mountain')
    if index == 9:
        string_actual.append('Office')
    if index == 10:
        string_actual.append('OpenCountry')
    if index == 11:
        string_actual.append('Store')
    if index == 12:
        string_actual.append('Street')
    if index == 13:
        string_actual.append('Suburb')
    if index == 14:
        string_actual.append('TallBuilding')

# print 'simple pred is'
# print string_predic
# print len(string_predic)
# print 'actual pred is'
# print string_actual
# print len(string_actual)

# columns = pd.Series(string_predic, name = 'Predicted')
# rows = pd.Series(string_actual, name = 'Actual')
# panda = pd.crosstab(rows,columns)
# print 'PART 1'
# print panda

#create confusion_matrix to compare the prediction strings vs the actual strings
# check the diagonal of confusion_matrix to accumulate the values which will then be
# divided by the total amount of test_labels to return correlation %
cm = confusion_matrix(string_predic,string_actual)
print 'PART 5'
print cm
a = 0
for row in range(len(cm)):
    if cm[row][row] != 0:
        a = a + cm[row][row]
a = a/len(test_labels)
print 'CORRELATION PART 5'
print a

        
        
  

print('Using support vector machine to predict test set categories\n')
#TODO: YOU CODE svm_classify function from classifers.py
pred_labels_svm = svm_classify(train_image_feats, train_labels, test_image_feats)

# create a simple_pred1 1d array of category numbers from the pred_labels_svm which is a nest array
simple_pred1 = []
total_accumulator1 =0
total_cat1 = 15
for image in range(len(pred_labels_svm)):
    for predic in range(len(pred_labels_svm[image])):
        if pred_labels_svm[image][predic] == 1:
            simple_pred1.append(predic)
            #if predic == train_labels[image]:
                #total_accumulator1 = total_accumulator1 + 1

#total_accumulator1 = total_accumulator1/total_cat1
# print "correlation % is FOR PART 2"
# print total_accumulator1

# go through simple1_values to convert the values into its string counterparts
string_predic1 = []
for index in simple_pred1:
    if index == 0:
        string_predic1.append('Bedroom')
    if index == 1:
        string_predic1.append('Coast')
    if index == 2:
        string_predic1.append('Forest')
    if index == 3:
        string_predic1.append('Highway')
    if index == 4:
        string_predic1.append('Industrial')
    if index == 5:
        string_predic1.append('InsideCity')
    if index == 6:
        string_predic1.append('Kitchen')
    if index == 7:
        string_predic1.append('LivingRoom')
    if index == 8:
        string_predic1.append('Mountain')
    if index == 9:
        string_predic1.append('Office')
    if index == 10:
        string_predic1.append('OpenCountry')
    if index == 11:
        string_predic1.append('Store')
    if index == 12:
        string_predic1.append('Street')
    if index == 13:
        string_predic1.append('Suburb')
    if index == 14:
        string_predic1.append('TallBuilding')
    

# print 'simple pred is'
# print string_predic
# print 'actual pred is'
# print string_actual

# columns1 = pd.Series(string_predic, name = 'Predicted')
# rows1 = pd.Series(string_actual, name = 'Actual')
# panda1 = pd.crosstab(rows1,columns1)

# compare predicted array with the actual string array to visualize the confusion_matrix
# compute the diagonal to obtain the correlation %
cm1 = confusion_matrix(string_predic1,string_actual)
print 'PART 6'
print cm1
a1 = 0
for row1 in range(len(cm1)):
    if cm1[row1][row1] != 0:
        a1 = a1 + cm1[row1][row1]
a1 = a1/len(test_labels)
print 'CORRELATION PART 6'
print a1


print('---Evaluation---\n')
# Step 3: Build a confusion matrix and score the recognition system for 
#         each of the classifiers.
# TODO: In this step you will be doing evaluation. 
# 1) Calculate the total accuracy of your model by counting number
#   of true positives and true negatives over all. 
# 2) Build a Confusion matrix and visualize it. 
#   You will need to convert the one-hot format labels back
#   to their category name format.


# Interpreting your performance with 100 training examples per category:
#  accuracy  =   0 -> Your code is broken (probably not the classifier's
#                     fault! A classifier would have to be amazing to
#                     perform this badly).
#  accuracy ~= .10 -> Your performance is chance. Something is broken or
#                     you ran the starter code unchanged.
#  accuracy ~= .50 -> Rough performance with bag of SIFT and nearest
#                     neighbor classifier. Can reach .60 with K-NN and
#                     different distance metrics.
#  accuracy ~= .60 -> You've gotten things roughly correct with bag of
#                     SIFT and a linear SVM classifier.
#  accuracy >= .70 -> You've also tuned your parameters well. E.g. number
#                     of clusters, SVM regularization, number of patches
#                     sampled when building vocabulary, size and step for
#                     dense SIFT features.
#  accuracy >= .80 -> You've added in spatial information somehow or you've
#                     added additional, complementary image features. This
#                     represents state of the art in Lazebnik et al 2006.
#  accuracy >= .85 -> You've done extremely well. This is the state of the
#                     art in the 2010 SUN database paper from fusing many 
#                     features. Don't trust this number unless you actually
#                     measure many random splits.
#  accuracy >= .90 -> You used modern deep features trained on much larger
#                     image databases.
#  accuracy >= .96 -> You can beat a human at this task. This isn't a
#                     realistic number. Some accuracy calculation is broken
#                     or your classifier is cheating and seeing the test
#                     labels.