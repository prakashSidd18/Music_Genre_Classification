import math
import numpy as np
import sklearn
import utils
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
from numpy.random import permutation, randint
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from statistics import mode
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
import pickle

# data = np.loadtxt('features_others_low_energy.npy',delimiter=',',dtype=float)
data = np.loadtxt('mfcc_gaussian.npy', delimiter=',',dtype=float)
# data = np.append(data1, data, 1)
labels = np.loadtxt('labels_others_low_energy.npy',delimiter=',',dtype=float)
labels.astype(int)
genres = [1,5,6,7]
#Create new data and labels that have genres : Classical, Jazz, Metal, Pop
new_data = data[np.logical_or.reduce([labels == x for x in genres])]
data = new_data
new_labels = labels[np.logical_or.reduce([labels == x for x in genres])]
labels = new_labels

#kmeans clustering
# kmeans = KMeans(n_clusters=len(genres), random_state=0).fit(data)#, train_label)
# labelmap = np.zeros((10,len(genres)))
# for i in range(0,len(kmeans.labels_)):
#     labelmap[int(labels[i])][kmeans.labels_[i]] += 1
# accuracy_kmeans = (sum(labelmap.max(axis=0))/len(labels))*100
# print 'k-Means : ', accuracy_kmeans

# Randomly shuffle the index of nba.
insertCounter = 0
for labelType in genres:
    genreData = data[np.logical_or.reduce([labels == labelType])]
    genreLabels = labels[np.logical_or.reduce([labels == labelType])]
    randomize = np.arange(len(genreData))
    np.random.shuffle(randomize)
    genreData = genreData[randomize]
    genreLabels = genreLabels[randomize]
    testGenreCutoff = int(math.floor(len(genreData)/3))
    if insertCounter > 0:
        test_data = np.append(test_data,genreData[0:testGenreCutoff],axis=0)
        test_label = np.append(test_label, genreLabels[0:testGenreCutoff],axis=0)
        train_data = np.append(train_data, genreData[testGenreCutoff:],axis=0)
        train_label = np.append(train_label, genreLabels[testGenreCutoff:],axis=0)
    else:
        test_data = genreData[0:testGenreCutoff]
        test_label = genreLabels[0:testGenreCutoff]
        train_data = genreData[testGenreCutoff:]
        train_label = genreLabels[testGenreCutoff:]
        insertCounter += 1


randomize = np.arange(len(test_data))
np.random.shuffle(randomize)
test_data = test_data[randomize]
test_label = test_label[randomize]

randomize = np.arange(len(train_data))
np.random.shuffle(randomize)
train_data = train_data[randomize]
train_label = train_label[randomize]

#SVM Classifier
# svm = svm.LinearSVC(random_state=0,penalty='l1',loss='hinge')
# svm.fit(train_data,train_label)
# predictions_svm = svm.predict(test_data)
# pickle.dump(svm, open('svm1_4Genre.model', 'wb'))
# print 'SVM1 : ', svm.score(test_data,test_label)*100
# #cnf_svm1 = confusion_matrix(test_label, predictions_svm1)

svm35 = svm.SVC()
print svm35
svm35.fit(train_data,train_label)
predictions_svm = svm35.predict(test_data)
pickle.dump(svm35, open('svm35_4Genre.model', 'wb'))
print 'SVM35 : ', svm35.score(test_data,test_label)*100

# svm36 = svm.NuSVC(decision_function_shape='ovo')
# svm36.fit(train_data,train_label)
# predictions_svm = svm36.predict(test_data)
# pickle.dump(svm36, open('svm36_4Genre.model', 'wb'))
# print 'SVM36 : ', svm36.score(test_data,test_label)*100
#
# svm37 = svm.NuSVC(decision_function_shape='ovo',nu=0.8)
# svm37.fit(train_data,train_label)
# predictions_svm = svm37.predict(test_data)
# pickle.dump(svm37, open('svm37_4Genre.model', 'wb'))
# print 'SVM37 : ', svm37.score(test_data,test_label)*100

svm1 = svm.LinearSVC(random_state=0,loss='hinge')
print svm1
svm1.fit(train_data,train_label)
predictions_svm = svm1.predict(test_data)
pickle.dump(svm1, open('svm1_4Genre.model', 'wb'))
print 'SVM1 : ', svm1.score(test_data,test_label)*100
#cnf_svm1 = confusion_matrix(test_label, predictions_svm1)

# svm1 = svm.LinearSVC(random_state=0,penalty='l1')
# svm1.fit(train_data,train_label)
# predictions_svm = svm1.predict(test_data)
# pickle.dump(svm1, open('svm3_4Genre.model', 'wb'))
# print 'SVM3 : ', svm1.score(test_data,test_label)*100

# svm2 = svm.LinearSVC(random_state=0)
# svm2.fit(train_data,train_label)
# predictions_svm = svm2.predict(test_data)
# pickle.dump(svm2, open('svm2_4Genre.model', 'wb'))
# print 'SVM2 : ', svm2.score(test_data,test_label)*100
#
# svm3 = svm.SVC(random_state=0,decision_function_shape='ovr',kernel='linear')
# svm3.fit(train_data,train_label)
# predictions_svm = svm3.predict(test_data)
# pickle.dump(svm3, open('svm3_4Genre.model', 'wb'))
# print 'SVM3 : ', svm3.score(test_data,test_label)*100

# svm4 = svm.NuSVC(random_state=0,decision_function_shape='ovr',kernel='linear')
# svm4.fit(train_data,train_label)
# predictions_svm = svm4.predict(test_data)
# pickle.dump(svm4, open('svm4_4Genre.model', 'wb'))
# print 'SVM4 : ', svm4.score(test_data,test_label)*100
#
# svm5 = svm.NuSVC(random_state=0,decision_function_shape='ovr',kernel='linear',nu=0.8)
# svm5.fit(train_data,train_label)
# predictions_svm = svm5.predict(test_data)
# pickle.dump(svm5, open('svm5_4Genre.model', 'wb'))
# print 'SVM5 : ', svm5.score(test_data,test_label)*100

svm6 = svm.SVC(random_state=0,decision_function_shape='ovr',kernel='poly',degree=2)
print svm6
svm6.fit(train_data,train_label)
predictions_svm = svm6.predict(test_data)
pickle.dump(svm6, open('svm6_4Genre.model', 'wb'))
print 'SVM6 : ', svm6.score(test_data,test_label)*100

# svm7 = svm.NuSVC(random_state=0,decision_function_shape='ovr',kernel='poly',degree=2)
# svm7.fit(train_data,train_label)
# predictions_svm = svm7.predict(test_data)
# pickle.dump(svm7, open('svm7_4Genre.model', 'wb'))
# print 'SVM7 : ', svm7.score(test_data,test_label)*100
#
# svm8 = svm.NuSVC(random_state=0,decision_function_shape='ovr',kernel='poly',nu=0.8,degree=2)
# svm8.fit(train_data,train_label)
# predictions_svm = svm8.predict(test_data)
# pickle.dump(svm8, open('svm8_4Genre.model', 'wb'))
# print 'SVM8 : ', svm8.score(test_data,test_label)*100

svm9 = svm.SVC(random_state=0,decision_function_shape='ovr',kernel='poly')
print svm9
svm9.fit(train_data,train_label)
predictions_svm = svm9.predict(test_data)
pickle.dump(svm9, open('svm9_4Genre.model', 'wb'))
print 'SVM9 : ', svm9.score(test_data,test_label)*100

# svm10 = svm.NuSVC(random_state=0,decision_function_shape='ovr',kernel='poly')
# svm10.fit(train_data,train_label)
# predictions_svm = svm10.predict(test_data)
# pickle.dump(svm10, open('svm10_4Genre.model', 'wb'))
# print 'SVM10 : ', svm10.score(test_data,test_label)*100

# svm11 = svm.NuSVC(random_state=0,decision_function_shape='ovr',kernel='poly',nu=0.8)
# svm11.fit(train_data,train_label)
# predictions_svm = svm11.predict(test_data)
# pickle.dump(svm11, open('svm11_4Genre.model', 'wb'))
# print 'SVM11 : ', svm11.score(test_data,test_label)*100

# svm12 = svm.SVC(random_state=0,decision_function_shape='ovr',kernel='poly',degree=5)
# svm12.fit(train_data,train_label)
# predictions_svm = svm12.predict(test_data)
# pickle.dump(svm12, open('svm12_4Genre.model', 'wb'))
# print 'SVM12 : ', svm12.score(test_data,test_label)*100
#
# # svm13 = svm.NuSVC(random_state=0,decision_function_shape='ovr',kernel='poly',degree=5)
# # svm13.fit(train_data,train_label)
# # predictions_svm = svm13.predict(test_data)
# # pickle.dump(svm13, open('svm13_4Genre.model', 'wb'))
# # print 'SVM13 : ', svm13.score(test_data,test_label)*100
#
# # svm14 = svm.NuSVC(random_state=0,decision_function_shape='ovr',kernel='poly',nu=0.8,degree=5)
# # svm14.fit(train_data,train_label)
# # predictions_svm = svm14.predict(test_data)
# # pickle.dump(svm14, open('svm14_4Genre.model', 'wb'))
# # print 'SVM14 : ', svm14.score(test_data,test_label)*100
#
# svm15 = svm.SVC(random_state=0,decision_function_shape='ovr')
# svm15.fit(train_data,train_label)
# predictions_svm = svm15.predict(test_data)
# pickle.dump(svm15, open('svm15_4Genre.model', 'wb'))
# print 'SVM15 : ', svm15.score(test_data,test_label)*100
#
# svm16 = svm.NuSVC(random_state=0,decision_function_shape='ovr')
# svm16.fit(train_data,train_label)
# predictions_svm = svm16.predict(test_data)
# pickle.dump(svm16, open('svm16_4Genre.model', 'wb'))
# print 'SVM16 : ', svm16.score(test_data,test_label)*100
#
# svm17 = svm.NuSVC(random_state=0,decision_function_shape='ovr',nu=0.8)
# svm17.fit(train_data,train_label)
# predictions_svm = svm17.predict(test_data)
# pickle.dump(svm17, open('svm17_4Genre.model', 'wb'))
# print 'SVM17 : ', svm17.score(test_data,test_label)*100
#
svm18 = svm.SVC(kernel='sigmoid')
print svm18
svm18.fit(train_data,train_label)
predictions_svm = svm18.predict(test_data)
pickle.dump(svm18, open('svm18_4Genre.model', 'wb'))
print 'SVM18 : ', svm18.score(test_data,test_label)*100

# svm19 = svm.NuSVC(random_state=0,decision_function_shape='ovr',kernel='sigmoid')
# svm19.fit(train_data,train_label)
# predictions_svm = svm19.predict(test_data)
# pickle.dump(svm19, open('svm19_4Genre.model', 'wb'))
# print 'SVM19 : ', svm19.score(test_data,test_label)*100
#
# svm20 = svm.NuSVC(random_state=0,decision_function_shape='ovr',kernel='sigmoid',nu=0.8)
# svm20.fit(train_data,train_label)
# predictions_svm = svm20.predict(test_data)
# pickle.dump(svm20, open('svm20_4Genre.model', 'wb'))
# print 'SVM20 : ', svm20.score(test_data,test_label)*100

# svm21 = svm.LinearSVC(random_state=0,loss='hinge',multi_class='crammer_singer')
# print svm21
# svm21.fit(train_data,train_label)
# predictions_svm = svm21.predict(test_data)
# pickle.dump(svm1, open('svm21_4Genre.model', 'wb'))
# print 'SVM21 : ', svm21.score(test_data,test_label)*100
# #cnf_svm1 = confusion_matrix(test_label, predictions_svm1)

# svm1 = svm.LinearSVC(random_state=0,penalty='l1')
# svm1.fit(train_data,train_label)
# predictions_svm = svm1.predict(test_data)
# pickle.dump(svm1, open('svm3_4Genre.model', 'wb'))
# print 'SVM3 : ', svm1.score(test_data,test_label)*100

# # svm22 = svm.LinearSVC(random_state=0,multi_class='crammer_singer')
# # svm22.fit(train_data,train_label)
# # predictions_svm = svm22.predict(test_data)
# # pickle.dump(svm22, open('svm22_4Genre.model', 'wb'))
# # print 'SVM22 : ', svm22.score(test_data,test_label)*100
# #
# # svm23 = svm.SVC(random_state=0,decision_function_shape='ovo',kernel='linear')
# # svm23.fit(train_data,train_label)
# # predictions_svm = svm23.predict(test_data)
# # pickle.dump(svm23, open('svm23_4Genre.model', 'wb'))
# # print 'SVM23 : ', svm23.score(test_data,test_label)*100
# #
# # svm24 = svm.NuSVC(random_state=0,decision_function_shape='ovo',kernel='linear')
# # svm24.fit(train_data,train_label)
# # predictions_svm = svm24.predict(test_data)
# # pickle.dump(svm24, open('svm24_4Genre.model', 'wb'))
# # print 'SVM24 : ', svm24.score(test_data,test_label)*100
# #
# # svm25 = svm.NuSVC(random_state=0,decision_function_shape='ovo',kernel='linear',nu=0.8)
# # svm25.fit(train_data,train_label)
# # predictions_svm = svm25.predict(test_data)
# # pickle.dump(svm25, open('svm25_4Genre.model', 'wb'))
# # print 'SVM25 : ', svm25.score(test_data,test_label)*100
# #
# # svm26 = svm.SVC(random_state=0,decision_function_shape='ovo',kernel='poly',degree=2)
# # svm26.fit(train_data,train_label)
# # predictions_svm = svm26.predict(test_data)
# # pickle.dump(svm26, open('svm26_4Genre.model', 'wb'))
# # print 'SVM26 : ', svm26.score(test_data,test_label)*100
# #
# # svm27 = svm.NuSVC(random_state=0,decision_function_shape='ovo',kernel='poly',degree=2)
# # svm27.fit(train_data,train_label)
# # predictions_svm = svm27.predict(test_data)
# # pickle.dump(svm27, open('svm27_4Genre.model', 'wb'))
# # print 'SVM27 : ', svm27.score(test_data,test_label)*100
# #
# # svm28 = svm.NuSVC(random_state=0,decision_function_shape='ovo',kernel='poly',nu=0.8,degree=2)
# # svm28.fit(train_data,train_label)
# # predictions_svm = svm28.predict(test_data)
# # pickle.dump(svm28, open('svm28_4Genre.model', 'wb'))
# # print 'SVM28 : ', svm28.score(test_data,test_label)*100
# #
# # svm29 = svm.SVC(random_state=0,decision_function_shape='ovo',kernel='poly')
# # svm29.fit(train_data,train_label)
# # predictions_svm = svm29.predict(test_data)
# # pickle.dump(svm29, open('svm29_4Genre.model', 'wb'))
# # print 'SVM29 : ', svm29.score(test_data,test_label)*100
# #
# # # svm10 = svm.NuSVC(random_state=0,decision_function_shape='ovr',kernel='poly')
# # # svm10.fit(train_data,train_label)
# # # predictions_svm = svm10.predict(test_data)
# # # pickle.dump(svm10, open('svm10_4Genre.model', 'wb'))
# # # print 'SVM10 : ', svm10.score(test_data,test_label)*100
# #
# # # svm11 = svm.NuSVC(random_state=0,decision_function_shape='ovr',kernel='poly',nu=0.8)
# # # svm11.fit(train_data,train_label)
# # # predictions_svm = svm11.predict(test_data)
# # # pickle.dump(svm11, open('svm11_4Genre.model', 'wb'))
# # # print 'SVM11 : ', svm11.score(test_data,test_label)*100
# #
# # svm32 = svm.SVC(random_state=0,decision_function_shape='ovo',kernel='poly',degree=5)
# # svm32.fit(train_data,train_label)
# # predictions_svm = svm32.predict(test_data)
# # pickle.dump(svm32, open('svm32_4Genre.model', 'wb'))
# # print 'SVM32 : ', svm32.score(test_data,test_label)*100
# #
# # # svm13 = svm.NuSVC(random_state=0,decision_function_shape='ovr',kernel='poly',degree=5)
# # # svm13.fit(train_data,train_label)
# # # predictions_svm = svm13.predict(test_data)
# # # pickle.dump(svm13, open('svm13_4Genre.model', 'wb'))
# # # print 'SVM13 : ', svm13.score(test_data,test_label)*100
# #
# # # svm14 = svm.NuSVC(random_state=0,decision_function_shape='ovr',kernel='poly',nu=0.8,degree=5)
# # # svm14.fit(train_data,train_label)
# # # predictions_svm = svm14.predict(test_data)
# # # pickle.dump(svm14, open('svm14_4Genre.model', 'wb'))
# # # print 'SVM14 : ', svm14.score(test_data,test_label)*100
# #
# # svm35 = svm.SVC(random_state=0,decision_function_shape='ovo')
# # svm35.fit(train_data,train_label)
# # predictions_svm = svm35.predict(test_data)
# # pickle.dump(svm35, open('svm35_4Genre.model', 'wb'))
# # print 'SVM35 : ', svm35.score(test_data,test_label)*100
# #
# # svm36 = svm.NuSVC(random_state=0,decision_function_shape='ovo')
# # svm36.fit(train_data,train_label)
# # predictions_svm = svm36.predict(test_data)
# # pickle.dump(svm36, open('svm36_4Genre.model', 'wb'))
# # print 'SVM36 : ', svm36.score(test_data,test_label)*100
# #
# # svm37 = svm.NuSVC(random_state=0,decision_function_shape='ovo',nu=0.8)
# # svm37.fit(train_data,train_label)
# # predictions_svm = svm37.predict(test_data)
# # pickle.dump(svm37, open('svm37_4Genre.model', 'wb'))
# # print 'SVM37 : ', svm37.score(test_data,test_label)*100
#
# svm38 = svm.SVC(random_state=0,decision_function_shape='ovo',kernel='sigmoid')
# svm38.fit(train_data,train_label)
# predictions_svm = svm38.predict(test_data)
# pickle.dump(svm38, open('svm38_4Genre.model', 'wb'))
# print 'SVM38 : ', svm38.score(test_data,test_label)*100
#
# svm39 = svm.NuSVC(random_state=0,decision_function_shape='ovo',kernel='sigmoid')
# svm39.fit(train_data,train_label)
# predictions_svm = svm39.predict(test_data)
# pickle.dump(svm39, open('svm39_4Genre.model', 'wb'))
# print 'SVM39 : ', svm39.score(test_data,test_label)*100
#
# svm40 = svm.NuSVC(random_state=0,decision_function_shape='ovo',kernel='sigmoid',nu=0.8)
# svm40.fit(train_data,train_label)
# predictions_svm = svm40.predict(test_data)
# pickle.dump(svm40, open('svm40_4Genre.model', 'wb'))
# print 'SVM40 : ', svm40.score(test_data,test_label)*100

#Visualize the audio samples as t-SNE
# X_tsne = TSNE(learning_rate=1000,n_components=2).fit_transform(new_data)
# plt.scatter(X_tsne[:,0], X_tsne[:,1], c=new_labels,label=genres, cmap=plt.cm.get_cmap("jet", 10))
# plt.title('T-SNE Plot for all audio files')
# plt.show()
#
# utils.plot_confusion_matrix(cnf_knn, title='Confusion Matrix for K-NN', classes=genres)
# utils.plot_confusion_matrix(cnf_svm, title='Confusion Matrix for SVM', classes=genres)
# utils.plot_confusion_matrix(cnf_dt, title='Confusion Matrix for Decision Tree', classes=genres)
# utils.plot_confusion_matrix(cnf_ann, title='Confusion Matrix for Artificial Neural Network', classes=genres)
# utils.plot_confusion_matrix(cnf_nb, title='Confusion Matrix for Naive Bayes', classes=genres)
# utils.plot_confusion_matrix(cnf_lr, title='Confusion Matrix for Logistic Regression', classes=genres)
# utils.plot_confusion_matrix(cnf_ensemble, title='Confusion Matrix for Ensemble classifier', classes=genres)
