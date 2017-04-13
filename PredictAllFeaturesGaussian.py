import math
import numpy as np
import sklearn
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
import matplotlib.pyplot as plt
#data = np.loadtxt('mfcc_1000_500.npy',delimiter = ',',dtype = float)
data = np.loadtxt('features_others_low_energy.npy',delimiter=',',dtype=float)
data1 = np.loadtxt('mfcc_gaussian.npy', delimiter=',',dtype=float)
data = np.append(data1, data, 1)

labels = np.loadtxt('labels_others_low_energy.npy',delimiter=',',dtype=float)
labels.astype(int)
print np.shape(data)

#K-Means on entire dataset
kmeans = KMeans(n_clusters=10, random_state=0).fit(data)
predictions_kmeans = kmeans.labels_
print "K-Means Accuracy: "
accuracy_kmeans = len(predictions_kmeans[np.where(predictions_kmeans == labels)])*100/len(labels)
print accuracy_kmeans

# Randomly shuffle the index of nba.
randomize = np.arange(len(data))
np.random.shuffle(randomize)
data = data[randomize]
labels = labels[randomize]

# Set a cutoff for how many items we want in the test set (in this case 1/3 of the items)
test_cutoff = int(math.floor(len(data)/3))
# Generate the test set by taking the first 1/3 of the randomly shuffled indices.
test_data = data[0:test_cutoff]
#Transform to scalar
scalar = sklearn.preprocessing.StandardScaler()
test_data = scalar.fit_transform(test_data)

test_label = labels[0:test_cutoff]
# Generate the train set with the rest of the data.
train_data = data[test_cutoff:]
train_label = labels[test_cutoff:]
#Transform to scalar
train_data = scalar.fit_transform(train_data)

#kmeans clustering
kmeans = KMeans(n_clusters=10, random_state=0).fit(train_data)
predictions_kmeans = kmeans.predict(test_data)
accuracy_kmeans = float(len(predictions_kmeans[np.where(predictions_kmeans == test_label)]))*100/len(test_label)
print 'k-Means Accuracy on test data : ', accuracy_kmeans


#KNN Classifier
neigh = KNeighborsClassifier(n_neighbors=5, algorithm='auto', metric='minkowski', p=1)
neigh.fit(train_data,train_label)
predictions_knn = neigh.predict(test_data)

#SVM Classifier
svc = svm.LinearSVC(random_state=0)
svc = OneVsRestClassifier(svc)
clf = CalibratedClassifierCV(svc, cv=10)
clf.fit(train_data,train_label)
predictions_svm = clf.predict(test_data)

#Decision Tree Classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_label)
predictions_decision = clf.predict(test_data)

#Neural Network Classifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,), random_state=1,activation='tanh')
clf.fit(train_data, train_label)
predictions_neural = clf.predict(test_data)

#Naive Bayes
clf = GaussianNB()
clf.fit(train_data,train_label)
predictions_naive = clf.predict(test_data)

predictions_ensemble = []
for x in range(0,test_cutoff):
    try:
        predictions_ensemble.append(mode([int(predictions_svm[x]),int(predictions_neural[x])]))
    except:
        val = randint(0,5)
        if val == 0:
            predictions_ensemble.append(int(predictions_knn[x]))
        if val == 1:
            predictions_ensemble.append(int(predictions_svm[x]))
        if val == 2:
            predictions_ensemble.append(int(predictions_decision[x]))
        if val == 3:
            predictions_ensemble.append(int(predictions_neural[x]))
        if val == 4:
            predictions_ensemble.append(int(predictions_naive[x]))
        pass

# Create the knn model.
# Look at the five closest neighbors.
# knn = KNeighborsRegressor(n_neighbors=5)
# # Fit the model on the training data.
# knn.fit(train_data, train_label)
# # Make point predictions on the test set using the fit model.
# predictions = knn.predict(test_data)
# print type(predictions)
knn_count = 0
svm_count = 0
decision_count = 0
neural_count = 0
ensemble_count = 0
naive_count = 0
kmeans_count = 0

for x in range(0,test_cutoff):
    # print int(test_label[x]),int(predictions_knn[x])
    if int(test_label[x]) == int(predictions_knn[x]):
        knn_count += 1
    if int(test_label[x]) == int(predictions_svm[x]):
        svm_count += 1
    if int(test_label[x]) == int(predictions_decision[x]):
        decision_count += 1
    if int(test_label[x]) == int(predictions_neural[x]):
        neural_count += 1
    if int(test_label[x]) == int(predictions_ensemble[x]):
        ensemble_count += 1
    if int(test_label[x]) == int(predictions_naive[x]):
        naive_count += 1
    if int(test_label[x]) == int(predictions_kmeans[x]):
        kmeans_count += 1

mse = (((predictions_knn - test_label) ** 2).sum()) / len(predictions_knn)
print "MSE: "
print mse

acc_knn = knn_count/float(test_cutoff) * 100
acc_svm = svm_count/float(test_cutoff) * 100
acc_decision = decision_count/float(test_cutoff) * 100
acc_neural = neural_count/float(test_cutoff) * 100
acc_ensemble = ensemble_count/float(test_cutoff) * 100
acc_naive = naive_count/float(test_cutoff) * 100
acc_kmeans = kmeans_count/float(test_cutoff) * 100
print "KNN Accuracy: "
print acc_knn
print "SVM Accuracy: "
print acc_svm
print "Decision Accuracy: "
print acc_decision
print "Neural Accuracy: "
print acc_neural
print "NaiveBayes Accuracy: "
print acc_naive
print "K-Means Accuracy: "
print acc_kmeans
print "Ensemble Accuracy: "
print acc_ensemble