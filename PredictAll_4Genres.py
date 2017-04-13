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

data = np.loadtxt('features_others_low_energy.npy',delimiter=',',dtype=float)
data1 = np.loadtxt('mfcc_gaussian.npy', delimiter=',',dtype=float)
data = np.append(data1, data, 1)
labels = np.loadtxt('labels_others_low_energy.npy',delimiter=',',dtype=float)
labels.astype(int)
genres = [1,5,6,7]
#Create new data and labels that have genres : Classical, Jazz, Metal, Pop
new_data = data[np.logical_or.reduce([labels == x for x in genres])]
data = new_data
new_labels = labels[np.logical_or.reduce([labels == x for x in genres])]
labels = new_labels

#kmeans clustering
kmeans = KMeans(n_clusters=len(genres), random_state=0).fit(data)#, train_label)
labelmap = np.zeros((10,len(genres)))
for i in range(0,len(kmeans.labels_)):
    labelmap[int(labels[i])][kmeans.labels_[i]] += 1
accuracy_kmeans = (sum(labelmap.max(axis=0))/len(labels))*100
print 'k-Means : ', accuracy_kmeans

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

# Set a cutoff for how many items we want in the test set (in this case 1/3 of the items)
#test_cutoff = int(math.floor(len(data)/3))
# Generate the test set by taking the first 1/3 of the randomly shuffled indices.
#test_data = data[0:test_cutoff]
#Transform to scalar
#scalar = sklearn.preprocessing.StandardScaler()
#test_data = scalar.fit_transform(test_data)

#test_label = labels[0:test_cutoff]
# Generate the train set with the rest of the data.
#train_data = data[test_cutoff:]
#train_label = labels[test_cutoff:]
#Transform to scalar
#train_data = scalar.fit_transform(train_data)

#KNN Classifier
for k in range(1,11):
    neigh = KNeighborsClassifier(n_neighbors=k, algorithm='auto', metric='minkowski', p=2)
    neigh.fit(train_data,train_label)
    predictions_knn = neigh.predict(test_data)
    pickle.dump(neigh, open('knn_4Genre.model', 'wb'))
print 'KNN : ', neigh.score(test_data, test_label) * 100#, ' k = ', 10
cnf_knn = confusion_matrix(test_label, predictions_knn)

#SVM Classifier
svc = svm.LinearSVC(random_state=0)
svc = OneVsRestClassifier(svc)
svm = CalibratedClassifierCV(svc, cv=10)
svm.fit(train_data,train_label)
predictions_svm = svm.predict(test_data)
pickle.dump(svm, open('svm_4Genre.model', 'wb'))
print 'SVM : ', svm.score(test_data,test_label)*100
cnf_svm = confusion_matrix(test_label, predictions_svm)

#Decision Tree Classifier
dt = tree.DecisionTreeClassifier()
dt = dt.fit(train_data, train_label)
predictions_decision = dt.predict(test_data)
pickle.dump(dt, open('DT_4Genre.model', 'wb'))
print 'Decision Tree : ', dt.score(test_data,test_label)*100
cnf_dt = confusion_matrix(test_label, predictions_decision)

#Neural Network Classifier
ann = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100,), random_state=1,activation='tanh')
ann.fit(train_data, train_label)
predictions_neural = ann.predict(test_data)
pickle.dump(ann, open('MLP_4Genre.model', 'wb'))
print 'Neural Network : ', ann.score(test_data,test_label)*100
cnf_ann = confusion_matrix(test_label, predictions_neural)

#Naive Bayes
nb = GaussianNB()
nb.fit(train_data,train_label)
predictions_naive = nb.predict(test_data)
pickle.dump(nb, open('GausianNaiveBayes_4Genre.model', 'wb'))
print 'Naive Bayes : ', nb.score(test_data,test_label)*100
cnf_nb = confusion_matrix(test_label, predictions_naive)

#Logistic Regression
logistic = LogisticRegression(multi_class='ovr')
logistic.fit(train_data,train_label)
predictions_lr = logistic.predict(test_data)
pickle.dump(nb, open('logistic_4Genre.model', 'wb'))
print 'LR : ', logistic.score(test_data,test_label)*100
cnf_lr = confusion_matrix(test_label, predictions_lr)

#Ensemble Voting Classifier
eclf3 = VotingClassifier(estimators=[
       ('knn', neigh), ('svm', svm), ('dt', dt), ('ann', ann), ('nb', nb), ('lr', logistic)],
       voting='soft', weights=[1,1,1,1,1,1])
eclf3 = eclf3.fit(train_data, train_label)
predictions_ensemble = eclf3.predict(test_data)
pickle.dump(nb, open('ensemble_4Genre.model', 'wb'))
print 'Ensemble : ', eclf3.score(test_data,test_label)*100
cnf_ensemble = confusion_matrix(test_label, predictions_ensemble)

#Visualize the audio samples as t-SNE
X_tsne = TSNE(learning_rate=1000,n_components=2).fit_transform(new_data)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=new_labels,label=genres, cmap=plt.cm.get_cmap("jet", 10))
plt.title('T-SNE Plot for all audio files')
plt.show()

utils.plot_confusion_matrix(cnf_knn, title='Confusion Matrix for K-NN', classes=genres)
utils.plot_confusion_matrix(cnf_svm, title='Confusion Matrix for SVM', classes=genres)
utils.plot_confusion_matrix(cnf_dt, title='Confusion Matrix for Decision Tree', classes=genres)
utils.plot_confusion_matrix(cnf_ann, title='Confusion Matrix for Artificial Neural Network', classes=genres)
utils.plot_confusion_matrix(cnf_nb, title='Confusion Matrix for Naive Bayes', classes=genres)
utils.plot_confusion_matrix(cnf_lr, title='Confusion Matrix for Logistic Regression', classes=genres)
utils.plot_confusion_matrix(cnf_ensemble, title='Confusion Matrix for Ensemble classifier', classes=genres)
