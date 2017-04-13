'''
Predict Genre for any audio file.
First, need to train the models and save them by running script PredictAll_4Genres.py
'''
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
import pickle
import librosa
import utils
import sounddevice as sd
import time
def getAllFeatures(filepath):
    features = []
    y, sr = librosa.load(filepath)
    y=y[0:617000]
    sd.play(y, sr)
    time.sleep(10)
    #Add hamming window overlap code
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.logamplitude(S, ref_power=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, sr=sr, n_mfcc=12)
    cov = np.cov(mfcc)
    mean = np.mean(mfcc, axis=1)
    cov = np.reshape(cov, (np.shape(cov)[0] * np.shape(cov)[1]))
    cov = np.concatenate((cov, mean), axis=0)
    features = np.concatenate((features, cov),axis=0)

    spectral_centroid = librosa.feature.spectral_centroid(y,sr)
    mean_spectral_centroid = np.mean(spectral_centroid, axis=1)
    std_spectral_centroid = np.std(spectral_centroid, axis=1)
    features = np.concatenate((features, mean_spectral_centroid), axis=0)
    features = np.concatenate((features, std_spectral_centroid), axis=0)

    spectral_rolloff = librosa.feature.spectral_rolloff(y, sr)
    mean_spectral_rolloff = np.mean(spectral_rolloff, axis=1)
    std_spectral_rolloff = np.std(spectral_rolloff, axis=1)
    features = np.concatenate((features, mean_spectral_rolloff), axis=0)
    features = np.concatenate((features, std_spectral_rolloff), axis=0)

    zcr = librosa.feature.zero_crossing_rate(y, sr)
    mean_zcr = np.mean(zcr, axis=1)
    std_zcr = np.std(zcr, axis=1)
    features = np.concatenate((features, mean_zcr), axis=0)
    features = np.concatenate((features, std_zcr), axis=0)

    rmse = librosa.feature.rmse(y)
    mean_rmse = np.mean(rmse, axis=1)
    low_energy = len(rmse[np.where(rmse < mean_rmse)])
    features = np.append(features, mean_rmse, axis=0)
    features = np.append(features, low_energy)

    return np.asarray(features)


filepath = 'Genres\jazz\jazz.00008.au'
X = getAllFeatures(filepath)
#Transform to scalar
scalar = sklearn.preprocessing.StandardScaler()
X = scalar.fit_transform(np.array([X]))

#KNN Classifier
neigh = pickle.load(open('knn_4Genre.model','rb'))
Y = neigh.predict(X)
print 'k-NN Genre predicted : ', utils.labels.get(Y[0])

#SVM Classifier
svm = pickle.load(open('svm_4Genre.model','rb'))
Y = svm.predict(X)
print 'SVM Genre predicted : ', utils.labels.get(Y[0])

#Decision Tree Classifier
dt = pickle.load(open('DT_4Genre.model','rb'))
Y = dt.predict(X)
print 'DT Genre predicted : ', utils.labels.get(Y[0])

#Neural Network Classifier
mlp = pickle.load(open('MLP_4Genre.model','rb'))
Y = mlp.predict(X)
print 'Neural Network Genre predicted : ', utils.labels.get(Y[0])

#Naive Bayes
naiveBayes = pickle.load(open('GausianNaiveBayes_4Genre.model','rb'))
Y = naiveBayes.predict(X)
print 'Gaussian Naive Bayes Genre predicted : ', utils.labels.get(Y[0])

#Logistic Regression
lr = pickle.load(open('logistic_4Genre.model','rb'))
Y = lr.predict(X)
print 'Logistic Regression Genre predicted : ', utils.labels.get(Y[0])

#Ensemble
ensemble = pickle.load(open('ensemble_4Genre.model','rb'))
Y = ensemble.predict(X)
print 'Ensemble Classifier Genre predicted : ', utils.labels.get(Y[0])
