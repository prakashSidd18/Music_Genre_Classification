'''
Compute Spectral Centroid, Spectral Roll Off, Zero crossing rate
Make Gaussian assumption and save mean and standard deviation for each feature
Save low energy as fraction of frames having rmse lower than the avg rmse of the audio file
'''
import numpy as np
import librosa
import os
import time
SOUND_SAMPLE_LENGTH = 30000
HAMMING_SIZE = 1000
HAMMING_STRIDE = 500
DIR = 'Genres'
labels = {
    'blues'     :   0,
    'classical' :   1,
    'country'   :   2,
    'disco'     :   3,
    'hiphop'    :   4,
    'jazz'      :   5,
    'metal'     :   6,
    'pop'       :   7,
    'reggae'    :   8,
    'rock'      :   9,
}

def getFeatures(filepath):
    # y, sr = librosa.load(filepath)
    # mfcc = librosa.feature.mfcc(y,sr)
    #featuresArray = []
    features = []
    y, sr = librosa.load(filepath)
    y=y[0:617000]
    #Add hamming window overlap code
    spectral_centroid = librosa.feature.spectral_centroid(y,sr)
    mean_spectral_centroid = np.mean(spectral_centroid, axis=1)
    std_spectral_centroid = np.std(spectral_centroid, axis=1)
    features.append(mean_spectral_centroid[0])
    features.append(std_spectral_centroid[0])

    spectral_rolloff = librosa.feature.spectral_rolloff(y, sr)
    mean_spectral_rolloff = np.mean(spectral_rolloff, axis=1)
    std_spectral_rolloff = np.std(spectral_rolloff, axis=1)
    features.append(mean_spectral_rolloff[0])
    features.append(std_spectral_rolloff[0])

    zcr = librosa.feature.zero_crossing_rate(y, sr)
    mean_zcr = np.mean(zcr, axis=1)
    std_zcr = np.std(zcr, axis=1)
    features.append(mean_zcr[0])
    features.append(std_zcr[0])

    rmse = librosa.feature.rmse(y)
    mean_rmse = np.mean(rmse, axis=1)

    low_energy = len(rmse[np.where(rmse < mean_rmse)])
    features.append(mean_rmse[0])
    features.append(low_energy)

    return np.asarray(features)

X=[]
Y=[]
F=[]
for subdir, dirs, files in os.walk(DIR):
    #print subdir, files
    files.sort()
    print files
    for file in files:
        temp = str(file).split('.')
        filepath = os.path.join(DIR, temp[0], file)
        start = time.time()
        if not str(filepath).__contains__('.DS_Store'):
            mfcc = getFeatures(filepath)
            print file+' processed'+str(time.time()-start)
            label = labels[temp[0]]
            X.append(mfcc)
            Y.append(label)
            F.append(file)
# np.savetxt('files.npy',F,delimiter=',')
np.savetxt('features_others_low_energy.npy',X,delimiter=',')
np.savetxt('labels_others_low_energy.npy', Y,delimiter=',')