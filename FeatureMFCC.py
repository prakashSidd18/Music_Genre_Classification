import numpy as np
import librosa
import os
import time
SOUND_SAMPLE_LENGTH = 30000
HAMMING_SIZE = 1000
HAMMING_STRIDE = 500
DIR = '/Users/pradhuman/Desktop/sml_music_genre_classification/Genres'
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
    features = np.zeros(shape=(12,1206), dtype=float)
    y, sr = librosa.load(filepath)
    y=y[0:617000]
    #Add hamming window overlap code
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.logamplitude(S, ref_power=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, sr=sr, n_mfcc=12)
    print np.shape(mfcc)
    features = np.add(features,np.asarray(mfcc))
    features = np.reshape(features, np.shape(features)[0]*np.shape(features)[1])

    return features

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
np.savetxt('mfcc_12_1293.npy',X,delimiter=',')
np.savetxt('labels.npy', Y,delimiter=',')