import numpy as np
from keras.models import load_model

from pyaudioclassification import feature_extraction, train, predict, print_leaderboard
import matplotlib.pyplot as plt
from matplotlib import cm
from python_speech_features import mfcc
import scipy.io.wavfile as wav

features, labels = feature_extraction('/home/abdullahz/Desktop/pyAudio/data')
np.save('%s.npy' % "musicFeatures", features)
np.save('%s.npy' % "musicLabels", labels)

features = np.load('%s.npy' % "musicFeatures")
labels = np.load('%s.npy' % "musicLabels")

model = train(features, labels)
model.save('newModel.h5')
model = load_model('newModel.h5')


# model = load_model('train1.h5')

pred = predict(model, "/home/abdullahz/Desktop/pyAudio/gel.wav")
print_leaderboard(pred, "/home/abdullahz/Desktop/pyAudio/data")

