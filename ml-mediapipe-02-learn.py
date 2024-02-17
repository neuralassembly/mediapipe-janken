import sys
import os
import glob
import math
import cv2 as cv
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

if len(sys.argv)!=2:
    print('Usage: python ml-mediapipe-learn.py savefile.h5')
    sys.exit()
savefile = sys.argv[1]

def calc_palm_moment(data):  
    palm_array = np.empty((0, 2), int)
    for i in [0, 1, 5, 9, 13, 17]:
        landmark_point = [np.array((data[i, 0], data[i, 1]))]
        palm_array = np.append(palm_array, landmark_point, axis=0)
    M = cv.moments(palm_array)
    cx, cy = 0, 0
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    return cx, cy

num_joints = 21
num_features = num_joints * 2 # 33 joints * 2 dimensions
data_store = np.empty((num_joints, 2), float) 

# X:input vectors, y: targets
X = np.empty((0, num_features), float) 
y = np.array([], int)

# (gu, choki, pa)
num_classes = 3 

# reading taining vectors
for hand_class in range(num_classes):
    # 画像番号0から999まで対応
    for i in range(1000):
        if hand_class==0: #グー画像
            filename = 'ml-learn/data_gu{0:03d}.csv'.format(i)
        elif hand_class==1: #チョキ画像
            filename = 'ml-learn/data_choki{0:03d}.csv'.format(i)
        elif hand_class==2: #パー画像
            filename = 'ml-learn/data_pa{0:03d}.csv'.format(i)

        print('Reading {0}...'.format(filename))
        try:
            data = np.loadtxt(filename, delimiter=',', dtype='float')
        except FileNotFoundError:
            break
        # setting a moment as origin
        x0, y0 = calc_palm_moment(data.astype(int))
        for index in range(num_joints):
            data[index, 0] -= x0
            data[index, 1] -= y0
        # normalizing |moment - wrist| as 1
        length = math.sqrt(data[0, 0]**2 +  data[0, 1]**2)
        for index in range(num_joints):
            data[index, 0] /= length
            data[index, 1] /= length         
        # Appending learning vector
        for flip in [0, 1]:
            if flip==1:
                flipval = -1
            else:
                flipval = 1
            for angle in range(0, 360, 10):
                costheta = math.cos(math.radians(angle))
                sintheta = math.sin(math.radians(angle))
                for index in range(num_joints):
                    data_store[index, 0] = costheta*(flipval*data[index, 0]) -sintheta*data[index, 1]
                    data_store[index, 1] = sintheta*(flipval*data[index, 0]) +costheta*data[index, 1]        
                X = np.append(X, np.array([data_store.flatten()]), axis=0)
                y = np.append(y, hand_class)

(num_samples, tmp) = X.shape

y_keras = keras.utils.to_categorical(y, num_classes)

model = Sequential()
model.add(Dense(units=200, activation='relu', input_shape=(num_features,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X, y_keras, epochs=200, validation_split=0, batch_size=200, verbose=2)

result = np.argmax(model.predict(X, verbose=0), axis=-1)

total = len(X)
success = sum(result==y)

print('Correct rate')
print(100.0*success/total)

model.save(savefile)

plt.xlabel('time step')
plt.ylabel('loss')

#plt.ylim(0, max(np.r_[history.history['val_loss'], history.history['loss']]))
plt.ylim(0, max(history.history['loss']))
#val_loss, = plt.plot(history.history['val_loss'], c='#56B4E9')
loss, = plt.plot(history.history['loss'], c='#E69F00')
#plt.legend([loss, val_loss], ['loss', 'val_loss'])
plt.legend([loss], ['loss'])
plt.show()


