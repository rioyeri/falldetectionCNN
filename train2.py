from _sitebuiltins import _Printer

import numpy as np
import cv2
from glob import glob
from features import *
from imutils.object_detection import non_max_suppression
from imutils import paths
from keras.backend import set_image_dim_ordering
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras import layers
from keras.optimizers import adam
from keras.regularizers import l2
from neural_net import TwoLayerNet
from sklearn.preprocessing import Imputer
from sklearn.metrics import classification_report
import datetime
import imutils
import os
#import tensorflow as tf

def training_NNTF(x, y, x_test=None, y_test=None,save_model=None):
    LR = 0.0001
    EP = 50
    input_dim = x.shape[1:]

    model = Sequential()  # inisialisasi JST

    model.add(Flatten(input_shape=input_dim))

    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Fully Connected 2
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(2,activation='softmax'))  # bentuk output layer dengan 2 neuron
    model.summary()  # tampilkan summary jaringan

    #model.compile(loss='binary_crossentropy', optimizer=adam(lr=LR), metrics=['accuracy'])  # inisialisasi model training, menghitung MSE dengan optimizer adam
    model.compile(loss='binary_crossentropy', optimizer=adam(lr=LR), metrics=['accuracy'])  # inisialisasi model training, menghitung MSE dengan optimizer adam
    model.fit(x, y, epochs=EP)
    if save_model:
        model.save('newmodel_NeuralNet_soft_bincross_lr0.0001_ep50_140dt_6fr_4fc4_2.h5')
        model.save_weights('weight_newmodel_NeuralNet_soft_bincross_lr0.0001_ep50_140dt_6fr_4fc4_2.h5')

    # tampilkan evaluasi training
    #scores = model.evaluate(x, y, verbose=0)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("\n%s: %.5f" % ('Validation loss', scores[0]))
    print("%s: %.5f" % ('Validation acc', scores[1]*100))
    print(scores[1])

def countValue(actual_val,pred_val,TP,predTrueVal,actTrueVal,predFalseVal,actFalseVal):
    #print(actual_val)
    if (pred_val == actual_val) and (actual_val == 1):
        TP+=1

    if (pred_val == 1):
        predTrueVal+=1

    if (actual_val == 1):
        actTrueVal+=1

    if (pred_val == 0):
        predFalseVal+=1

    if (actual_val == 0):
        actFalseVal+=1

    return TP,predTrueVal,actTrueVal,predFalseVal,actFalseVal


def training_CNNTF(x, y, x_test=None, y_test=None,save_model=None):
    LR = 0.0001
    EP = 50
    input_dim = x.shape[1:]

    model = Sequential()  # inisialisasi JST

    model.add(Conv2D(5, (3, 3), padding='same', input_shape=input_dim, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    """
    # Conv Layer 2 (depth 15, ukuran filter 3 x 3) - MaxPool 2
    model.add(Conv2D(15, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    """
    model.add(Flatten())

    """
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    """

    # Fully Connected 1
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))


    # Fully Connected 2
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))


    model.add(Dense(8, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))


    model.add(Dense(4, activation='relu'))  # Bentuk Hidden layer dgn 10 neuron dan fungsi
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # aktivasi relu

    model.add(Dense(2,activation='sigmoid'))  # bentuk output layer dengan 2 neuron
    model.summary()  # tampilkan summary jaringan

    model.compile(loss='categorical_crossentropy', optimizer=adam(lr=LR), metrics=['accuracy'])  # inisialisasi model training, menghitung MSE dengan optimizer adam
    model.fit(x, y, epochs=EP)
    if save_model:
        model.save('newmodel_CNNTF_sigm_catcross_lr0.0001_ep50_140dt_6fr_1conv_4fc4_2.h5')
        model.save_weights('weight_newmodel_CNNTF_sigm_catcross_lr0.0001_ep50_140dt_6fr_1conv_4fc4_2.h5')

    # tampilkan evaluasi training
    scores = model.evaluate(x, y, verbose=0)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("\n%s: %.5f" % ('Validation loss', scores[0]))
    print("%s: %.5f" % ('Validation acc', scores[1]*100))
    print(scores[1])


def testing(x_test,y_test, loadmodel):
    predTrueVal=0
    actTrueVal=0
    predFalseVal=0
    actFalseVal=0
    TP=0
    FP=0
    TN=0
    FN=0

    prediction = []
    ytrue = []
    model = load_model(loadmodel)
    predict = model.predict(x_test)
    n=0

    while(n<x_test.__len__()):
        pre = np.argmax(predict[n])
        act = np.argmax(y_test[n])
        TP,predTrueVal,actTrueVal,predFalseVal,actFalseVal = countValue(act,pre,TP,predTrueVal,actTrueVal, predFalseVal,actFalseVal)
        prediction.append(pre)
        ytrue.append(act)
        n+=1

    print("ActTrue = ",actTrueVal)
    print("ActFalse = ",actFalseVal)
    print("PredTrue = ",predTrueVal)
    print("PredFalse = ",predFalseVal)

    print("TP = ",TP)
    FP = predTrueVal-TP
    print("FP = ",FP)
    FN = actTrueVal - TP
    print("FN = ",FN)
    TN = actFalseVal - FP
    print("TN = ",TN)

    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    acc = (TP+TN) / (TP+TN+FP+FN)
    f1score = 2* ((precision*recall)/(precision+recall))
    print("precision = ",precision)
    print("recall = ",recall)
    print("accuracy = ",acc)
    print("f1-score = ",f1score)

    p=0
    while(p<len(prediction)):
        if(prediction[p]==1):
            print("jatuh")
        elif(prediction[p]==0):
            print("tidak jatuh")
        p+=1

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("\n%s: %.5f" % ('Test loss', scores[0]))
    print("%s: %.5f" % ('Test acc', scores[1]*100))
    print(classification_report(y_true=ytrue,y_pred=prediction))


def saveXY(x,y):
    dataset = {'x':x,'y':y}
    np.save('dataset.npy', dataset)

def saveXY_train_test(xtrain,ytrain,xtest,ytest):
    dataset = {'x_train':xtrain,'y_train':ytrain,'x_test':xtest,'y_test':ytest}
    np.save('dataset.npy', dataset)

def HOG():
    i=0
    try:
        if not os.path.exists('data'):
            os.makedirs('data')
    except OSError:
        print('Error: Creating directory of data')
    ekstrak = []
    label =[]
    set_image_dim_ordering('th')

    while(i<2):
        if(i==0):
            j=0
            isilabel = 1
            path = 'dataset/fall2/'
            path += '/*.mp4'
            destiny = './data/fall-frame'
            inp = 0
            extractFrame(path,destiny,j,inp,ekstrak,label,isilabel)
        elif(i==1):
            j=0
            isilabel = 0
            path = 'dataset/adl2/'
            path += '/*.mp4'
            destiny = './data/ADL-frame'
            inp = 0
            extractFrame(path,destiny,j,inp,ekstrak,label,isilabel)
        i+=1
    ekstrak = np.array(ekstrak)
    label = np.array(label)
    label = to_categorical(label, 2)
    return ekstrak, label

def HOG_training():
    i=0
    try:
        if not os.path.exists('data'):
            os.makedirs('data')
    except OSError:
        print('Error: Creating directory of data')
    ekstrak = []
    label =[]
    set_image_dim_ordering('th')

    while(i<2):
        if(i==0):
            j=0
            isilabel = 1
            path = 'dataset3/Training/fall/'
            path += '/*.mp4'
            destiny = './data/training-fall-frame'
            inp = 0
            extractFrame(path,destiny,j,inp,ekstrak,label,isilabel)
        elif(i==1):
            j=0
            isilabel = 0
            path = 'dataset3/Training/adl/'
            path += '/*.mp4'
            destiny = './data/training-ADL-frame'
            inp = 0
            extractFrame(path,destiny,j,inp,ekstrak,label,isilabel)
        i+=1
    ekstrak = np.array(ekstrak)
    label = np.array(label)
    label = to_categorical(label, 2)
    return ekstrak, label

def HOG_validation():
    i=0
    try:
        if not os.path.exists('data'):
            os.makedirs('data')
    except OSError:
        print('Error: Creating directory of data')
    ekstrak = []
    label =[]
    set_image_dim_ordering('th')

    while(i<2):
        if(i==0):
            j=0
            isilabel = 1
            path = 'dataset3/Validation/fall/'
            path += '/*.mp4'
            destiny = './data/validation-fall-frame'
            inp = 0
            extractFrame(path,destiny,j,inp,ekstrak,label,isilabel)
        elif(i==1):
            j=0
            isilabel = 0
            path = 'dataset3/Validation/adl/'
            path += '/*.mp4'
            destiny = './data/validation-ADL-frame'
            inp = 0
            extractFrame(path,destiny,j,inp,ekstrak,label,isilabel)
        i+=1
    ekstrak = np.array(ekstrak)
    label = np.array(label)
    label = to_categorical(label, 2)
    return ekstrak, label

def HOG_testing():
    i=0
    try:
        if not os.path.exists('data'):
            os.makedirs('data')
    except OSError:
        print('Error: Creating directory of data')
    ekstrak = []
    label =[]
    set_image_dim_ordering('th')

    while(i<2):
        if(i==0):
            j=0
            isilabel = 1
            path = 'dataset3/Testing/fall/'
            path += '/*.mp4'
            destiny = './data/testing-fall-frame'
            inp = 0
            extractFrame(path,destiny,j,inp,ekstrak,label,isilabel)
        elif(i==1):
            j=0
            isilabel = 0
            path = 'dataset3/Testing/adl/'
            path += '/*.mp4'
            destiny = './data/testing-ADL-frame'
            inp = 0
            extractFrame(path,destiny,j,inp,ekstrak,label,isilabel)
        i+=1
    ekstrak = np.array(ekstrak)
    label = np.array(label)
    label = to_categorical(label, 2)
    return ekstrak, label

def extractFeat(x,feature_fns):
    x = np.array(x)
    x = x/255
    #x = x.reshape(120,3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    #x = x.reshape(80,3, 40,40).transpose(0, 2, 3, 1).astype("float")
    x = x.reshape(25, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")

    x_feats = extract_features(x, feature_fns, verbose=True)
    # Normalisasi:
    # kurangi data citra dengan rata-rata citra
    mean_feat = np.mean(x_feats, axis=0, keepdims=True)
    x_feats -= mean_feat

    # Preprocessing:
    # Bagi data dengan standard deviation.
    # Proses ini akan menjamin setiap fitur akan memiliki
    # skala yang hampir sama
    std_feat = np.std(x_feats, axis=0, keepdims=True)
    x_feats /= std_feat

    # Preprocessing:
    # tambahkan (append) vektor bernilai 1 sebagai pengali bias ke setiap data
    x_feats = np.hstack([x_feats, np.ones((x_feats.shape[0], 1))])
    #print(x_feats)
    #print(x_feats.size)
    #print(x_feats.shape)
    return x_feats

def extractFrame(path,destiny,j,inp,ekstrak,label,isilabel):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    #num_color_bins = 18  # Jumlah bins yang digunakan dalam color histogram
    #feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
    feature_fns = [hog_feature]
    for img_path in glob(path):
        cap = cv2.VideoCapture(img_path)
        #fgbg = cv2.createBackgroundSubtractorKNN()
        ret = True
        currentFrame = 1
        fra = 1
        aframe = []
        bframe = []
        inp += 1
        while(ret):
            #Capture frame-by-frame
            ret, frame = cap.read()
            #fgmask = fgbg.apply(frame)
            if ret is True:
                fgmask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                continue

            if (currentFrame % 5 == 0):
                #frame = imutils.resize(frame, width=min(400, frame.shape[1]))

                #rects, weights = hog.detectMultiScale(frame, winStride=(3, 3), padding=(6, 6), scale=1.05)
                #rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
                #pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
                #for (x, y, w, h) in pick:
                #    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                #fgmask = frame
                if (currentFrame % 10 == 0):
                    # Saves image of the current frame in jpg file
                    #name = destiny + str(inp) + str(fra)+ '.jpg'
                    #cv2.imwrite(name, fgmask)
                    bframe.append(extractFeat(img_to_array(fgmask),feature_fns)) #with HOG
                else:
                    aframe.append(extractFeat(img_to_array(fgmask),feature_fns)) #with HOG
                j += 1
                fra += 1
            currentFrame+=1
            #cv2.imshow('frame',fgmask)
            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
        ekstrak.append(aframe)
        ekstrak.append(bframe)
        label.append(isilabel)
        label.append(isilabel)
        cap.release()
        cv2.destroyAllWindows()
    return ekstrak, label

def splitdataset():
    mask = range(180)
    x_train = x[mask]
    y_train = y[mask]

    mask = range(180,205)
    x_val = x[mask]
    y_val = y[mask]


    mask = range(205,220)
    x_test = x[mask]
    y_test = y[mask]

    return x_train,y_train,x_val,y_val,x_test,y_test

def splitdataset2(x, y):
    mask = range(140)
    x_train = x[mask]
    y_train = y[mask]

    mask = range(140,190)
    x_val = x[mask]
    y_val = y[mask]

    return x_train,y_train,x_val,y_val

# dataset from Interdisciplinary Centre for Computational Modelling University of Rzeszow

"""
X, Y = HOG() #1
acak = np.random.permutation(len(Y))
x, y = X[acak], Y[acak]
saveXY(x,y)
"""

xtrain, ytrain = HOG_training() #2
acak = np.random.permutation(len(ytrain))
xtrain, ytrain = xtrain[acak], ytrain[acak]

xval, yval = HOG_validation()
acak = np.random.permutation(len(ytrain))
xval, yval = xval[acak], yval[acak]

xtest, ytest = HOG_testing()
acak = np.random.permutation(len(ytest))
xtest, ytest = xtest[acak], ytest[acak]

saveXY_train_test(xtrain, ytrain, xtest, ytest)


x=[]
y=[]
set_image_dim_ordering ('th')

"""
dataset = np.load('dataset.npy')  #1
#print(dataset)
x = dataset.item().get('x')
y = dataset.item().get('y')
"""

dataset = np.load('dataset.npy')  #2
#print(dataset)
x_train = dataset.item().get('x_train')
y_train = dataset.item().get('y_train')
x_test = dataset.item().get('x_test')
y_test = dataset.item().get('y_test')

#x_train,y_train,x_val,y_val,x_test,y_test = splitdataset() #1
x_train,y_train, x_val, y_val = splitdataset2(x_train, y_train)  #2


print("y_train : ",len(y_train))
print("y_val : ",len(y_val))
print("y_test : ",len(y_test))


#training_NNTF(x=x_train, y=y_train, x_test=x_val, y_test=y_val, save_model=True) #training with NN model
training_CNNTF(x=x_train, y=y_train, x_test=x_val, y_test=y_val, save_model=True) #training with CNN model

#testing(x_test, y_test, loadmodel='newmodel_NeuralNet_soft_bincross_lr0.0001_ep50_140dt_6fr_4fc4_2.h5') #testing with NN's model
testing(x_test, y_test, loadmodel='newmodel_CNNTF_sigm_catcross_lr0.0001_ep50_140dt_6fr_1conv_4fc4_2.h5') #testing with CNN's model
