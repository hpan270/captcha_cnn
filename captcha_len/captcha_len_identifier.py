from __future__ import print_function
import os.path
import numpy as np
np.random.seed(1717) 
import argparse
from keras.models import Sequential
from keras.utils import np_utils
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import model_from_json
from keras import optimizers
import h5py
from keras.callbacks import ModelCheckpoint
import captcha_params
from load_data import *



def load_data2(tol_num,train_num):
    data = np.empty((tol_num, 1, opt.height, opt.width),dtype="float32")
    label = np.empty((tol_num,output_len),dtype="uint8")

    imgs = os.listdir(opt.dataset)
    
    for i in range(tol_num):
        img = get_image_from_file(opt.dataset+imgs[i])

        arr = np.asarray(img,dtype="float32")
        try:
            data[i,:,:,:] = arr
            captcha_len = imgs[i][1:2]
            v=np.zeros(output_len)          
            v[int(captcha_len)-opt.min_clen] = 1
            label[i]= v
        except:
            pass

    # the data, shuffled and split between train and test sets
    rr = [i for i in range(tol_num)] 
    random.shuffle(rr)
    X_train = data[rr][:train_num]
    y_train = label[rr][:train_num]
    X_test = data[rr][train_num:]
    y_test = label[rr][train_num:]
    
    return (X_train,y_train),(X_test,y_test)




# input image dimensions
img_rows, img_cols = captcha_params.get_height(), captcha_params.get_width()
aparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
aparser.add_argument('--tol_num', type=int, default=25000, help='total number of captchas')
aparser.add_argument('--train_num', type=int, default=20000, help='number of captchas for train')
aparser.add_argument('--min_clen', type=int, default=2, help='min captcha length')
aparser.add_argument('--max_clen', type=int, default=6, help='max captcha length')
aparser.add_argument('--batch_size', type=int, default=128, help='batch_size')
aparser.add_argument('--epoch', type=int, default=64, help='number of epochs')
aparser.add_argument('--dataset', type=str, default='./dataCL/', help='the path of dataset')
aparser.add_argument('--width', type=int, default=120, help='width of captcha')
aparser.add_argument('--height', type=int, default=60, help='height of captchas')
opt = aparser.parse_args()
batch_size = opt.batch_size
nb_epoch = opt.epoch
output_len = opt.max_clen - opt.min_clen +1

# the data, shuffled and split between train and test sets
(X_train, Y_train), (X_test, Y_test) = load_data2(tol_num = opt.tol_num,train_num = opt.train_num)
print(Y_train)

# i use the theano backend
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(Y_test)



model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3,3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3,3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.5))

# Fully connected layer
model.add(Flatten())
model.add(Dense(3072))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(output_len))
model.add(Activation('softmax'))

json_string = model.to_json()
open("./model/model_cap_len_iden.json","w").write(json_string)

adam = optimizers.adam(lr=0.0005)

model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])

wfilepath = './model/wghts_cap_len_iden.h5'
checkpoint = ModelCheckpoint(wfilepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test,Y_test), callbacks=callbacks_list)

score = model.evaluate(X_test, Y_test, verbose=0)
predict = model.predict(X_test,batch_size = batch_size,verbose = 0)
print(predict)

# calculate the accuracy with the test data
acc = 0
for i in range(X_test.shape[0]): #todo y_test, y_train
    trueval = get_max(Y_test[i,])
    predict2 = get_max(predict[i,])
    if trueval == predict2:
        acc+=1
    if i<20:
        print (i,' true val: ',trueval)
        print (i,' predict: ',predict2)
print('predict correctly: ',acc)
print('total prediction: ',X_test.shape[0])
print('Score: ',score)

