import sys
import io
import os
import csv
import time
from load_data import *
import captcha_params
from PIL import Image
from keras.models import model_from_json
from keras.models import Sequential
from keras import optimizers

# MAX_CAPTCHA = captcha_params.get_captcha_size()
# CHAR_SET_LEN = captcha_params.get_char_set_len()
jfile='./model/model_cap_len_iden.json'
wfile='./model/wghts_cap_len_iden.h5'

class CaptchaEval:

  def __init__(self):
    # load the trained model
    self.model = Sequential()
    print ("loading the trained model")
    self.model = model_from_json(open(jfile).read())  
    self.model.load_weights(wfile)
    adam = optimizers.adam(lr=0.0005)
    self.model.compile(loss='categorical_crossentropy',
		              optimizer=adam,
		              metrics=['accuracy'])

  def predict_from_img(self, img):
    X_test = get_x_input_from_image(img)

    predict = self.model.predict(X_test)

    text = ''
    print(predict[0,])
    text=get_max(predict[0,])
    return text


with open('results.csv', mode='w') as csvfile:
  pth="./data/gen"
  captchaEval = CaptchaEval()
  for fn in os.listdir(pth):
    pfn=''.join([pth, "/", fn])
    print(pfn)
    st = time.time()
    try:
      with open(pfn, mode='rb') as file: 
        fileContent = file.read()
      stream = io.BytesIO(fileContent)
      localImage = Image.open(stream)
      text = captchaEval.predict_from_img(localImage)
    except:
      pass

    cw=csv.writer(csvfile,delimiter=',')
    cw.writerow([fn, text, time.time()-st])
print("done")