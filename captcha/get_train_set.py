from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import captcha_params

height_p = 60
width_p = 120

# generate  the captcha text randomly from the char lists above
def random_captcha_text(char_set=captcha_params.get_char_set(), captcha_size=captcha_params.get_captcha_size()):
	captcha_text = []
	for i in range(captcha_size):
		c = random.choice(char_set)
		captcha_text.append(c)
	return captcha_text
 
# generate the captcha text and image and save the image 
def gen_captcha_text_and_image(i):
	image = ImageCaptcha(width=width_p, height=height_p, font_sizes=[30])
 
	captcha_text = random_captcha_text()
	captcha_text = ''.join(captcha_text)

	path = './data/test/'
	if os.path.exists(path) == False: # if the folder is not existed, create it
		os.mkdir(path)
                
	captcha = image.generate(captcha_text)

	# naming rules: num(in order)+'_'+'captcha text'.include num is for avoiding the same name
	image.write(captcha_text, path+str(i)+'_'+captcha_text + '.png') 
 
	captcha_image = Image.open(captcha)
	captcha_image = np.array(captcha_image)
	return captcha_text, captcha_image
 
if __name__ == '__main__':

        
        for i in range(2000):     
                text, image = gen_captcha_text_and_image(i)


