import argparse
from captcha.image import ImageCaptcha  
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import captcha_params

height_p = 60
width_p = 120

# generate the captcha text randomly from chars from captcha_params.py
def random_captcha_text(char_set=captcha_params.get_char_set()):
	captcha_size=opt.cap_len
	captcha_text = []
	for i in range(captcha_size):
		c = random.choice(char_set)
		captcha_text.append(c)
	return captcha_text
 
# generate the captcha image and save the image 
def gen_captcha_text_and_image(i):
	image = ImageCaptcha(width=width_p, height=height_p, font_sizes=[30])
 
	captcha_text = random_captcha_text()
	captcha_text = ''.join(captcha_text)

	path = './data/gen/'
	if os.path.exists(path) == False: 
		os.mkdir(path)
                
	captcha = image.generate(captcha_text)

	image.write(captcha_text, path+'L'+str(opt.cap_len)+"-"+str(i)+'_'+captcha_text + '.png') 
 
	captcha_image = Image.open(captcha)
	captcha_image = np.array(captcha_image)
	return captcha_text, captcha_image
 
if __name__ == '__main__':
	aparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	aparser.add_argument('--cap_len', type=int, default=4, help='the length of digit captcha')
	aparser.add_argument('--gen_size', type=int, default=5000, help='the number of captcha generated')
	opt = aparser.parse_args()
	for i in range(opt.gen_size):     
		text, image = gen_captcha_text_and_image(i)
		print(text)