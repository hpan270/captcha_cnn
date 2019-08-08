from keras import models, layers, optimizers, losses
import os
import sys
import numpy as np
from util import util
from keras.callbacks import ModelCheckpoint


def initialize(self, opt):
	self.opt = opt
	self.batchSize = opt.batchSize
	self.keep_prob = opt.keep_prob
	self.save_dir = os.path.join(opt.checkpoints_dir, opt.cap_scheme, str(opt.train_size))
	self.model = self.define_lenet5()

	if self.opt.isTrain:
		adam = optimizers.adam(lr=opt.lr)
		self.model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])
	if self.opt.callbacks:
		bast_model_name = os.path.join(self.save_dir,
									   self.opt.cap_scheme + '-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
		checkpoint = ModelCheckpoint(bast_model_name,
									 monitor='val_acc', verbose=1, save_best_only=True, mode='max')
		self.callbacks_list = [checkpoint]
	else:
		self.callbacks_list = None

def define_3conv2d(self):
	model = models.Sequential()
	# 3 Convolutional Layers
	model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same',
							input_shape=(self.opt.loadHeight, self.opt.loadWidth, 1)))
	model.add(layers.Activation('relu'))
	model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(layers.Dropout(self.keep_prob))

	model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
	model.add(layers.Activation('relu'))
	model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(layers.Dropout(self.keep_prob))

	model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
	model.add(layers.Activation('relu'))
	model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(layers.Dropout(self.keep_prob))

	# Fully Connected Layers
	model.add(layers.Flatten())
	model.add(layers.Dense(2048))
	model.add(layers.Activation('relu'))
	model.add(layers.Dropout(self.keep_prob))

	model.add(layers.Dense(self.opt.cap_len * self.opt.char_set_len))
	model.add(layers.Activation('softmax'))
	# model.summary()
	return model


def fit_generator(self,
				  generator,
				  steps_per_epoch=20,
				  epochs=20,
				  validation_data=None,
				  validation_steps=None,
				  class_weight='auto',
				  callbacks = None
				  ):
	return self.model.fit_generator(
		generator=generator,
		steps_per_epoch=steps_per_epoch,
		epochs=epochs,
		validation_data=validation_data,
		validation_steps=validation_steps,
		class_weight=class_weight,
		callbacks=callbacks
	)


def save_model(self):
	model_checkpoint_base_name = os.path.join(self.save_dir, self.opt.cap_scheme + '.model')
	self.model.save(model_checkpoint_base_name)


def load_weight(self):
	model_checkpoint_base_name = os.path.join(self.save_dir, self.opt.base_model_name)
	self.model.load_weights(model_checkpoint_base_name)

def save(self, history):
	print('Print the training history:')
	with open(self.save_dir + '/cnn_train.txt', 'a') as opt_file:
		opt_file.write(str(history.history))
	self.save_model()

 
