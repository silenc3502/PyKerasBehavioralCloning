import csv
import cv2
import numpy as np

path_to_image_data = '/home/sdr/self_drive/IMG/'

lines = []
with open('/home/sdr/self_drive/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = '/home/sdr/self_drive/IMG/' + filename
		image = cv2.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		measurements.append(measurement)

x_train = np.array(images)
y_train = np.array(measurements)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image, 1))
	augmented_measurements.append(measurement * -1.0)

# -t NvidiaFull
# -d /home/sdr/self_drive/driving_log.csv
# -ne (number of epoches)
# -ns (number of samples)
# learning rate = 0.001
#                train_model(model=args.train_model,
#                            dataset_csv_filename=args.dataset_filename,
#                            nb_epochs=int(args.number_epochs),
#                            nb_samples_per_epoch=int(args.number_samples_epoch), learning_rate=lr)

import pickle
import sys
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#from networks.LeNet import LeNet
#from networks.NvidiaFull import NvidiaFull
#from networks.NvidiaLight import NvidiaLight
#from networks.VGG import VGG

from keras.models import Sequential
from keras.layers import Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout

import DataAugmentation as da
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

BATCH_SIZE = 256
ROI = [20, 60, 300, 138]

def prepare_datasets(csv_filename):
	samples = []
	with open(csv_filename) as csv_file:
		reader = csv.reader(csv_file)
		next(reader, None)

		for line in reader:
			samples.append(line)

	train_samples, validation_samples = train_test_split(shuffle(samples), test_size=0.2)

	return train_samples, validation_samples

def generator(samples, batch_size=128, augment=True):
	samples = shuffle(samples)

	nb_samples = len(samples)
	nb_total_samples = 0

	while 1:  # loop forever so the generator never terminates
		for offset in range(0, nb_samples, batch_size):
			batch_samples = samples[offset:offset + batch_size]

			nb_batch_samples = 0
			images = []
			angles = []
			crop_size = (64, 64)

			for batch_sample in batch_samples:
				angle_center = float(batch_sample[3])

				# add 60% augmented images and 40% recorded images
				if augment and np.random.rand() <= 0.6:
					# add random center, left or right image
					clr = np.random.randint(low=0, high=3)

					if clr == 0:
						# add center image
						image = cv2.imread(path_to_image_data + '/' + batch_sample[0].lstrip())
						angle = angle_center
					elif clr == 1:
						# add left image
						image = cv2.imread(path_to_image_data + '/' + batch_sample[1].lstrip())
						angle = angle_center + 6.25 / 25.0
					elif clr == 2:
						# add right image
						image = cv2.imread(path_to_image_data + '/' + batch_sample[2].lstrip())
						angle = angle_center - 6.25 / 25.0

					if image == True:
						image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
					else:
						continue

					# apply random translation
					image, angle = da.DataAugmentation.random_translation(image, angle, [70, 30], probability=0.5)

					# apply random perspective transformation
					# TODO: image, angle = da.DataAugmentation.random_perspective_transformation(image, angle, [100, 50], probability=0.5)

					# apply shadow augmentation
					image = da.DataAugmentation.random_shadow(image, probability=0.5)

					# apply random flip, lr_bias = 0.0 (no left/right bias correction of dataset)
					image, angle = da.DataAugmentation.flip_image_horizontally(image, angle, probability=0.5, lr_bias=0.0)

					# apply random brightness
					image = da.DataAugmentation.random_brightness(image, probability=0.5)

				else:
					# add center image
					image = cv2.imread(path_to_image_data + '/' + batch_sample[0].lstrip())
					if image == True:
						image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
					else:
						continue
					angle = angle_center

				# final image pre-processing: improve contrast by histogram equalization and convert to HSV
				# TODO: image = da.DataAugmentation.equalize_histogram(image)
				image = da.DataAugmentation.crop_image(image, self.roi, crop_size)
				# TODO: image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

				images.append(image)
				angles.append(angle)
				nb_batch_samples += 1

				if self.verbose == 2:
					# show generated images in a separate window
					if image == True:
						image_gen = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
					else:
						continue
					da.DataAugmentation.draw_steering_angles(image_gen, steering_angle=angle)
					cv2.imshow('Generator output', image_gen)
					cv2.waitKey(100)

			nb_total_samples += nb_batch_samples

#			if self.verbose > 0:
#				print(' Generator: nb_batch_samples: {:4d} nb_total_samples: {:5d}/{:5d}'.format(nb_batch_samples, nb_total_samples, nb_samples))

		# convert to numpy arrays
		x_train = np.array(images)
		y_train = np.array(angles)
		yield shuffle(x_train, y_train)

train_samples, validation_samples = prepare_datasets('/home/sdr/self_drive/driving_log.csv')

#network = NvidiaCNN(3, 64, 64, 1, ROI, 6.25)

model = Sequential()

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(64, 64, 3)))

model.add(Convolution2D(3, 5, 5, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(24, 5, 5, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(36, 5, 5, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(48, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))

model.add(Dense(1))

#model.compile(loss='mse', optimizer='adam')
#model.fit(x_train, y_train, validation_split=0.1, shuffle=True, nb_epoch=2)

batch_size = 128

nb_train_samples = len(train_samples)
nb_validation_samples = len(validation_samples)

train_generator = generator(train_samples, batch_size=batch_size, augment=True)
validation_generator = generator(validation_samples, batch_size=batch_size, augment=True)

#train(nb_epoches=7, nb_samples_per_epoch=20.000, learning_rate=0.001)

checkpoint = ModelCheckpoint('NvidiaCNN' + '_checkpoint_{epoch:02d}_{val_loss:.4f}.h5', monitor='val_loss', verbose=1, save_best_only=False, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=True)

print('checkpoint')
print(checkpoint)
print('early_stopping')
print(early_stopping)

optimizer = adam(lr=0.001)
model.compile(optimizer=optimizer, loss='mse')
history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=20.000,
                                     validation_data=validation_generator,
                                     nb_val_samples=nb_validation_samples,
                                     nb_epoch=7,
                                     verbose=1,
                                     callbacks=[checkpoint, early_stopping])

model.save('model.h5')
exit()
