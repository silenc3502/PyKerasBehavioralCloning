import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import cv2
import random

def load_data(fpath):
	dataset = []
	with open(fpath) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			try:
				dataset.append({'center':line[0], 'left':line[1], 'right':line[2],
								'steering':float(line[3]), 'throttle':float(line[4]),
								'brake':float(line[5]), 'speed':float(line[6])})
			except:
				continue

	return dataset


def plot_steering_hist(steerings, title, num_bins=100):
        plt.hist(steerings, num_bins)
        plt.title(title)
        plt.xlabel('Steering Angles')
        plt.ylabel('Images')
        plt.show()

def plot_dataset_hist(dataset, title, num_bins=100):
        steerings = []
        for item in dataset:
                steerings.append(float(item['steering']))
        plot_steering_hist(steerings, title, num_bins)

def add_shadow(img):
	new_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	h,w = new_img.shape[0:2]
	mid = np.random.randint(0,w)
	factor = np.random.uniform(0.6,0.8)
	if np.random.rand() > .5:
		new_img[:,0:mid,0] = (new_img[:,0:mid,0] * factor).astype('uint8')
	else:
		new_img[:,mid:w,0] = (new_img[:,mid:w,0] * factor).astype('uint8')
	new_img = cv2.cvtColor(new_img, cv2.COLOR_YUV2BGR)
	return new_img

def shift_horizon(img):
	h, w, _ = img.shape
	horizon = 2 * h / 5
	v_shift = np.random.randint(-h/8,h/8)
	pts1 = np.float32([[0,horizon],[w,horizon],[0,h],[w,h]])
	pts2 = np.float32([[0,horizon+v_shift],[w,horizon+v_shift],[0,h],[w,h]])
	M = cv2.getPerspectiveTransform(pts1,pts2)
	return cv2.warpPerspective(img,M,(w,h), borderMode=cv2.BORDER_REPLICATE)

def crop(img, c_lx, c_rx, c_ly, c_ry):
	return img[c_lx:-c_rx, c_ly:-c_ry,]

def brightness(img, value=0):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	hsv = hsv.astype('int32')
	hsv[:,:,2] += value
	hsv = np.clip(hsv, 0, 255)
	hsv = hsv.astype('uint8')
	return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def augment_img(img):
	aug_img = brightness(img, random.randint(-20, 20))
	plt.imshow(aug_img)
	plt.savefig('bright.jpg', format='jpg')
	aug_img = shift_horizon(aug_img)
	plt.imshow(aug_img)
	plt.savefig('horizon_shift.jpg', format='jpg')
	aug_img = add_shadow(aug_img)
	plt.imshow(aug_img)
	plt.savefig('add_shadow.jpg', format='jpg')
	return aug_img

old_fpath = "/home/sdr/Pictures/test/center_2017_10_27_16_00_44_798.jpg"
fpath = "/home/sdr/self_drive/letsgo/driving_log.csv"
"""
orig = cv2.imread(fpath)
augment = augment_img(orig)

for img in [orig, augment]:
	img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	plt.imshow(img)
	plt.savefig('aug_BGR2YUV.jpg', format='jpg')

	img = crop(img, 50, 20, 0, 1)
	plt.imshow(img)
	plt.savefig('crop.jpg', format='jpg')

	img = cv2.resize(img, (200, 66), interpolation = cv2.INTER_AREA)
	plt.imshow(img)
	plt.savefig('resize.jpg', format='jpg')

	image_flip = np.fliplr(img)
	plt.imshow(image_flip)
	plt.savefig('flip_img.jpg', format='jpg')
"""

dataset = load_data(fpath)

plot_dataset_hist(dataset, 'Num of Images per S.A before Image Augmentation', num_bins=100)
