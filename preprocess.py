import csv
import numpy as np
import cv2
import random
import scipy.misc
import sys

# set this parameter to provide left/right camera correction
correction = 0.2
correction_multipliers = [0, 1.0, -1.0]

# methods to be used for augmentation
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

def rotate_img(img, s_angle=-10, end_angle=10):
    tmp = scipy.misc.imresize(img, 1.2)
    angle = random.randint(s_angle, end_angle)
    tmp = scipy.misc.imrotate(tmp, angle)
    crop = int(abs(1.1 * angle)) + 1
    return scipy.misc.imresize(tmp[crop:-crop, crop:-crop,], img.shape)

def brightness(img, value=0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype('int32')
    hsv[:,:,2] += value
    hsv = np.clip(hsv, 0, 255)
    hsv = hsv.astype('uint8')
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def augment_img(img):
    aug_img = brightness(img, random.randint(-20, 20))
    aug_img = shift_horizon(aug_img)
    aug_img = add_shadow(aug_img)
    return aug_img

def generate_training_data():#use_generator=False, batch_size=128):
    lines = []
    with open("/home/sdr/self_drive/letsgo/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurementes = []
    for line in lines:
        # for each image from every camera
        for i in range(3):
            # load the image
            source_path = line[i]
            filename = source_path.split('/')[-1]
            path = "/home/sdr/self_drive/letsgo/IMG/"
            current_path = path + filename
            try:
                org = cv2.imread(current_path)
                augmented = augment_img(org)
                for img in [org, augmented]:
                    # follow the nvidia paper
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                    img = crop(img, 50, 20, 0, 1)
                    img = cv2.resize(img, (200, 66), interpolation = cv2.INTER_AREA)

                    # append the image to the images list
                    images.append(img)

                    # add also flipped image
                    image_flipped = np.fliplr(img)
                    images.append(image_flipped)

                    # add measurements
                    steering = float(line[3])
                    steering += correction * correction_multipliers[i]
                    if steering > 1.0:
                        steering = 1.0
                    elif steering < -1.0:
                        steering = -1.0

                    # add steering and flipped steering
                    measurementes.extend([steering, -steering])

                    #if len(images) == batch_size and use_generator:
                    #    yield np.array(images), np.array(measurementes)
                    #    images = []
                    #    measurementes = []
            except:
                print("Error for: {}".format(current_path))
                print(sys.exc_info())
                sys.exit(-1)

    # if not use_generator:
    X_train = np.array(images)
    y_train = np.array(measurementes)

    np.save("x_train", X_train)
    np.save("y_train", y_train)


if __name__ == "__main__":
    generate_training_data()
    print("Done")
