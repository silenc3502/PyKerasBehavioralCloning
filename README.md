# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a NvidiaCNN with Keras to predicts steering angles from images
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report
---

[//]: # (Image References)

[orig]: ./pic_data/center_2017_10_27_16_00_44_798.jpg "Center"
[left]: ./pic_data/left_2017_10_27_16_00_44_798.jpg "Left"
[right]: ./pic_data/right_2017_10_27_16_00_44_798.jpg "Right"
[shadow]: ./pic_data/add_shadow.jpg "Add Shadow"
[BGR2YUV]: ./pic_data/aug_BGR2YUV.jpg "BGR2YUV"
[bright]: ./pic_data/bright.jpg "Brightness"
[crop]: ./pic_data/crop.jpg "Crop"
[flip]: ./pic_data/flip_img.jpg "Flip"
[horizon_shift]: ./pic_data/horizon_shift.jpg "Horizon Shift"
[resize]: ./pic_data/resize.jpg "Resize"
[hist]: ./pic_data/Histogram.jpg "Hist"

### Core Files
---
* temp.py: Histogram Analysis for Steering Angle
* preprocess.py: Preprocessing and Augmenting before Training for track 2
* model.py: Implementation of the Model Architecture(NvidiaCNN) for track 2
* make_train.py: Make Model for track 1
* drive.py: for track 1(no modify)
* drive2.py: for track 2(modify Coefficient of PI Controller & add multiflying factor of steering angle)
* video.py: video recording
* run1.mp4: driving track 1
* run2.mp4: driving track 2
---

### How to execute it ?
#### Track 1
---
1. python make_train.py

This command will be create trained data for steering.

2. Run your Car Simulator with autonomous driving mode.

3. python drive.py model.h5

Now you can see the autonomous driving for track 1

---

#### Track 2
---
1. python preprocess.py

This command will be create x_train and y_train with training mode data of car simulator.

2. python model.py

This command will be make best_model.h5

3. Run your Car Simulator with autonomous driving mode.

4. python drive2.py best_model.h5

Now you can see the autonomous driving that trained by Keras(track 2).

---

### Details of Code
#### preprocess.py
---
There are 7 functions(include main):
7 functions are add_shadow, shift_horizon, crop, brightness, augment_img, generate_training_data, and main.

* main function:

Call generate_training_data to create x train and y train.

* Generate_training_data function:

Read csv file to understand driving log.
Driving log has center, left, right images and steering angle, throttle, break, speed data.
Now, get image that already finished brightness, shift_horizon, and add_shadow rpcoess.
And crop image for our intereseting area with BGR to YUV converting images.
Resize this images for processing.

np.fliplr() change images phase(180 degree) and we add it too.
Now get steering data from 3 lines(data which one has steering info with images).
And calculating steering angle with upon information.
Currently, success to make x train and y train.

* augment_img function:

Applying brightness, shift_horizon, and add_shadow to images.

* brightness function:

Convert BGR image to HSV and give some random brightness value to V factor.
And make clipping data that range is 0 to 255.
After convert HSV to BGR.
This is for get various brightness images.

* crop function:

Cropping Image for processing ROI.

* shift_horizon function:

This is the technique of my first project that is undistortion.
make image to be straight to proper processing.

* add_shadow function:

Add shadow image to make train data more robust.

---

#### model.py
---
It has ELU(Exponential Linear Unit) Model Architecture.
First Normalization with input_shape=(66, 200, 300) and doing CNN.
CNN has(24, (5, 5)), (36, (5, 5)), (48, 5, 5)), (64, (3, 3)), (64, (3, 3)).
After creating CNN than have to do make Fully-connected layer.
However my computing performance is not so good.
So I remove Dense(1164) part and just do Dense(100), Dense(50), Dense(10), Dense(1).
It's not perfect but also good.

---

### Data Collection Strategy
---
* drive 1 lap in the middle of the second lane clockwise
* drive 1 lap in the middle of the second lane counter-clockwise
* Add some driving data to solve problem(fall, curve lane fail, don't go straight)

![alt_text][orig]
![alt_text][crop]

Moreover, images coming from all 3 cameras was used with a correction of 0.2.

---

### Data Augmentation
---
Already explain it above for what the code do.
There are several augmentation strategies to make algorithm more robust.

* Flip Image
* Random Brightness
* Add Random Shadow
* Horizon Shift

Let's check Distribution of data.

Final Distribution of data looks like below.

---

### Simulation
#### Track 1
---

---

#### Track 2
---

---

### Future Works
---
I will try it with My TI AM5728 with OpenCL(Based on C++).
And that case I'll try to use Full Nvidia CNN.
AM5728 has Imagination SGX544.
So, maybe it can do it with C6000 DSP.

---

### References

1. https://github.com/jaeoh2/self-driving-car-nd/tree/master/CarND-Behavioral-Cloning-P3
2. https://github.com/joekidd/CarND-Behavioral-Cloning-P3
3. https://github.com/Markusgami/CarND-Behavioral-Cloning
4. https://arxiv.org/pdf/1604.07316v1.pdf
5. https://github.com/SvenMuc/CarND-Behavioral-Cloning-P3
6. https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project
7. https://datascienceschool.net/view-notebook/1d93b9dc6c624fbaa6af2ce9290e2479/
8. https://keras.io/getting-started/sequential-model-guide/
9. https://keras.io/models/model/
10. https://stackoverflow.com/questions/42666046/loading-a-trained-keras-model-and-continue-training
11. http://idiap.ch/~katharas/importance-sampling/training/
