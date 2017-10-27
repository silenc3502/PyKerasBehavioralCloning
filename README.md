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

[run1]: ./pic_data/run1.gif "Track 1"
[run2]: ./pic_data/run2.gif "Track 2"

### Core Files
---
* temp.py: Histogram Analysis for Steering Angle
* preprocess.py: Preprocessing and Augmenting before Training for track 2(Explain it below section)
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
Below Image represents Nvidia's CNN Model Architecture and I remove Dense(1164) part.
Because of my computer get so big latency.

<img src="https://raw.githubusercontent.com/silenc3502/PyKerasBehavioralCloning/master/pic_data/nvidia_cnn_model.png"/>

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

<table>
<tr>
<td><img src="https://raw.githubusercontent.com/silenc3502/PyKerasBehavioralCloning/master/pic_data/center_2017_10_27_16_00_44_798.jpg"/></td>
<td> Original Image </td>
</tr>
<tr>
<td><img src="https://raw.githubusercontent.com/silenc3502/PyKerasBehavioralCloning/master/pic_data/crop.jpg"/></td>
<td> Cropping ROI after Preprocessing </td>
</tr>
</table>

Moreover, images coming from all 3 cameras was used with a correction of 0.2.

<table>
<tr>
<td><img src="https://raw.githubusercontent.com/silenc3502/PyKerasBehavioralCloning/master/pic_data/left_2017_10_27_16_00_44_798.jpg"/></td>
<td> Left Camera Image </td>
</tr>
<tr>
<td><img src="https://raw.githubusercontent.com/silenc3502/PyKerasBehavioralCloning/master/pic_data/center_2017_10_27_16_00_44_798.jpg"/></td>
<td> Center Camera Image </td>
</tr>
<tr>
<td><img src="https://raw.githubusercontent.com/silenc3502/PyKerasBehavioralCloning/master/pic_data/right_2017_10_27_16_00_44_798.jpg"/></td>
<td> Right Camera Image </td>
</tr>
</table>

---

### Data Augmentation
---
Already explain it above for what the code do.
There are several augmentation strategies to make algorithm more robust.

* Flip Image
* Random Brightness
* Add Random Shadow
* Horizon Shift

Each strategies are like below.

<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/silenc3502/PyKerasBehavioralCloning/master/pic_data/bright.jpg"/>
</td>
<td> Random Brightness </td>
<td>
<img src="https://raw.githubusercontent.com/silenc3502/PyKerasBehavioralCloning/master/pic_data/horizon_shift.jpg"/>
</td>
<td> Horizon Shift </td>
</tr>
<tr>
<td>
<img src="https://raw.githubusercontent.com/silenc3502/PyKerasBehavioralCloning/master/pic_data/add_shadow.jpg"/>
</td>
<td> Add Random Shadow </td>
<td>
<img src="https://raw.githubusercontent.com/silenc3502/PyKerasBehavioralCloning/master/pic_data/aug_BGR2YUV.jpg"/>
</td>
<td> BGR2YUV </td>
</tr>
<tr>
<td>
<img src="https://raw.githubusercontent.com/silenc3502/PyKerasBehavioralCloning/master/pic_data/resize.jpg"/>
</td>
<td> Resize </td>
<td>
<img src="https://raw.githubusercontent.com/silenc3502/PyKerasBehavioralCloning/master/pic_data/flip_img.jpg"/>
</td>
<td> Flip </td>
</tr>
</table>

Final Distribution of data looks like below.

<img src="https://raw.githubusercontent.com/silenc3502/PyKerasBehavioralCloning/master/pic_data/Histogram.png"/>

It looks like Gaussian Distribution that is good!
However it has outlier because of I drive it with full steering and full throttle very many times at training.

---

### Simulation
#### Track 1
---

![alt_text][run1]

---

#### Track 2
---

![alt_text][run2]

---

### Conclusions
---
I can understand why people say big data.
Because of when I do it with small data then I can get the terrible result(fall down, assult to wall or rock, don't go straight, and so on).
When I take enough data for learning then the results are also good too.

So I think most important thing in deep learning is big data.
And this data must have good correction for avoid error.
If I give many data that is bad correction then the results are also terrible.

---

### Future Works
---
I will try it with My TI AM5728 with OpenCL(Based on C++).
And that case I'll try to use Full Nvidia CNN.
AM5728 has Imagination 2 x SGX544 Graphic Cards, 2 x ARM Cortex-A15, Video Codec, 2 x Cortex-M4, and 2 x C6678 DSPs.
So, if I use above resource very well with OpenCL then maybe I can do this work with AM5728 too.

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
12. https://github.com/SvenMuc/CarND-Behavioral-Cloning-P3
13. https://medium.com/@jmitchell1991/behavioral-cloning-self-driving-car-simulation-14531358c87e
14. http://image-net.org/challenges/posters/JKU_EN_RGB_Schwarz_poster.pdf
15. https://medium.com/udacity/udacity-self-driving-car-nanodegree-project-3-behavioral-cloning-446461b7c7f9
16. https://arxiv.org/pdf/1604.04112.pdf
17. https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
18. https://medium.com/towards-data-science/selu-make-fnns-great-again-snn-8d61526802a9
19. https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
20. http://opencv-python.readthedocs.io/en/latest/doc/10.imageTransformation/imageTransformation.html
21. https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.ndarray.html
22. https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.fliplr.html
23. https://en.wikipedia.org/wiki/HSV-2
24. https://keras.io/callbacks/
25. https://keras.rstudio.com/articles/training_visualization.html
26. http://www.coldvision.io/2017/02/15/predicting-steering-angles-using-deep-learning-and-behavioral-cloning/
