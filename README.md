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

### Core Files
---
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

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

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
