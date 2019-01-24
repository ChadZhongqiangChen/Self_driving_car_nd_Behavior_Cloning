**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* generator.py containing a generator function used in the model to read in, store and provide batches of data
* drive.py for driving the car in autonomous mode
* video.py for generate a video from continuous shooting images
* flip_image.py to show flipped (mirror) image
* model2.h5 containing a trained convolution neural network, which can be used to run both Track 1 and Track 2
* Behavioral_Cloning_Writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model2.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes in depths 24, 36, and 48 (model.py lines 93-95) , and 3x3 filter sizes in depths 64 (model.py lines 96 and 97).

The model includes RELU layers to introduce nonlinearity (model.py lines 93-97), and linear layers to optain the final output (model.py lines 99-107).

The data is normalized in the model using a Keras lambda layer (model.py line 87), and cropped using a Keras Cropping2D layer (model.py line 89).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line 100). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 80-81 and 125-129). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 125).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and reverse directon center lane driving.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was that the model was able to learn from my driving skills and operate a vehicle on the track autonomously. 

My first step was to use a convolution neural network model similar to the NVIDIA model structure mentioned in their paper "End to end learning for self-driving car". I thought this model might be appropriate because the NVIDIA model was able to drive a vehicle on the track autonomously after training.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

My first model was model0.h5. I found that my first model had a very low mean squared error (loss was 0.0025) on the training set but a high mean squared error on the validation set (loss was 0.024) . This implied that the model was overfitting. The Loss vs. Epoch graph is showed as the following. 

![model0_Loss_epoch](./Figures/model0_Loss_epoch.jpg "model0_Loss_epoch")

I ran the simulator, and it turned out that the vehicle was able to ran in the first track without felling off the road, aand it was able to kept in the center of the lane for most of the time . However, it felled off the road in the seoncd track soon after it started.

To combat the overfitting, I increased the dropout rate in the dropout layer, and provided more training and validation data about the second track. In my model, a large batch_size would lead to a longer training time. Due to the GPU quota limitation, I had to reduce the batch_size to half (from 32 to 16), and trained the model for 3 epochs, which took about three hours.

This second model was model1.h5. The Loss vs. Epoch is showed as the following. The training loss is around 0.025, and the validation loss is around 0.059. Both both train and validation error loss are higher than the first model, and train loss is still much lower than validation loss. And this implied that overfitting still existed.

![model1_Loss_epoch](./Figures/model1_Loss_epoch.jpg "model1_Loss_epoch")

I ran the simulator, and it turned out that vehicle was able to ran in the second track without felling off the road. However, in the first track, it fell off the road right before the stone bridge, where road surface texture changed.

To overcome the overfitting, I decided to train the model for longer time. The training epoch was increased from 3 to 5. Due to the GPU quota limitation, I decided to further reduce batch_size, from 16 to 4. The training took about one hour. 

This third model was model2.h5. The Loss vs. Epoch is showed as the following graph. The train loss is around 0.040, and the validation loss is around 0.060. Both train and validation loss are higher than the second model. train loss is still lower than validation loss. And this implied that overfitting still existed, but had been reduced.

![model2_Loss_epoch](./Figures/model2_Loss_epoch.jpg "model2_Loss_epoch")

Then I ran the simulator for the third model model2.h5. The performance turned out to be satisfactory. 

In the first track, the vehicle was able to keep on track all the time, and keep in the center of the lane for most of time. 

In the second track, the vehicle was able to keep on track almost all the time, and there were only two spots where the vehicle drove on the edge of the road, but it sucessfully recovered to the center of the lane.

In conclusion, at the end of the process, the vehicle model model2.h5 is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 93-107) consisted of a convolution neural network with the following layers and layer sizes. There was one Keras lamda layer for normalization and one Keras cropping2D layer for cropping images. And there was also one dropout layer between flatten layer and the first fully connected layer.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![Network Architecture](/figures/Network_Architecture.jpg "Model Visualization")

#### 3. Creation of the Training Set 

##### 3.1. Collection of Images

To capture good driving behavior, using center lane driving, I first recorded four laps on track one for clock direction, and four laps on track one for counter-clock direction. Here is an example image of center lane driving:

The first one is from center camera.
![Center Lane Driving](/figures/Center_lane_driving.jpg "Center Camera Center Lane Driving")

This one is from left camera.
![Left Center Lane Driving](/figures/Left_center_lane_driving.jpg "Left Camera Center Lane Driving")

This one is from right camera
![Right Center Lane Driving](/figures/Right_center_lane_driving.jpg "Right Camera Center Lane Driving")

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to return back to center lane once it drives close to the edges. These images show what a recovery looks like starting from the sides of the road :

These three are center, left and right images for recovering from left side of the road.

![Recover from Left](/figures/Recover_from_left.jpg "Center Camera Recovering from Left")

![Left Recover from Left](/figures/Left_Recover_from_left.jpg "Left Camera Recovering from Left")

![Right_Recover from Left](/figures/Right_Recover_from_left.jpg "Right Camera Recovering from Left")

The next three are center, left and right images for recovering from right side of the road.

![Recover from Right](/figures/Recover_from_right.jpg "Center Camera Recovering from Right")

![Left Recover from Right](/figures/Left_Recover_from_right.jpg "Left Camera Recovering from Right")

![Right_Recover from Right](/figures/Right_Recover_from_right.jpg "Right Camera Recovering from Right")

Then I recorded three laps on track two (mountain track) in one direction to get more data points.

The next three are center, left, and right images for mountain track driving

![Center Mountain](/figures/Center_mountain.jpg "Center Camera Mountain Driving")

![Left Mountain](/figures/Left_mountain.jpg "Left Camera Mountain Driving")

![Right_Mountain](/figures/Right_mountain.jpg "Right Camera Mountain Driving")

To augment the data set, I also flipped images thinking that this would help generalize the model. For example, here is an image that has then been flipped.

This is the original image.
![Center Lane Driving](/figures/Center_lane_driving.jpg "Center Camera Center Lane Driving")

This is the flipped (mirror) image. 
![Mirror Center Lane Driving](/figures/Mirror_Center_lane_driving.jpg "Mirror Center Camera Center Lane Driving")

The next two are the corresponding flipped left and right images.

![Mirror Left Center Lane Driving](/figures/Mirror_Left_center_lane_driving.jpg "Mirror Left Camera Center Lane Driving")

![Mirror Right Center Lane Driving](/figures/Mirror_Right_center_lane_driving.jpg "Mirror Right Camera Center Lane Driving")

##### 3.2. Collection of Steering Angles

In autonomous driving mode, vehicle steering angle would be predicted based on center image. 

In traning process, left and right images would be treated as center image. Thus left and right angle measurements would need to be corrected as if they were based on center camera as well. The following figure from Udacity illustrates the differences of angles between destination and each camera.

![Angles between Destination and Each Camera](/figures/Angles_between_Destination_and_Each_Camera.jpg "Angles between Destination and Each Camera")

The correction measurement number was taken as 0.2. 

For left camera, left_angle = recorded_angle + 0.2. For right camera, right_angle = recorded_angle - 0.2

For mirrored left and right images, the corresponding angles would also be taken negative after corrections.

#### 4. Training Process

All the center, left and right images were used for training. 

To help reduce memory usage, I used a generator to read in, store and provide batches of data to the model. For details, please see generator.py. 

The generator function also provided options to read in left and right images (generator.py code lines 30-53), as well as flipping images (generator code lines 57-71) , which were all implemented in this model.

I finally randomly shuffled the data set and put 20% of the data into a validation set. The training set was used to train the model, and the validation set helped determine if the model was over or under fitting.

The batch_size was set as 4, and there were total 20228 data points for training, and 5057 data points for validation.

An appropriate number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.
