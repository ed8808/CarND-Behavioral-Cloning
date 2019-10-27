# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-architecture-362x516.png "Nvidia CNN -- Deep Learning Self Driving Car"
[image2]: ./examples/center_2019_10_25_22_51_08_242.jpg "Center Lane Image"
[image3]: ./examples/left_2019_10_25_22_51_08_242.jpg "Left side Image"
[image4]: ./examples/right_2019_10_25_22_51_08_242.jpg "Right side Image"
[image5]: ./examples/center_2019_10_25_22_51_08_242.jpg "Normal Image"
[image6]: ./examples/center_2019_10_25_22_51_08_242.jpg "Flipped Image"
[image7]: ./examples/center_2019_10_25_22_51_08_242c.jpg "Cropped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (model.py lines 97-106) 

The model includes RELU layers to introduce nonlinearity (code line 97-101), and the data is normalized in the model using a Keras lambda layer (code line 82). 

#### 2. Attempts to reduce overfitting in the model

The model does not contain dropout layers in order to reduce overfitting because training loss and validation loss are quite closed at last 5th epoch

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 12). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 108).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of 
i) center lane driving, 
ii) recovering from the left and right sides of the road
iii) driving clockwise
iv) flipping left and right camera images and reversing corresponding steering direction

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to adopt with the conventional Nvidia CNN for self driving car deep learning.

My first step was to use a convolution neural network model similar to the LeNet I thought this model might be appropriate because LeNet is a powerful CNN.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

I then switched to Nvidia CNN to compare, the result was satisfactory.  
Here are the 5 epochs with training loss and validation loss

458/458 [==============================] - 138s 300ms/step - loss: 0.1743 - val_loss: 0.0243
Epoch 2/5
458/458 [==============================] - 135s 294ms/step - loss: 0.0231 - val_loss: 0.0214
Epoch 3/5
458/458 [==============================] - 135s 295ms/step - loss: 0.0210 - val_loss: 0.0204
Epoch 4/5
458/458 [==============================] - 135s 295ms/step - loss: 0.0199 - val_loss: 0.0197
Epoch 5/5
458/458 [==============================] - 136s 296ms/step - loss: 0.0193 - val_loss: 0.0197

Apart from first epochs, the training loss and validation loss were quite closed, there is not much overfitting or underfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I used the default training set together with training sets of non-center line driving and opposite clockwise driving.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 97-106) Nvidia consisted of a convolution neural network with the following layers and layer sizes. It has about 27 million connections and 250 thousand parameters.
Conv2D(24,5,5,subsample=(2,2),activation="relu")
Conv2D(36,5,5,subsample=(2,2),activation="relu")
Conv2D(48,5,5,subsample=(2,2),activation="relu")
Conv2D(64,3,3,activation="relu")
Conv2D(64,3,3,activation="relu")
Flatten()
Dense(100)
Dense(50)
Dense(10)
Dense(1)

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

Etc ....

After the collection process, I had 18321 number of data points. I then preprocessed this data by normalization (value / 255 ) - 0.5 so that it is in range between -0.5 and 0.5.  Then the images are cropped to remove upper and lower parts of images to reduce influence to deep learning model

![alt text][image7]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
