# SDC-P3 **Behavioral Cloning**   Author: Qitong Hu
---
## Files Submitted & Code Quality

### 1. Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.json & model.h5 containing a trained convolution neural network architecture and weight (end-to-end method)
* README.md (this file) summarizing the results
* transfer_vgg.py contains my trial code for using transfer learning with VGG
* transfer_model.py contains my trial code for using transfer learning with my own CNN

### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network (end-to-end method). The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

### 1. An appropriate model architecture has been employed
My model consists of a convolution neural network with architecture explaining in following section (model.py lines 49-80 function build_model() ). 

The model includes tanh layers to introduce nonlinearity (code line 66 & 74), and the data is normalized before inputing to the network (code line 37). 

### 2. Attempts to reduce overfitting in the model
The model contains dropout layers in order to reduce overfitting (model.py lines 65 & 73). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (in separate code). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 3. Model parameter tuning
The model used an adam optimizer, so the learning rate was not tuned manually. Other hyperparameter will explain in "training strategy" section.

### 4. Appropriate training data
Training data was chosen to keep the vehicle driving on the center of the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.
---
## Model Architecture and Training Strategy

### 1. Solution Design Approach
To do an end-to-end driver learning, the desired model should do a image recognition, extract high-level abstract of the visual input such as corner, curb etc. to make decision for steering just like what a human drive behave. Thus by intuition (and what I know as well as the NVidia paper [1] inspired), I am going to use a ConvNet structure to extract high-level features from images and connect it with a full-connected neural network to predict the desired action. Also tried to feed the model with more data after first-time training if it encounter problem with some corners through something similar to online learning. I tried to lower the learning rate [2] for extra data.  
Besides what I have presented here, I also tried to do transfer learning with VGG_16 [3]. I freezed from the Conv layers to the flatten layer. At the beginning I tried to use fine tune method to directly train the new full-connected layers but the model is too big to fit in a single g2.xlarge. Then I turned to feature extraction method and store the mediated output as a 25088 feature input to train the full-connected net. (in transfer_vgg.py, which is not a complete code).  
And I think since this is sort of a transcription task, which transcribe continuous images into steering command and those images and order-relevant, a RNN/GNN network could be more approriate. However, I don't have the know-how of constructing one, I think I will try it later. 

```sh
References:  
[1] My deep learning architecture is mainly follow the Nvidia paper (in model.py), http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf  
[2] And for later "online learning" with lower learning rate, https://keras.io/optimizers/ , https://arxiv.org/pdf/1412.6980v8.pdf for setting appropriate learning rates  
[3] VGG16, https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
```
My model is inspired mainly by NVidia paper, the differences need to notice here are:  
    a. the input image size is tuned according to the training data  
    b. Use maxpooling instead of conv kernel stride for those conv layers after the first one (notice the variable in build_model() still named kernel_sizes)  
    c. Add dropout layer for all the hidden layers with dropout value 0.5 to reduce overfit  
    d. Add one more conv layer to better fit the input dimensions  
    e. Use tanh as activation function  
    f. Use fit generator to generate data for training  

### 2. Final Model Architecture
(resize/input)3@160x80 -> (conv/maxpool)24@78x38 -> (conv/maxpool)36@37x17 -> (conv/maxpool)48@16x6 -> (conv/maxpool)64@14x4 -> (conv/maxpool)72@12x2 -> (conv)84@11x1 -> (flatten)924 -> (fc)100 -> (fc)50 -> (fc)10 -> (fc/output)1

Here is a visualization of the architecture 

![alt img1](https://raw.githubusercontent.com/qitong/SDC-P3/master/demonstration_images/other/model.png)

### 3. Creation of the Training Set & Training Process

#### Training data:  
training data is record from 50Hz Linux Simulator, with keyboard (so there maybe some zigzag), only center camera images are used for training  
generated 70k+ ordinary driving images and 30k+ swing back data (also some pullover-recovery data not used for training)    
test model on 10Hz Mac Simulator  

#### Validation data:  
validation data is randomly picked from training data, set aside 10% of total images generated as test set (around 10k)

#### Data preprocessing:  
In my own model, I resize image to 160x80 from 320x160, and do normalization for each image matrix to make value range in [-1,1]. Maybe YUV or grayscale can help in this situation, but since RGB give me positive output, I didn't switch to them.
For VGG transfer learning, I resize the image to 224x224 to fit original VGG input. And also switch the column 0 and 2 since it use dim_ordering theano rather than tensorflow.  

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example set of images of center lane driving:

![alt straight1](https://raw.githubusercontent.com/qitong/SDC-P3/master/demonstration_images/straight/center_2017_02_02_01_33_01_440.jpg)
![alt straight2](https://raw.githubusercontent.com/qitong/SDC-P3/master/demonstration_images/straight/center_2017_02_02_01_33_43_348.jpg)
![alt straight3](https://raw.githubusercontent.com/qitong/SDC-P3/master/demonstration_images/straight/center_2017_02_02_01_34_55_203.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover to drive in the center of the road These images show what a recovery looks:

![alt recovery1](https://raw.githubusercontent.com/qitong/SDC-P3/master/demonstration_images/recovery/center_2017_02_02_04_17_56_388.jpg)
![alt recovery2](https://raw.githubusercontent.com/qitong/SDC-P3/master/demonstration_images/recovery/center_2017_02_02_04_17_58_188.jpg)
![alt recovery3](https://raw.githubusercontent.com/qitong/SDC-P3/master/demonstration_images/recovery/center_2017_02_02_04_17_58_596.jpg)
![alt recovery4](https://raw.githubusercontent.com/qitong/SDC-P3/master/demonstration_images/recovery/center_2017_02_02_04_17_59_335.jpg)


### 4.Training  
The hyperparameters in the model such as architecture, kernel parameters is decided by intuition and input dimensions, batch size is decided by RAM.
Number of training epoch is figured out by loss on validation set, after somewhere around 30 it gives no significant gains, so I set it to 30.

## Others
Change to Drive.py:  
changed drive.py to have same resized and normalized image as training, and make the throttle higher to 0.3  

The result is on Youtube:  
https://youtu.be/3AdUvpxXzrw  
