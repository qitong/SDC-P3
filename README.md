# SDC-P3

MODEL:  

1. Model Design  
To do an end-to-end driver learning, the desired model should do a image recognition, extract high-level abstract of the visual input such as corner, curb etc. to make decision for steering just like what a human drive behave. Thus by intuition (and what I know as well as the NVidia paper [1] inspired), I am going to use a ConvNet structure to extract high-level features from images and connect it with a full-connected neural network to predict the desired action. Also tried to feed the model with more data after first-time training if it encounter problem with some corners through something similar to online learning. I tried to lower the learning rate [2] for extra data.
Besides what I have presented here, I also tried to do transfer learning with VGG_16 [3]. I freezed from the Conv layers to the flatten layer. At the beginning I tried to use fine tune method to directly train the new full-connected layers but the model is too big to fit in a single g2.xlarge. Then I turned to feature extraction method and store the mediated output as a 25088 feature input to train the full-connected net. (in transfer_vgg.py, which is not a complete code).
And I think since this is sort of a transcription task, which transcribe continuous images into steering command and those images and order-relevant, a RNN/GNN network could be more approriate. However, I don't have the know-how of constructing one, I think I will try it later. 


References:
[1] My deep learning architecture is mainly follow the Nvidia paper (in model.py), http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf  
[2] And for later "online learning" with lower learning rate, https://keras.io/optimizers/ , https://arxiv.org/pdf/1412.6980v8.pdf for setting appropriate learning rates  
[3] VGG16, https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

2. Architecture  
(resize/input)3@160x80 -> (conv/maxpool)24@78x38 -> (conv/maxpool)36@37x17 -> (conv/maxpool)48@16x6 -> (conv/maxpool)64@14x4 -> (conv/maxpool)72@12x2 -> (conv)84@11x1 -> (flatten)924 -> (fc)100 -> (fc)50 -> (fc)10 -> (fc/output)1

the difference from original NVidia paper need to notice here are:  
    a. the input image size is tuned according to the training data  
    b. Use maxpooling instead of conv kernel stride for those conv layers after the first one (notice the variable in build_model() still named kernel_sizes)  
    c. Add dropout layer for all the hidden layers with dropout value 0.5 to reduce overfit  
    d. Add one more conv layer to better fit the input dimensions  
    e. Use tanh as activation function  
    f. Use fit generator to generate data for training  

3. Data  
Training data:  
training data is record from 50Hz Linux Simulator, with keyboard (so there maybe some zigzag), only center camera images are used for training  
generated 70k+ ordinary driving images and 30k+ swing back data (also some pullover-recovery data not used for training)    
test model on 10Hz Mac Simulator  

Validation data:  
validation data is randomly picked from training data, set aside 10% of total images generated as test set (around 10k)

Data preprocessing:  
In my own model, I resize image to 160x80 from 320x160, and do normalization for each image matrix to make value range in [-1,1]. Maybe YUV or grayscale can help in this situation, but since RGB give me positive output, I didn't switch to them.
For VGG transfer learning, I resize the image to 224x224 to fit original VGG input. And also switch the column 0 and 2 since it use dim_ordering theano rather than tensorflow.  

4. Training  
The hyperparameters in the model such as architecture, kernel parameters is decided by intuition and input dimensions, batch size is decided by RAM.
Number of training epoch is figured out by loss on validation set, after somewhere around 30 it gives no significant gains, so I set it to 30.


Change to Drive.py:  
changed drive.py to have same resized and normalized image as training, and make the throttle higher to 0.3  


The result is on Youtube:  
https://youtu.be/3AdUvpxXzrw  

* There are still some cases the car probably drive onto the curb found after uploading.
