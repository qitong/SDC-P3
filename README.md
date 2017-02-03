# SDC-P3

MODEL:
My deep learning architecture is mainly follow the Nvidia paper (train_model.py):

(resize/input)3@160x80 -> (conv/maxpool)24@78x38 -> (conv/maxpool)36@37x17 -> (conv/maxpool)48@16x6 -> (conv/maxpool)64@14x4 -> (conv/maxpool)72@12x2 -> (conv)84@11x1 -> (flatten)924 -> (fc)100 -> (fc)50 -> (fc)10 -> (fc/output)1

the changes need to notice here are:
1. the input image size is tuned according to the training data
2. Use maxpooling instead of conv kernel stride for those conv layers after the first one (notice the variable in build_model() still named kernel_sizes)
3. Add dropout layer for all the hidden layers with dropout value 0.5 to reduce overfit
4. Add one more conv layer to better fit the input dimensions
5. Use tanh as activation function
6. Use fit generator to generate data for training

Training data:
training data is record from 50Hz Linux Simulator, with keyboard (so there maybe some zigzag), only center camera images are used for training
generated 70k+ ordinary driving images and 30k+ swing back data (also some pullover-recovery data not used for training)  
test model on 10Hz Mac Simulator


Change to Drive.py:
changed drive.py to have same resized and normalized image as training, and make the throttle higher to 0.3


The result is on Youtube:
https://youtu.be/3AdUvpxXzrw

* There are still some cases the car probably drive onto the curb found after uploading.
