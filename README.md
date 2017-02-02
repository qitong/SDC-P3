# SDC-P3

The deep learning architecture is mainly follow the Nvidia paper (train_model.py):
(resize/input)3@160x80 -> (conv/maxpool)24@78x38 -> (conv/maxpool)36@37x17 -> (conv/maxpool)48@16x6 -> (conv/maxpool)64@14x4 -> (conv/maxpool)72@12x2 -> (conv)84@11x1 -> (flatten)924 -> (fc)100 -> (fc)50 -> (fc)10 -> (fc/output)1

training data is record from 50Hz Linux Simulator, 
test model on 10Hz Mac Simulator

changed drive.py to have same resized image as training

the current result is on Youtube:
https://youtu.be/_p_6cAVHrHk
