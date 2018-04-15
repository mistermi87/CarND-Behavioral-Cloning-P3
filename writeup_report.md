# **Behavioral Cloning** 

## Project Writeup

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./resources/steering_angles_hist.png "Distribution of steering angles for training"
[image3]: ./resources/graph_of_losses.png "Training and Validation loss for each Epoch of Training "
[image2]: ./resources/aug_img.jpg "1/8 of a batch of augmented images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


___


## Model Architecture 

### Design Idea 

One big motivation during this project (as in the previous project of traffic sign classification) was to build a robust model, that is also small and fast. Of course one can be successful in completing all tracks with a model of 5 million trained variables, even if the whole model is ineffective or poorly designed. The reviewer of my previous model pointed me to the [video](https://www.microsoft.com/en-us/research/video/small-deep-neural-networks-advantages-design/) of a talk  in which Forrest Iandola of DeepScale describes new (Nov '17) architectures for lightweight neural networks. He also presents basic design principles of successful small neural networks:
1.	Replace Fully-Connected Layers with Convolutions
2.	Kernel Reduction of the Convolutions
3.	Channel Reduction
4.	Evenly spaced Downsampling
5.	Depthwise Seperable Convolutions
6.	Shuffle Operations
7.	Distillation and Compression

One of the models described was [MobileNet](https://arxiv.org/abs/1704.04861) by Google. I already tried to implement it in the Traffic Sign Classifier unsuccessfully. It covers numbers 1,2,3,4 and 5 of the list above. I opted to use this basic architecture as a starting point for further research.

###  Implementation and alterations

A simple, well documented  (and fun) implementation of this model was named "DeepDog" and used in the "not-HotDog" App of the developers of the HBO-Series "Silicon Valley" ([Medium Link](https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3))

Some of their remarks on the implementation, from the article:
>We do not use Batch Normalization & Activation between depthwise and pointwise convolutions, because the XCeption paper (which discussed depthwise convolutions in detail) seemed to indicate it would actually lead to less accuracy in architecture of this type (as  [helpfully pointed out](https://www.reddit.com/r/MachineLearning/comments/663m43/r_170404861_mobilenets_efficient_convolutional/dgfaoz1/) 

>We use ELU instead of ReLU. Just like with our SqueezeNet experiments, it provided superior convergence speed & final accuracy when compared to ReLU

>We maintain the use of Batch Normalization with ELU. There are many indications that this should be unnecessary, however, every experiment we ran without Batch Normalization completely failed to converge. This could be due to the small size of our architecture.

>   We used Batch Normalization  _before_ the activation. While this is a  [subject of some debate](https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/)  these days, our experiments placing BN after activation on small networks failed to converge as well.

I addopted the main tweaks to MobileNet from there but added a few more things:
- I added more fully connected layers, because this is not a classification task but a task were **smooth** weighting of different features is necessary for a non discrete output.
- I reduced the number if channels in each convolution layer (~tuning $\alpha=0.25$ in the MobileNet paper)
- there is no activation between depthwise and pointwise convolution steps in the depthwise seperable convolution (unlike in the original Mobilenet). To my understanding, this means that the biases of the two steps would could be added up in this case. Since there is a batch normalization applied afterwards, those biases can be neglected (setting `use_bias=0`)  as the normalization applies bias-like parameters . Unfortunately those values are not calculated during regular back-propagation but Keras takes care of this.
- I introduced **L2-regularization** on the pointwise convolution steps as suggested in the original MobileNet paper. This was the only hyperparameter to tune (outside of the model architecture) to fight over-fitting. I only used regularization on the pointwise kernels of the `SeparableConvolution2D` layers (as suggested by the MobileNets paper) and the `Dense` layers. I did not use "activity_regularizer" because batch normalization should take care of the output of the convolutional layers.
- I explicitly **initialized the kernels** of the pointwise convolutions and the fully connected layers. The kernels of the depthwise convolutions, each with 9 parameters applied basically on every pixel of every input image, did not need to be initialized explicit.
- Instead of the [Cyclical Learning Rates](https://arxiv.org/abs/1506.01186) used in "DeepDog", I opted for using the traditional **ADAM-Solver**. In my understanding the benefits of CLRs only become evident with a larger number of Epochs (unlike the <20 I am using). Researching about this, I stumbled over the current discussion whether the ADAM-Solver has a error in its [Weight Decay Regularization](https://openreview.net/forum?id=rk6qdGgCZ) ([Also](https://openreview.net/forum?id=ryQu7f-RZ)). The correction for this is called AMSgrad . It seems that, while it does not improve the final accuracy of the model by a lot, it lets the model converge faster ([Reference](https://fdlm.github.io/post/amsgrad/)), which is crucial letting it train on a 4-year old Laptop-GPU. That is why I activated it by setting `optimizers.Adam([...],amsgrad=True)`. I believe that the results are a bit better now, but this might be anecdotal. 

### Detailed Architecture
The input images are normalized and cropped in the model. For this I used a `Cropping2D` layer followed by a `Lambda`layer.  Those steps were copied from the original Udacity code, given as a starting point.
The first network-layer is a regular `Conv2D` layer with a filter of 3x3, a stride of [2,2], no padding, and a output depth of 8. From there on the input is funnelled through a cascade of (mostly) pairs of blocks of `SeparableConvolution2D` layers, each with a depthwise filter of 3x3 and a stride of [1,1] and then [2,2] (except for the last one with padding='SAME'). Each Convolutional layer is followed by `Batch_normlization` and and  an `elu` activation. 
I later added an additional block at a depth of 64 channels with a stride of [1,1], identical to the previous one. I saw improvement in validation accuracy doing that.
After the last Convolutional layer the vertical image dimension becomes 1 and the remaining data is flattened and connected to the first of the five fully-connected `Dense` layers. In the the first three `Dense` layers L2-regularization is applied to the same extent as to the pointwise regularization. The sizes of the layers are 100, 50, 25, 10, 1. No batch normalization is applied here.
The single node of the last `Dense` layer is not activated. Instead it's output is directly used for the Mean-squared-error calculation of the loss-function and later for the suggested steering angle.

| Layer (type) |   Output Shape | Param # |Comment| 
| ----------------------- | ---------------------------- | :------: |-----|
| cropping2d_1 (Cropping2D) |  (None, 90, 320, 3) | 0 ||
| lambda_1 (Lambda) |  (None, 90, 320, 3) | 0 ||
| conv2d_1 (Conv2D) |  (None, 45, 160, 8) | 216 | k:[3,3],s:[2,2],p:'same'|
| batch_normalization_1 (BatchNormalization) |  (None, 45, 160, 8) | 32 ||
| activation_1 (Activation) |  (None, 45, 160, 8) | 0 |'elu'|
| separable_conv2d_1 (SeparableConv2D) |  (None, 45, 160, 16) | 200 |k:[3,3],s:[1,1],p:'same'|
| batch_normalization_2 (BatchNormalization) |  (None, 45, 160, 16) | 64 ||
| activation_2 (Activation) |  (None, 45, 160, 16) | 0 |'elu'|
| separable_conv2d_2 (SeparableConv2D) |  (None, 23, 80, 16) | 400 | k:[3,3],s:[2,2],p:'same'|
| batch_normalization_3 (BatchNormalization) |  (None, 23, 80, 16) | 64 ||
| activation_3 (Activation) |  (None, 23, 80, 16) | 0 |'elu'|
| separable_conv2d_3 (SeparableConv2D) |  (None, 23, 80, 32) | 656 |k:[3,3],s:[1,1],p:'same'|
| batch_normalization_4 (BatchNormalization) |  (None, 23, 80, 32) | 128 ||
| activation_4 (Activation) |  (None, 23, 80, 32) | 0 |'elu'|
| separable_conv2d_4 (SeparableConv2D) |  (None, 12, 40, 32) | 1312 | k:[3,3],s:[2,2],p:'same'|
| batch_normalization_5 (BatchNormalization) |  (None, 12, 40, 32) | 128 ||
| activation_5 (Activation) |  (None, 12, 40, 32) | 0 |'elu'|
| separable_conv2d_5 (SeparableConv2D) |  (None, 12, 40, 64) | 2336 |k:[3,3],s:[1,1],p:'same'|
| batch_normalization_6 (BatchNormalization) |  (None, 12, 40, 64) | 256 ||
| activation_6 (Activation) |  (None, 12, 40, 64) | 0 |'elu'|
| separable_conv2d_6 (SeparableConv2D) |  (None, 12, 40, 64) | 4672 |k:[3,3],s:[1,1],p:'same'|
| batch_normalization_7 (BatchNormalization) |  (None, 12, 40, 64) | 256 ||
| activation_7 (Activation) |  (None, 12, 40, 64) | 0 |'elu'|
| separable_conv2d_7 (SeparableConv2D) |  (None, 6, 20, 64) | 4672 | k:[3,3],s:[2,2],p:'same'|
| batch_normalization_8 (BatchNormalization) |  (None, 6, 20, 64) | 256 ||
| activation_8 (Activation) |  (None, 6, 20, 64) | 0 |'elu'|
| separable_conv2d_8 (SeparableConv2D) |  (None, 6, 20, 128) | 8768 |k:[3,3],s:[1,1],p:'same'|
| batch_normalization_9 (BatchNormalization) |  (None, 6, 20, 128) | 512 ||
| activation_9 (Activation) |  (None, 6, 20, 128) | 0 |'elu'|
| separable_conv2d_9 (SeparableConv2D) |  (None, 3, 10, 128) | 17536 | k:[3,3],s:[2,2],p:'same'|
| batch_normalization_10 (BatchNormalization) |  (None, 3, 10, 128) | 512 ||
| activation_10 (Activation) |  (None, 3, 10, 128) | 0 |'elu'|
| separable_conv2d_10 (SeparableConv2D) |  (None, 1, 8, 128) | 17536 |k:[3,3],s:[1,1],p:'valid'|
| batch_normalization_11 (BatchNormalization) |  (None, 1, 8, 128) | 512 ||
| activation_11 (Activation) |  (None, 1, 8, 128) | 0 |'elu'|
| flatten_1 (Flatten) |  (None, 1024) | 0 ||
| dense_1 (Dense) |  (None, 100) | 102500 |activation='elu'|
| dense_2 (Dense) |  (None, 50) | 5050 |activation='elu'||
| dense_3 (Dense) |  (None, 25) | 1275 |activation='elu'||
| dense_4 (Dense) |  (None, 10) | 260 |activation='elu'||
| dense_5 (Dense) |  (None, 1) | 11 |activation='elu'||

>Total params: 170,120
Trainable params: 168,760
Non-trainable params: 1,360

## Model parameter tuning

### Hyperparameters tuned:

- **ADAM-Solver**: 
	- Default: beta_1, beta_2, epsilon, decay
	- learning rate = **1e-4**: in the forum, fellow students suggested that lr=1e-4 was superior to the default value of 1e-3 for this task. I tested it and saw small benefits
	- amsgrad= **True**: As mentioned above, this option lets the model converge faster
	
- **L2-regularization**: **5e-3** I started with a learning rate of 1e-3 and [this paper](https://arxiv.org/abs/1706.05350) suggested a L2-regularization of the weights of equally 1e-3. After I changed the learning rate to 1e-4, the paper suggested a regularization factor of 1e-1 which was too big for such a deep and complex model. Following the paper, I still opted for a stronger regularization with smaller learning rate (~inverse dependency)

- **Batch size**:  **128** I wanted to use the largest batch my GPU-memory would allow and not change it later, so that the regularizing effect of the batch normalization (which depends on the batch size) would not interfere with the tuning of the other hyperparameters. 
- **Augmentation factor**: **8** This number was introduced to quantify the number of images generated and altered from a single image in order to have more training data in one Epoch. If set to 0 or 1, no data augmentation is applied (--> Validation).

### Architectural parameters tuned:
- I added more `SeparableConvolution2D` layers when I assumed that the model was underfitting. This was a computationally cheap way to make the model deeper and thus more complex without adding too many trainable parameters.
- In order to reduce the  number of trainable parameters I successively reduced the size of the `Dense` layers. 

## Training data and augmentation

### Recorded data
Some fellow students tried to create a model that was successfully finishing a course with as few training examples as possible. This was not my approach. I want to have a small model with perfectly trained weights. Regularization (especially Dropout) is based on creating redundancies in the weights of the layers to make the model more robust. In order to have a small model I tried to keep the usage of Regularization to a minimum and avoid overfitting by using a lot of training data.

First I recorded following driving data for each track. In both tracks I recorded two laps driving counterclockwise and two laps driving clockwise. Additionally I recorded  "recovering" scenarios, where I started recording when I was driving towards the side of the road, just to recover back to the center.
In the Circuit Course the first models struggled when leaving the bridge so I added some training data there. In the Jungle Course I recorded some additional sharp corners and the last meters of the finish line as my model aimed for the opposing track in that situation.

### Augmentation

All recordings were done using 'fastest' graphics setting, so that no real shadows appear in the training images, I only discovered that later, when I let the model drive itself on a shaded track.
To this data I added the training data given by Udacity.
This results in a total of ~20,000 data-points each with 3 images. Removing 10% for validation, and having an augmentation factor of 8, the model is trained on ~ 14400 different images each epoch. 

![Distribution of steering angles for training ][image1]


As just mentioned I applied image augmentation to boost the amount of training data. My focus on this project was not 
In the beginning I used a generator to create the batches that were fed into the model. Working with a batch size of 128 and an augmentation factor of 8 the generator creates a virtual batch size of $128/8=32$.  32 Data-points are taken from the shuffled Training set. This slowed down the trainign by a factor of approx. five, as only alterations of 32 different images were used in every batch and altered in eight different ways, making Batch Normalization and Adam-Solver extremly ineffective.
When I discovered this mistake I just let the generator alter all images randomly and let the model run 8x through the data for one epoch. This solution works but is not especially pretty.

The augmentation pipeline for one image:
- one of the three images taken at the bumper of the car (left, right, center) is selected randomly. Color Space is adjusted (BGR2RGB). For images that are not from the center, the steering angle is adjusted by +/-0.25 (a value proven by other students to be the most solid in this scenario).
```python
angles_in_column=(0, 0.25, -0.25)
which_image=np.random.randint(0,2)
image=cv2.imread(batch_sample[which_image])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
angle=float(batch_sample[3])+float(angles_in_column[which_image])
```
- half of those images are flipped (incl. the steering angle)
```python
if np.random.uniform()>0.5:
        image=cv2.flip(image,1)
        angle=angle*-1.0
```
- 1/3 of them are then is randomly translated and the steering angle is corrected by 0.004 for every pixel of horizontal translation (This is based on the model from [here](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9), but a bit simplified)
```python 
if np.random.uniform()>0.66:
# 1 in 3 images 
        translation_x = np.random.randint(-60, 60)
        translation_y = np.random.randint(-20, 20)
        angle += translation_x * 0.004 translation_matrix = np.float32([[1, 0, translation_x],[0, 1, translation_y]])
        image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
```
- the brightness of 1/3 of the same images  is then randomly changed by up to to -50%. It would have been simpler to transform the image into the HLV-Color Space first, but but this gave weird aliasing effects in the Traffic Sign assignment. Since I only used images from brightly lit tracks I opted for only darkening the images in order to generalize the model to drive on shaded roads.
```python 
if np.random.uniform()>0.66:
        rand_bright = 1-(np.random.uniform())*0.5
        image[:,:,0] = image[:,:,0]*rand_bright
        image[:,:,1] = image[:,:,1]*rand_bright
        image[:,:,2] = image[:,:,2]*rand_bright
        
```

When I later saw that my model could not handle shadows at all I also applied [this former students](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9) shadow augmentation on 1/4 of the images. This improved the model even though it still struggles on the shaded jungle tracks, being trained only on 'fake' shadows.

Choosing random bumper camera angles, flipping and translating the images led to a much more even distribution of the steering angles of the training data. I chose above values to get as close as possible to a normal distribution. Nevertheless, peaks at 0,-0.25 and +0.25 were unavoidable.

![1/8 of a batch of augmented images][image2]


## Conclusions from attempts to  improve in the model
After I had my rough architecture running I tweaked these values to fight overfitting:
- More data --> Augmentation (the right way)
- higher L2-Normailzation factor

At some point I felt my model was underfitting so I:
- added layers 
	- -convolution, batch norm, activation- Block at a channel depth of 64
	- Fully connected layer with 25 Nodes, between the 50-Node layer and the 10-Node layer
- did NOT lower regularization as I felt it made the model unstable (fluctuating loss function after every epoch)

In the beginning I only judged a model's quality by its low validation loss. After testing several iterations "on the road' I realized that this is only a mathematical benchmark that does not necessarily correlate with driving behavior. A much better indicator is the validation accuracy.

## Results
The training of the final model:
````
Epoch 1/20
1124/1123 [==============================] - 731s 650ms/step - loss: 0.1165 - acc: 0.0936 - val_loss: 0.0354 - val_acc: 0.2681
Epoch 2/20
1124/1123 [==============================] - 698s 621ms/step - loss: 0.0394 - acc: 0.0976 - val_loss: 0.0335 - val_acc: 0.2631
Epoch 3/20
1124/1123 [==============================] - 709s 631ms/step - loss: 0.0360 - acc: 0.0956 - val_loss: 0.0311 - val_acc: 0.2693
Epoch 4/20
1124/1123 [==============================] - 700s 623ms/step - loss: 0.0343 - acc: 0.0967 - val_loss: 0.0320 - val_acc: 0.2692
Epoch 5/20
1124/1123 [==============================] - 713s 634ms/step - loss: 0.0328 - acc: 0.0979 - val_loss: 0.0293 - val_acc: 0.2697
Epoch 6/20
1124/1123 [==============================] - 713s 634ms/step - loss: 0.0319 - acc: 0.0981 - val_loss: 0.0281 - val_acc: 0.2704
Epoch 7/20
1124/1123 [==============================] - 712s 633ms/step - loss: 0.0318 - acc: 0.0986 - val_loss: 0.0260 - val_acc: 0.2706
Epoch 8/20
1124/1123 [==============================] - 718s 638ms/step - loss: 0.0310 - acc: 0.0979 - val_loss: 0.0277 - val_acc: 0.2714
Epoch 9/20
1124/1123 [==============================] - 717s 637ms/step - loss: 0.0303 - acc: 0.0978 - val_loss: 0.0250 - val_acc: 0.2712
Epoch 10/20
1124/1123 [==============================] - 713s 634ms/step - loss: 0.0305 - acc: 0.0983 - val_loss: 0.0283 - val_acc: 0.2672
Epoch 11/20
1124/1123 [==============================] - 707s 629ms/step - loss: 0.0301 - acc: 0.0980 - val_loss: 0.0280 - val_acc: 0.2706
Epoch 12/20
1124/1123 [==============================] - 723s 643ms/step - loss: 0.0297 - acc: 0.0992 - val_loss: 0.0253 - val_acc: 0.2720
Epoch 13/20
1124/1123 [==============================] - 708s 630ms/step - loss: 0.0291 - acc: 0.0975 - val_loss: 0.0256 - val_acc: 0.2698
Epoch 14/20
1124/1123 [==============================] - 685s 609ms/step - loss: 0.0293 - acc: 0.0984 - val_loss: 0.0263 - val_acc: 0.2693
Epoch 15/20
1124/1123 [==============================] - 689s 613ms/step - loss: 0.0287 - acc: 0.0980 - val_loss: 0.0258 - val_acc: 0.2710
Epoch 16/20
1124/1123 [==============================] - 691s 615ms/step - loss: 0.0288 - acc: 0.0985 - val_loss: 0.0272 - val_acc: 0.2715
Epoch 17/20
1124/1123 [==============================] - 680s 605ms/step - loss: 0.0288 - acc: 0.0983 - val_loss: 0.0257 - val_acc: 0.2732
Epoch 18/20
1124/1123 [==============================] - 695s 618ms/step - loss: 0.0286 - acc: 0.0986 - val_loss: 0.0248 - val_acc: 0.2716
Epoch 19/20
1124/1123 [==============================] - 684s 609ms/step - loss: 0.0284 - acc: 0.0987 - val_loss: 0.0250 - val_acc: 0.2728
Epoch 20/20
1124/1123 [==============================] - 689s 613ms/step - loss: 0.0282 - acc: 0.0971 - val_loss: 0.0262 - val_acc: 0.2705
````

![Training and Validation loss for each Epoch of Training ][image3]

This was the last iteration of this model. While I had other models that were shallower and performed smoother when driving on graphics mode= "fastest", this one was able to navigate through the shadows on the track on higher qualities, even though it was never trained on data recorded with shadows. All models failed when driving through the shaded area of the jungle with graphics mode on good or better. Most likely it was not possible to identify the end of the pavement in the darkness. Augmenting heavily darkened images were to no avail. 

## Youtube-Videos of completing both tracks 
#### First course (Circuit), no shadows, full-speed, SUCCESSFUL:
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/GwgeX6GCPQE/0.jpg)](https://www.youtube.com/watch?v=GwgeX6GCPQE)

#### Second Course (Jungle), no shadows, full-speed, SUCCESSFUL:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/ACvSl39WOCE/0.jpg)](https://www.youtube.com/watch?v=ACvSl39WOCE)

#### First Course (Circuit), with shadows, full-speed, SUCCESSFUL (touching line once):

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/0Qf2FnrbTlU/0.jpg)](https://www.youtube.com/watch?v=0Qf2FnrbTlU)

