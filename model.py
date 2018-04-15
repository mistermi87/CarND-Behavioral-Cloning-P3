# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 16:33:22 2018

@author: mstei
"""
import csv
import cv2
import numpy as np
import sklearn
from random import shuffle
from keras.layers import Flatten, Dense, Conv2D, Lambda, Activation, BatchNormalization, SeparableConvolution2D
from keras.models import Sequential
from keras.layers.convolutional import Cropping2D
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras import regularizers, initializers
from keras import optimizers


# Utilities----------------------------------


def illustrate_losses_in_training (history_object, model):
    
    ### print the keys contained in the history object
    print(history_object.history.keys())
    
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
  
    
    
def plot_train_data_histogram(train_samples):
    n_bins = 41
    sample_size=len(train_samples)
    raw_angles=[]
    aug_approx=[]
    
    for i in range(sample_size):
        angle = float(train_samples[i][3])
        raw_angles.append(angle)
        
        ## Aproximation of augmented steering angles
        aug_angle=angle+np.random.randint(-1,1)*0.25
        if np.random.uniform()>0.5:
            aug_angle=aug_angle*-1.0
        if np.random.uniform()>0.66:
            aug_angle=aug_angle+np.random.randint(-60, 60)*0.004
        aug_angle=np.clip(aug_angle,-1,1)
        aug_approx.append(aug_angle)
        
        
    plt.hist(raw_angles, n_bins, alpha=0.5, label='Original angles')
    plt.hist(aug_approx, n_bins, alpha=0.5, label='Approx. augm. angles for training')
    plt.legend(loc='upper right')
    plt.title("Distribution of Steering Angles")
    plt.savefig('steering_angles_hist.png', bbox_inches='tight')
    plt.show()
    
#---------------------------------------



def load_and_split_data():
    samples = []
    with open('./driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
        
        from sklearn.model_selection import train_test_split
        train_samples, validation_samples = train_test_split(samples, test_size=0.1)
        return train_samples, validation_samples


 
def add_random_shadow(image):
    #from: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
    top_x = 0
    bot_x = image.shape[0]
    width= image.shape[1]
    top_y = width*np.random.uniform()
    bot_y = width*np.random.uniform()
    #image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    rand_bright = 1-(np.random.uniform())*0.5
    cond1 = shadow_mask==1
    cond0 = shadow_mask==0
    if np.random.randint(2)==1:
        image[:,:,0][cond1] = image[:,:,0][cond1]*rand_bright
        image[:,:,1][cond1] = image[:,:,1][cond1]*rand_bright
        image[:,:,2][cond1] = image[:,:,2][cond1]*rand_bright
    else:
        image[:,:,0][cond0] = image[:,:,0][cond0]*rand_bright
        image[:,:,1][cond0] = image[:,:,1][cond0]*rand_bright
        image[:,:,2][cond0] = image[:,:,2][cond0]*rand_bright
                 
    return image



def augument_image(image, angle):
# every second image gets flipped horizontally
    if np.random.uniform()>0.5:
        image=cv2.flip(image,1)
        angle=angle*-1.0
  
    if np.random.uniform()>0.66:
# 1 in 3 images is randomly translated, partly from: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
        translation_x = np.random.randint(-60, 60)
        translation_y = np.random.randint(-20, 20)
        angle += translation_x * 0.004 #for every Pixel tranlated, the steering angle is corrected by 0.4%
        translation_matrix = np.float32([[1, 0, translation_x],[0, 1, translation_y]])
        image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    
    if np.random.uniform()>0.66:
        # 1 in 3 images gets a brightness reduction of up to 50%
        # I could have transformed into HLV-Color-space and just adjusted one channel but this gave weird result in the Traffic Sign assignment
        rand_bright = 1-(np.random.uniform())*0.5
        image[:,:,0] = image[:,:,0]*rand_bright
        image[:,:,1] = image[:,:,1]*rand_bright
        image[:,:,2] = image[:,:,2]*rand_bright
        
        image[image>255] = 255
        
    if np.random.uniform()>0.75:
        # 1 in 4 images gets an augumented shadow from: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
        image=add_random_shadow(image)
    
    return image, angle



def generator(samples, batch_size=128, augmentation=0):
    num_samples = len(samples)
    #angle correction for each column in img_data
    angles_in_column=(0, 0.25, -0.25)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            # Following part is called during validation or if no augmentation is wanted
            if augmentation<=1:
                for batch_sample in batch_samples:
                    center_image = cv2.imread(batch_sample[0])
                    center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                    center_angle = float(batch_sample[3])
                    images.append(center_image)
                    angles.append(center_angle)
                
        
            # Following part is called during training with augmentation
            if augmentation>1:
                for batch_sample in batch_samples:
                    #choose randomly if left right or center image from bumper camera
                    which_image=np.random.randint(0,2)
                    image=cv2.imread(batch_sample[which_image])
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #colorConversion
                    angle=float(batch_sample[3])+float(angles_in_column[which_image])
                    
                    (image, angle)=augument_image(image, angle)
                    angle=np.clip(angle,-1,1) #No steering angle above one
                    
                    images.append(image)
                    angles.append(angle)                  
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)



def create_model():
    model = Sequential()
    # Preprocess incoming data
    img_crop_top=50
    img_crop_bot=20
    
    ch, row, col =3 , 160, 320  #image format
    
    model.add(Cropping2D(cropping=((img_crop_top,img_crop_bot), (0,0)), input_shape=(row, col,ch)))

    row_cr=row-img_crop_top-img_crop_bot
    
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(row_cr, col,ch),output_shape=(row_cr, col,ch)))
    
    initial=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None) # all weights are initialized by a normal distribution, centered around zero with small standard deviation
    ker_reg=regularizers.l2(0.005) # Kernel regularizer
    # no Activation regularizer as Batch norm takes care of that
    
    # =============================================================================
    model.add(Conv2D(8, (3, 3), strides=(2, 2), padding='same',use_bias=0, kernel_regularizer=ker_reg, kernel_initializer=initial))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
     
    model.add(SeparableConvolution2D(16, (3, 3), strides=(1, 1), padding='same', use_bias=0, pointwise_regularizer=ker_reg, pointwise_initializer=initial))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
     
    model.add(SeparableConvolution2D(16, (3, 3), strides=(2, 2), padding='same', use_bias=0, pointwise_regularizer=ker_reg, pointwise_initializer=initial))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
     
    model.add(SeparableConvolution2D(32, (3, 3), strides=(1, 1), padding='same', use_bias=0, pointwise_regularizer=ker_reg, pointwise_initializer=initial))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
     
    model.add(SeparableConvolution2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=0, pointwise_regularizer=ker_reg, pointwise_initializer=initial))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    
    model.add(SeparableConvolution2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=0, pointwise_regularizer=ker_reg, pointwise_initializer=initial))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
        
    model.add(SeparableConvolution2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=0, pointwise_regularizer=ker_reg,pointwise_initializer=initial))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
  
    model.add(SeparableConvolution2D(64, (3, 3), strides=(2, 2), padding='same', use_bias=0, pointwise_regularizer=ker_reg, pointwise_initializer=initial))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    
    model.add(SeparableConvolution2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=0, pointwise_regularizer=ker_reg, pointwise_initializer=initial))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
   
    model.add(SeparableConvolution2D(128, (3, 3), strides=(2, 2), padding='same', use_bias=0, pointwise_regularizer=ker_reg, pointwise_initializer=initial))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    
    model.add(SeparableConvolution2D(128, (3, 3), strides=(1, 1), padding='valid', use_bias=0, pointwise_regularizer=ker_reg, pointwise_initializer=initial))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    
    model.add(Flatten())
    
    model.add(Dense(100, activation='elu',kernel_regularizer=ker_reg,kernel_initializer=initial))
    model.add(Dense(50,  activation='elu',kernel_regularizer=ker_reg,kernel_initializer=initial))
    model.add(Dense(25,  activation='elu',kernel_regularizer=ker_reg,kernel_initializer=initial))
    model.add(Dense(10,  activation='elu',kernel_initializer=initial)) #,kernel_regularizer=regularizers.l2(ker_reg)))
    model.add(Dense(1))
    
    return model



def main():
    
    aug_factor=8 #How many Images are used (some of them augumented) from one recorded timestep
    batch_s=128 #Batch Size
    
    #Load data and plit into training and validation
    (train_samples, validation_samples) = load_and_split_data()
    
    #create model
    model=create_model()
    
    #define generators
    train_generator = generator(train_samples, batch_size=batch_s, augmentation=aug_factor)
    validation_generator = generator(validation_samples, batch_size=batch_s, augmentation=0)
    
    #define optimizer with non-default values
    optimizers.Adam(lr=0.0001, amsgrad=True)
    
    #define model-checkpoint where the model is saved, if the val_los is lower than in the previous epoch
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',  monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
    
    #compile model
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    
    # Print model data with widend output for all layer names
    print(model.summary(line_length=100,positions=[.50, .8, .95, 1.]))

    #fit model
    history_object=model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_s*aug_factor, validation_data=validation_generator, validation_steps=len(validation_samples)/32, epochs=20, verbose=1, callbacks=[checkpoint])
    
    #save model
    model.save('model.h5')
    
    #create graph of training history (from udacity)
    illustrate_losses_in_training(history_object, model)
    
    return 0

if __name__ == '__main__':
    main()

