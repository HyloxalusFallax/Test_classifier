from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from styleaug import StyleAugmentor
from random import seed
from random import random

import torch
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import pandas as pd

seed(1)
augmentor = StyleAugmentor()
toTensor = ToTensor()

def style_aug(inputs):
    outputs = inputs.copy()
    if random() < 0.30:
        im_torch = toTensor(outputs).unsqueeze(0)
        im_torch = im_torch.to('cpu')

        # randomize style:
        im_restyled = augmentor(im_torch)
        outputs = im_restyled.squeeze().cpu()
        outputs = im_restyled[0].cpu().detach().numpy()
        outputs = np.transpose(outputs, (1, 2, 0))

        outputs = (outputs * 255)
    return outputs

opt = optimizers.Adam(
    learning_rate=0.001, beta_1=0.5, beta_2=0.999, epsilon=1e-03, amsgrad=False,
    name='Adam'
)

augmentor = StyleAugmentor()

img_width = 100
img_height = 100

#load inceptv3 model with random weights and without top layer
old_model = InceptionV3(include_top=False, weights=None, input_shape = (img_width, img_height, 3))

#add our new layers
x = layers.Flatten()(old_model.output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(31, activation='softmax')(x)

#add models together
model = Model(old_model.input, x)

#print model structure
#print(model.summary())

#compile resulting model
model.compile(optimizer = opt, loss='categorical_crossentropy', metrics=['accuracy'])

#make data generators for three parts of our dataset

train_datagen = ImageDataGenerator(rescale=1./255.)

#train_datagen = ImageDataGenerator(rescale=1./255., preprocessing_function=style_aug)

# train_datagen = ImageDataGenerator(rescale=1./255., preprocessing_function=style_aug,
    # shear_range=0.2,
    # rotation_range=15,
    # brightness_range=[0.75,1.25],
    # zoom_range=[0.75,1.25],
    # horizontal_flip=True,
    # width_shift_range=.15,
    # height_shift_range=.15)

test_datagen = ImageDataGenerator(rescale=1./255.)

train_it = train_datagen.flow_from_directory('./img/train/',
    class_mode='categorical',
    batch_size=16,
    target_size=(100, 100),
    shuffle=True,
    color_mode="rgb")
    
# test_it = test_datagen.flow_from_directory('./img/test/',
    # class_mode='categorical',
    # batch_size=16,
    # target_size=(100, 100),
    # shuffle=True,
    # color_mode="rgb")

val_it = test_datagen.flow_from_directory('./img/test/',
    class_mode='categorical', 
    batch_size=16, 
    target_size=(100, 100), 
    shuffle=True, 
    color_mode="rgb")

#train
history = model.fit(
    train_it,
    epochs = 250,
    validation_data = val_it,
    verbose=2
)

#save model
model.save_weights('./models/dataset_shift_classifier_1.h5')
pd.DataFrame.from_dict(history.history).to_csv('history_dataset_shift.csv',index=False)

#use after model is trained
model.load_weights('./models/dataset_shift_classifier_1.h5')
print(model.evaluate(val_it))
    