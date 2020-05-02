from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model

img_width = 96
img_height = 96

#load inceptv3 model with random weights and without top layer
old_model = InceptionV3(include_top=False, weights=None, input_shape = (img_width, img_height, 3))

#add our new layers
x = layers.Flatten()(old_model.output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(10, activation='softmax')(x)

#add models together
model = Model(old_model.input, x)

#print model structure
print(model.summary())

#compile resulting model
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

#make data generators for three parts of our dataset
datagen = ImageDataGenerator(rescale=1./255.)
train_it = datagen.flow_from_directory('./img/train/', class_mode='categorical', batch_size=16, target_size=(96, 96), shuffle=True, color_mode="rgb")
test_it = datagen.flow_from_directory('./img/test/', class_mode='categorical', batch_size=16, target_size=(96, 96), shuffle=True, color_mode="rgb")
val_it = datagen.flow_from_directory('./img//validation/', class_mode='categorical', batch_size=16, target_size=(96, 96), shuffle=True, color_mode="rgb")

#train
history = model.fit(
    train_it,
    epochs = 150,
    validation_data = val_it,
    verbose=2
)

#save model
model.save_weights('./models/classifier_aug.h5')

#use after model is trained
#model.load_weights('./models/classifier_aug.h5')
#print(model.evaluate(test_it))