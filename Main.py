
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Concatenate, Convolution2D, DepthwiseConv2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input, ReLU, concatenate, MaxPool2D, GlobalMaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import Model

batch_size = 16
epochs = 20
IMG_HEIGHT = 160
IMG_WIDTH = 160

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
        )

validation_datagen = ImageDataGenerator(rescale=1./255,                             
                      shear_range=0.2,
                      zoom_range=0.2,
                      horizontal_flip=True,
                      vertical_flip = False,
                      rotation_range=45,
                  width_shift_range=0.1,
                  height_shift_range=0.1,
                  brightness_range=None,
                  featurewise_center=False,
                  featurewise_std_normalization=False,
                  samplewise_center=False,
                  samplewise_std_normalization=False,
                  zca_whitening=False,
                  zca_epsilon=1e-06,
                  fill_mode = 'nearest'
                 )

test_datagen = ImageDataGenerator(rescale=1./255,
                                    # horizontal_flip=False,
                                    # vertical_flip = False,
                                    # brightness_range=None,
                                    # featurewise_center=False,
                                    # featurewise_std_normalization=False,
                                    # samplewise_center=False,
                                    # samplewise_std_normalization=False,
                                    # zca_whitening=False,
                                    # zca_epsilon=1e-06,
                                    # fill_mode = 'nearest'
                                    )

train_generator = train_datagen.flow_from_directory(
        'C:/Dataset/Data_cf/Train',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle = True,
        subset = 'training'
        )

validation_generator = train_datagen.flow_from_directory(
        'C:/Dataset/Data_cf/Validation',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle = True,
        subset = 'validation'
        )


        
STEP_SIZE_TRAIN = train_generator.n/batch_size
STEP_SIZE_VALID = validation_generator.n/batch_size


weather_in = Input(shape = (IMG_HEIGHT, IMG_WIDTH, 3))

weather = Conv2D(8, (2, 2),padding='same', activation='relu')(weather_in)
weather = BatchNormalization()(weather)
weather = DepthwiseConv2D(8,(2,2),padding='same', activation='relu')(weather)
weather = Conv2D(16,(1,1))(weather)
weather = BatchNormalization()(weather)
weather = DepthwiseConv2D(16,(2,2),padding='same', activation='relu')(weather)
weather = Conv2D(32,(1,1))(weather)
weather = BatchNormalization()(weather)
weather = DepthwiseConv2D(32,(2,2),padding='same', activation='relu')(weather)
weather = Conv2D(32,(1,1))(weather)
weather = DepthwiseConv2D(32,(2,2),padding='same', activation='relu')(weather)
weather = Conv2D(64,(1,1))(weather)
weather = BatchNormalization()(weather)
weather = DepthwiseConv2D(64,(2,2),padding='same', activation='relu')(weather)
weather = Conv2D(64,(1,1))(weather)
weather = DepthwiseConv2D(64,(2,2),padding='same', activation='relu')(weather)
weather = Conv2D(128,(1,1))(weather)
weather = BatchNormalization()(weather)
weather = MaxPooling2D((2, 2))(weather)

weather = Flatten()(weather)

edge_in =  Input(shape=(1,))
edge = Dense(1, activation = 'relu')(edge_in)

hsv_in = Input(shape=(1,))
hsv = Dense(1, activation = 'relu')(hsv_in)

merged = concatenate([weather, edge, hsv])

fc = Dense(128, activation = 'relu')(merged)
drop = Dropout(0.5)(fc)
fc = Dense(32, activation = 'relu')(drop)
output = Dense(4, activation = 'softmax')(fc)

combinedModel = Model(inputs=[weather_in, edge_in, hsv_in], outputs=[output])

combinedModel.summary()



#model 후처리
threshold_x = 150
threshold_y = 50

lower_blue = np.array([90,50,50])
upper_blue = np.array([110,255,255])

width = IMG_WIDTH
height = IMG_HEIGHT

def canny(X):
    edges = []
    temp = []
    for img in X:
        edge = cv2.Canny((img*255).astype(np.uint8),threshold_x,threshold_y)
        temp.append(np.sum(edge)/(width))
    temp = np.asarray(temp)
    edges.append(temp)
    edges = np.asarray(edges)
    return edges

def hsv(batch):
    hsv_blue = []
    for img in batch:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
        hsv_blue.append(np.sum(mask)/(width*height))
    hsv_blue = np.asarray(hsv_blue)
    return hsv_blue
    
    
def generate_generator_multiple(train_generator):
    while True:
        X1i = train_generator.next()
        X2i = canny(X1i[0])
        X3i = hsv(X1i[0])
        yield [X1i[0], X2i[0], X3i], X1i[1]  #Yield both images and their mutual label
            
def val_generator_multiple(validation_generator):
    while True:
        X1i = validation_generator.next()
        X2i = canny(X1i[0])
        X3i = hsv(X1i[0])
        yield [X1i[0], X2i[0], X3i], X1i[1]  #Yield both images and their mutual label
            
inputgenerator = generate_generator_multiple(train_generator)
valgenerator = val_generator_multiple(validation_generator)

combinedModel.compile(loss = "categorical_crossentropy",
                      optimizer ='Adam',
              metrics=['accuracy'])
          

callbacks = [ EarlyStopping(patience=5, verbose=1),
             ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
             ModelCheckpoint('Classify_cpu_2nd.h5', verbose=1, save_best_only=True, save_weights_only=True) ]



#combinedModel.load_weights('Classify_cpu_2nd.h5')

history = combinedModel.fit(
            inputgenerator,
            steps_per_epoch = STEP_SIZE_TRAIN,
            epochs=epochs,
            validation_data = valgenerator,
            validation_steps = STEP_SIZE_VALID,
            callbacks=callbacks)

combinedModel.save_weights('Mobilenet_edge_hsv_combined_160x160_20epochs.h5')

#accuracy graph
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()




#TEST

def canny_test(X):
    edges = []
    temp = []
    for img in X:
        edge = cv2.Canny((img*255).astype(np.uint8),threshold_x,threshold_y)
        temp.append(np.sum(edge)/255)
    temp = np.asarray(temp)
    edges.append(temp)
    edges = np.asarray(edges)
    return edges

def test_generator_multiple(test_generator):
    while True:
        Y = test_generator.next()
        Y2 = canny_test(Y[0])
        Y3 = hsv(Y[0])
        yield [Y[0], Y2[0], Y3], Y[1]   #Yield both images and their mutual label

test_generator = test_datagen.flow_from_directory(
    'C:/Dataset/cf_test',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle = False)


# Saved Model 부르기 
new_model = combinedModel
new_model.load_weights('Mobilenet_edge_hsv_combined_160x160_25epochs.h5')

testgenerator = test_generator_multiple(test_generator)
STEP_SIZE_TEST=test_generator.n//batch_size
total_test = test_generator.n

#model predict
class_names = ['HAZE', 'RAINY', 'SNOWY', 'SUNNY']
predictions = new_model.predict(testgenerator,steps = STEP_SIZE_TEST, verbose=1)

print('Number of test gen',test_generator.n)
print('prediction shape',predictions.shape)

count = 0
test_images = np.zeros((total_test,IMG_HEIGHT,IMG_WIDTH,3))
test_labels = np.zeros((total_test,4))
for batch, CLS in test_generator:
    for j in range(len(batch)):
        img = batch[j]
        test_images[count]=img
        cls_temp = CLS[j]
        test_labels[count]=cls_temp
        count += 1  
        if count == (total_test):
            break
    if count == (total_test):
        break

test_label_cls=np.argmax(test_labels,axis=1)
prediction_label=np.argmax(predictions,axis=1)
print(prediction_label,prediction_label.shape)
print(test_label_cls,test_label_cls.shape)
'''
#model evaluate
test_loss, test_acc = new_model.evaluate(testgenerator,steps=STEP_SIZE_TEST, verbose=2)
print(test_acc)

acc_each=np.zeros(4)
for i in range(test_generator.n):
  if prediction_label[i] == test_label_cls[i]:
    if prediction_label[i] == 0:
      acc_each[0] +=1
    elif prediction_label[i] == 1:
      acc_each[1] +=1
    elif prediction_label[i] == 2:
      acc_each[2] +=1
    elif prediction_label[i] == 3:
      acc_each[3] +=1

print(acc_each/100)

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label_onehot, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  true_label = np.argmax(true_label_onehot)

  plt.imshow(img)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),  
                                class_names[true_label]),
                                color=color,fontsize=5)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label_onehot = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  true_label = np.argmax(true_label_onehot)
  thisplot = plt.bar(range(4), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

num_rows = 8
num_cols = 5
num_images = num_rows*num_cols
plt.tight_layout
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()'''


