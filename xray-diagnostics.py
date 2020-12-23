#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential


# In[2]:


from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import cv2
import os


# ## init the model

# In[3]:


model=Sequential()


# ## Adding convolution layers

# In[4]:


model.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation="relu"))


# In[5]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[6]:


model.add(Conv2D(32,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


# ## adding flatten layer

# In[7]:


model.add(Flatten())


# ## full connection

# In[8]:


model.add(Dense(activation="relu",units=512))


# ## output layer

# In[9]:


model.add(Dense(activation="sigmoid",units=1))


# In[10]:


model.summary()


# ## compile the model

# In[11]:


model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])


# # pre processing image

# In[12]:


labels = ['PNEUMONIA', 'NORMAL']
img_size = 150 
def get_img_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)


# In[13]:


num_of_test_samples = 600
batch_size = 32


# In[14]:


from keras.preprocessing.image import ImageDataGenerator 


# In[15]:


train_datagen=ImageDataGenerator(
rescale=1./255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True)


# In[16]:


test_datagen=ImageDataGenerator(rescale=1./255)


# In[17]:


x_train=train_datagen.flow_from_directory(
r'C:\Users\Hello\Desktop\chest x-ray dataset\chest_xray\train',
target_size=(64, 64),
batch_size=32,
class_mode="binary")


# In[18]:


validation_generator=test_datagen.flow_from_directory(
r'C:\Users\Hello\Desktop\chest x-ray dataset\chest_xray\val',
target_size=(64, 64),
batch_size=32,
class_mode="binary")


# In[19]:


x_test=test_datagen.flow_from_directory(
r'C:\Users\Hello\Desktop\chest x-ray dataset\chest_xray\test',
target_size=(64, 64),
batch_size=32,
class_mode="binary")


# In[20]:


model.fit_generator(
x_train,
steps_per_epoch=163,
epochs=3,
validation_data=validation_generator,
validation_steps=624,
verbose=1)


# In[21]:


test_accu = model.evaluate_generator(x_test,steps=624)


# In[22]:


print('The testing accuracy is :',test_accu[1]*100, '%')


# In[23]:


model=model.save('model.h5')

