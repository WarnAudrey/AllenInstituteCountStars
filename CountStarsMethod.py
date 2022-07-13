#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg
from PIL import Image
from time import time
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage.io import imread
import glob
from math import sqrt


# In[2]:


im = Image.open("jake-weirick-Q_RBVFFXR_g-unsplash.jpg")
plt.imshow(im)


# In[3]:


#Convert image to B&W to make star counting easier
def monochrome(picture, threshold):
    """loops over the pixels in the loaded image, 
    replacing the RGB values of each with either 
    black or white depending on whether its total 
    luminance is above or below some threshold 
    passed in by the user"""
    black = (0, 0, 0)
    white = (255, 255, 255)
    xsize, ysize = picture.size
    temp = picture.load()
    for x in range(xsize):
        for y in range(ysize):
            r,g,b = temp[x,y]
            if r+g+b >= threshold: 
                temp[x,y] = black
            else:
                temp[x,y] = white


# In[4]:


#Get another copy to convert to B/W
bwpic = Image.open("jake-weirick-Q_RBVFFXR_g-unsplash.jpg")

#Remember, this threshold is a scalar, not an RGB triple
#we're looking for pixels whose total color value is 600 or greater
monochrome(bwpic,200+200+200)
plt.imshow(bwpic);


# In[5]:


BLACK = (0,0,0)
RED = (255,0,0)
def count(picture):
    """scan the image top to bottom and left to right using a nested loop.
    when black pixel is found, increment the count, then call the fill
    function to fill in all the pixels connected to that one."""
    xsize, ysize = picture.size
    temp = picture.load()
    result = 0
    for x in range(xsize):
        for y in range(ysize):
            if temp[x,y] == BLACK:
                result += 1
                fill(temp,xsize,ysize,x,y)
    return result


# In[6]:


#fill function
def fill(picture, xsize, ysize, x, y):
    """recursive fill function. It starts by checking the pixel 
    it has been asked to look at. If that pixel is not black, 
    this 'fill' returns immediately: there is nothing for it to do.
    If the pixel is black, 'fill' turns it red and then calls 'fill' 
    on its left-hand neighbor (if it has one). It then goes on to 
    call 'fill' for the right-hand, top, and bottom neighbors as well.
    This has the same effect as using an explicit work queue; each call
    to 'fill' takes care of one pixel, then calls 'fill' again to take 
    care of the neighbors"""
    if picture[x, y] != BLACK:
        return
    picture[x, y] = RED
    if x > 0:
        fill(picture, xsize, ysize, x-1, y)
    if x < (xsize-1):
        fill(picture, xsize, ysize, x+1, y)
    if y > 0:
        fill(picture, xsize, ysize, x, y-1)
    if y < (ysize-1):
        fill(picture, xsize, ysize, x, y+1)


# In[7]:


count(bwpic)


# In[19]:


#Evaluate the accuracy of the model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report


# In[9]:


npframe = np.array(bwpic.getdata())
df = pd.DataFrame(npframe)
df


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(df[0], df[1])


# In[11]:


X_train= X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)


# In[12]:


gnb = GaussianNB()


# In[13]:


y_pred = gnb.fit(X_train, y_train).predict(X_test)


# In[14]:


gnb.score(X_test, y_test)


# In[16]:


cm = confusion_matrix(y_test,y_pred)
cm


# In[18]:


color = 'white'
matrix = plot_confusion_matrix(gnb, X_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()


# In[20]:


print(classification_report(y_test, y_pred))


# In[ ]:




