#!/usr/bin/env python
# coding: utf-8

# # LetsGrowMore Data Science Internship (VIP)
# ## Name: kajal kashyap
# ## Title: Image to Pencil Sketch with Python
# ## Batch: March
# 

# ## step 1:Import libaries

# In[2]:


import numpy as np
import imageio
import matplotlib.pyplot as plt
import cv2


# ## step 2: Read IMAGE

# In[3]:


img=cv2.imread("dog.jpg")


# In[4]:


cv2.imshow("dog",img)


# ## step 3: Read IMAGE Image in RGB format

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)


# In[6]:


gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_img


# ## step 4: Read IMAGE Image in greyscale Format

# In[7]:


plt.imshow(gray_img)


# ##  step 5: Read IMAGE Inverting Image

# In[8]:


inverted_img=255-gray_img
plt.imshow(inverted_img)


# ## step 6: Read IMAGE blurred_Image

# In[10]:


smoothing_img=cv2.GaussianBlur(inverted_img,(21,21),sigmaX=0,sigmaY=0)
plt.imshow(smoothing_img)


# In[11]:


def dodgeV2(X,Y):
    return cv2.divide(X,255-Y,scale=256)


# In[12]:


#cv2.imshow("dog",pencil_sketch)
#v2.waitkey(0)


# ## step 7: Read IMAGE pencil_skech_image

# In[13]:


pencil_sktech=dodgeV2(gray_img,smoothing_img)
plt.imshow(pencil_sktech)


# In[ ]:





# In[ ]:




