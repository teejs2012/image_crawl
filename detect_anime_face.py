
# coding: utf-8

# In[2]:


import cv2
import sys
import os.path


# In[18]:


cascade_file = "model/lbpcascade_animeface.xml"

# In[38]:


result_dir = "results"


# In[39]:


if not os.path.isdir(result_dir):
    os.makedirs(result_dir)


# In[43]:


cascade = cv2.CascadeClassifier(cascade_file)

for filename in os.listdir('.'):
    if not ".png" in filename:
        continue
    try:
        filename_no_ext = filename[:-4]
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = cascade.detectMultiScale(gray,
                                         # detector options
                                         scaleFactor = 1.1,
                                         minNeighbors = 5,
                                         minSize = (24, 24))
        for i,(x, y, w, h) in enumerate(faces):
            y_low = int(y-0.25*h)
            if y_low<0: y_low=0
            x_low = int(x-0.25*w) 
            if x_low<0: x_low=0

            y_high = int(y+1.25*h) 
            if y_high>image.shape[0]-1: y_high=image.shape[0]-1
            x_high = int(x+1.25*w) 
            if x_high>image.shape[1]-1: x_high=image.shape[1]-1
            cv2.imwrite(os.path.join(result_dir,str.format("{0}-{1}.png",filename_no_ext,i)),image[y_low:y_high,x_low:x_high])
    except:
        print(filename)
    os.remove(filename)
