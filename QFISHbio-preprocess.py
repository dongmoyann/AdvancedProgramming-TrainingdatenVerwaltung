#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd 
import imutils
import cv2
from enum import Enum


# In[2]:


imgcount = 0

basepath = r"C:\Users\Admin\Documents\Master\Sommer\WPM-Advanced Programing\bio"

chrdestpath = join(basepath, r'chromosomes\test')
chrdestfilename = "chr_{0:0=5d}.png".format(imgcount);

teldestpath = join(basepath, r'telomeres\test')
teldestfilename = "png_{0:0=5d}.png".format(imgcount);

assert os.path.isdir(chrdestpath)
assert os.path.isdir(teldestpath)

orignamechr = 'orig0_chr_adj_1700-1000'
orignametel = 'orig0_tel_adj_1700-1000'

chrsourcepath = join(basepath, r'chromosomes\orig')
chrsourcefilename = orignamechr + '.png'
chrsourcefullpath = join(chrsourcepath, chrsourcefilename)

telsourcepath = join(basepath, r'telomeres\orig')
telsourcefilename = orignametel + '.png'
telsourcefullpath = join(telsourcepath, telsourcefilename)


assert os.path.isdir(chrsourcepath)
assert os.path.isdir(telsourcepath)
assert os.path.isfile(chrsourcefullpath)
assert os.path.isfile(telsourcefullpath)

f_name = "/home/inf/Bilder/bio1/data/test_data.csv"

chrimg = cv2.imread(chrsourcefullpath,1)
telimg = cv2.imread(telsourcefullpath,1)

assert chrimg.shape == telimg.shape

height= chrimg.shape[0]
width= chrimg.shape[1]

print(height)
print(width)

rectanglesize=min(height, width)//6
crosssize=rectanglesize//12
crossthick=1
imgsize = 128
rectsize = 80
startpoint = (500,240)
endpoint = (startpoint[0]+rectsize, startpoint[1]+rectsize )


# In[3]:


assert(os.path.isfile(chrsourcefullpath))
assert(os.path.isfile(telsourcefullpath))

chrimg = cv2.imread(chrsourcefullpath,1)
telimg = cv2.imread(telsourcefullpath,1)

chrimgrect = chrimg.copy()
telimgrect = telimg.copy()


# In[4]:


numofrects = min(int((chrimgrect.shape[1] - 2*startpoint[0])/rectsize),int((chrimgrect.shape[0] - 2*startpoint[1])/rectsize))
print(numofrects)
        


# In[5]:


def grid(chrimgrect, startpoint, rectsize, numofrects = None):
    positions = []
    if numofrects == None:
        numofrects = min(int((chrimgrect.shape[1] - 2*startpoint[0])/rectsize),int((chrimgrect.shape[0] - 2*startpoint[1])/rectsize))
    for i in range(numofrects):
        for j in range(numofrects):
            polygon = []
            spoint = (startpoint[0]+i*rectsize, startpoint[1]+j*rectsize )
            epoint = (startpoint[0]+i*rectsize + rectsize, startpoint[1]+j*rectsize + rectsize)
            polygon.append(spoint)
            polygon.append(epoint)
            cnt = np.array(polygon)
            positions.append(cv2.boundingRect(cnt))
            #_,_,_,_ = cv2.boundingRect(cnt)
            chrimgrect = cv2.rectangle(chrimgrect, spoint, epoint, (255,0,0), 1) 
    return chrimgrect, positions, numofrects


# In[6]:


positions = []
numofrects = None

while True:
    chrimgrect = chrimg.copy()
    positions.clear()
    chrimgrect, positions, numofrects = grid(chrimgrect, startpoint, rectsize, numofrects)
    cv2.imshow('chrimage',chrimgrect) 
    key=cv2.waitKeyEx(100)

    if key & 0xFF == ord("q") : 
        break
    if key & 0xFF == ord("s") :
        i = 0
        for position in positions:
            single = np.zeros((rectsize,rectsize,chrimg.shape[2]), dtype="uint8")
            singlenorm = np.zeros((128,128,chrimg.shape[2]), dtype="uint8")
            x,y,w,h = position
            assert (w-1) == rectsize and (h-1) == rectsize
            single = chrimg.copy()[y:y+h-1,x:x+w-1,:]
            assert single.shape[0] == rectsize and single.shape[1] == rectsize
            fname = "c{}_".format(orignamechr) + "{0:0=5d}.png".format(i)
            singlenorm = cv2.resize(single,(128,128), interpolation = cv2.INTER_AREA)
            cv2.imwrite(join(chrdestpath,fname), singlenorm)      
            single = telimg.copy()[y:y+h-1,x:x+w-1,:]
            fname = "t{}_".format(orignametel) + "{0:0=5d}.png".format(i)   
            singlenorm = cv2.resize(single,(128,128), interpolation = cv2.INTER_AREA)
            cv2.imwrite(join(teldestpath,fname), singlenorm)
            i += 1
        break
    if key & 0xFF == ord("+") : 
        numofrects += 1
    if key & 0xFF == ord("-") :
        if numofrects > 0:
            numofrects -= 1        
    if key == 2424832 : 
        if startpoint[0] - 20 > 0:
            startpoint = (startpoint[0] - 20, startpoint[1] )
    if key == 2490368 : 
        if startpoint[1] - 20 > 0:
            startpoint = (startpoint[0], startpoint[1] - 20)
    if key == 2555904 : 
        if startpoint[0] + 20 < chrimgrect.shape[1]:
            startpoint = (startpoint[0] + 20, startpoint[1])
    if key == 2621440 :
        if startpoint[1] + 20 < chrimgrect.shape[0]:
            startpoint = (startpoint[0], startpoint[1] + 20)


cv2.destroyAllWindows() 


# In[7]:


cv2.destroyAllWindows()


# In[6]:


cv2.imshow('telimage',telimgrect) 
key=cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




