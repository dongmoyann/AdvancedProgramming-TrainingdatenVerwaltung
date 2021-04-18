#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd 
import imutils
import cv2
import random
from enum import Enum
#from keras_segmentation.models.segnet import segnet


# In[13]:


class Mode(Enum):
    CHROM = 0
    TELO = 1


# In[14]:


#csvtrainname = '/home/inf/Bilder/bio1/data/train_data.csv'
#csvvalidname = '/home/inf/Bilder/bio1/data/valid_data.csv'
#csvtestname = '/home/inf/Bilder/bio1/data/test_data.csv'

#csvname = csvtestname



basepath = r"C:\Users\Admin\Documents\Master\Sommer\WPM-Advanced Programing\trainingsdaten"

chrtraindestpath = join(basepath, r'chromosomes\data\train')
teltraindestpath = join(basepath, r'telomeres\data\train')

chrvaliddestpath = join(basepath, r'chromosomes\data\valid')
telvaliddestpath = join(basepath, r'telomeres\data\valid')

chrtestdestpath = join(basepath, r'chromosomes\data\test')
teltestdestpath = join(basepath, r'telomeres\data\test')

assert(os.path.isdir(chrtraindestpath))
assert(os.path.isdir(teltraindestpath))
assert(os.path.isdir(chrvaliddestpath))
assert(os.path.isdir(telvaliddestpath))
assert(os.path.isdir(chrtestdestpath))
assert(os.path.isdir(teltestdestpath))

chrdestpath = chrtestdestpath
teldestpath = teltestdestpath

chrtrainmaskpath = join(basepath, r'chromosomes\masks\train')
teltrainmaskpath = join(basepath, r'telomeres\masks\train')

chrvalidmaskpath = join(basepath, r'chromosomes\masks\valid')
telvalidmaskpath = join(basepath, r'telomeres\masks\valid')

chrtestmaskpath = join(basepath, r'chromosomes\masks\test')
teltestmaskpath = join(basepath, r'telomeres\masks\test')

assert(os.path.isdir(chrtrainmaskpath))
assert(os.path.isdir(teltrainmaskpath))
assert(os.path.isdir(chrvalidmaskpath))
assert(os.path.isdir(telvalidmaskpath))
assert(os.path.isdir(chrtestmaskpath))
assert(os.path.isdir(teltestmaskpath))

chrmaskpath = chrtestmaskpath
telmaskpath = teltestmaskpath



imgcount = 0;
imgsize = 128


# In[4]:


# probably not needed
def write(fname, chrname, telname, cb, cc, tb, tc): 
  
    if isfile(fname): 
  
        df = pd.read_csv(f_name, index_col = 0) 
 
        data = [{'chr': chrname, 'tel': telname, 'cbright': cb, 'ccontrast': cc, 'tbright': tb, 'tcontrast':tc}] 
  
        # Creates DataFrame. 
        latest = pd.DataFrame(data) 


        df = pd.concat((df, latest), ignore_index = True, sort = False) 
  
    else: 
  
        # Providing range only because the data 
        # here is already flattened for when 
        # it was store in f_list
        data = [{'chr': chrname, 'tel': telname, 'cbright': cb, 'ccontrast':cc, 'tbright': tb, 'tcontrast':tc}] 
        df = pd.DataFrame(data) 

    df.to_csv(fname) 


# In[5]:


def makecolor(chromo, telo):
    global chromothresh
    global telothresh
    chromogray = cv2.cvtColor(chromo, cv2.COLOR_BGR2GRAY)
    telogray = cv2.cvtColor(telo, cv2.COLOR_BGR2GRAY)
    
    imgret = np.zeros((imgsize, imgsize,3), np.uint8)
    
    imgret[0:imgsize, 0:imgsize,1] = chromogray
    imgret[0:imgsize, 0:imgsize,0] = telogray
    
    return imgret


# In[6]:


def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


# In[7]:


print(chrdestpath)
print(chrmaskpath)


# In[ ]:





# In[8]:


files = []

for file in os.listdir(chrmaskpath):
    if file.endswith(".png"):
        files.append(file)
        
files.sort()
        
#for file in files:
#    print(file)
    
#print(files[len(files)-1])

#imgcount = getnumofelements(csvname)
#print(imgcount)
print(len(files))


# In[9]:


#chrdestpath = chrtraindestpath
#teldestpath = teltraindestpath

#chrmaskpath = chrtrainmaskpath
#telmaskpath = teltrainmaskpath

chrlist = []
tellist = []

for file in os.listdir(chrdestpath):
    if file.endswith(".png"):
        chrlist.append(os.path.join(chrdestpath, file))
        
for file in os.listdir(teldestpath):
    if file.endswith(".png"):
        tellist.append(os.path.join(teldestpath, file))
    
chrlist.sort()
tellist.sort()
assert len(chrlist) == len(tellist)
for i in range(len(chrlist)):
    #print(chrlist[i][:])
    #print(tellist[i][:])
    assert chrlist[i][-11:] == tellist[i][-11:]
print(chrdestpath)
print(chrmaskpath)
print(len(tellist))


# In[10]:


imgcount = len(chrlist)
print(imgcount)

#files = []
#for file in os.listdir(telmaskpath):
#    if file.endswith(".png"):
#        files.append(file)    
#files.sort()
#
#print(len(files))

cv2.namedWindow('mask')
theEnd = False
theNext = False
theThresh = False
mode = Mode.CHROM
thresh = 127

imgstart = 0
print(imgcount)
assert imgstart < imgcount

#df = pd.read_csv(csvname, index_col = 0) 

for i in range(imgstart, imgcount):

    contrast = 0
    bright = 0

    #if imgcount != 0:
    #    if row["tel"] <= files[len(files)-1]:
    #        continue

    while True:    

        chrimg = cv2.imread(chrlist[i],1)
        telimg = cv2.imread(tellist[i],1)
    
        if mode == Mode.CHROM:
            conc = chrimg
        else:
            conc = telimg
    
        concimg=apply_brightness_contrast(conc, bright, contrast)
    
        conccolor = makecolor(chrimg, telimg)
    

        
        if theThresh == True:
            ret,concimg=cv2.threshold(concimg,thresh,255,cv2.THRESH_BINARY)
            concimgdisp = cv2.addWeighted(concimg, 0.3, conccolor, 0.7, 0) 
        else:
            concimgdisp = concimg.copy()
    
        if mode == Mode.CHROM:
            cv2.putText(concimgdisp,os.path.basename(chrlist[i][:-4]), (2,12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(concimgdisp,os.path.basename(tellist[i][:-4]), (2,12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        #cv2.putText(concimgdisp,"b:{}".format(bright), (2,22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        #cv2.putText(concimgdisp,"c:{}".format(contrast), (2,32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(concimgdisp,"t:{}".format(thresh), (2,42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        while True:
            cv2.imshow('mask',concimgdisp)
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q") :
                theEnd = True
                break
            if key & 0xFF == ord("s") :
                if mode == Mode.CHROM and theThresh == True:
                    cv2.imwrite(join(chrmaskpath,os.path.basename(chrlist[i])), concimg)
                    theNext = True
                elif mode == Mode.TELO and theThresh == True:
                    cv2.imwrite(join(telmaskpath,os.path.basename(tellist[i])), concimg)
                    theNext = True
                break
            if key & 0xFF == ord("n") :
                theNext = True
                break
            if key & 0xFF == ord("i") :
                bright = bright + 1
                theThresh = False
                break
            if key & 0xFF == ord("k") :
                bright = bright - 1
                theThresh = False
                break
            if key & 0xFF == ord("o") :
                contrast = contrast + 1
                theThresh = False
                break
            if key & 0xFF == ord("l") :
                contrast = contrast - 1
                theThresh = False
                break
            if key & 0xFF == ord("u") :
                thresh = thresh + 1
                theThresh = True
                break
            if key & 0xFF == ord("j") :
                thresh = thresh - 1
                theThresh = True
                break  
            if key & 0xFF == ord("c") :
                if mode == Mode.TELO:
                    mode = Mode.CHROM
                break
            if key & 0xFF == ord("t") :
                if mode == Mode.CHROM:
                    mode = Mode.TELO
                break
        if theEnd == True:
            break
        if theNext == True:
            theNext = False
            break
            
    if theEnd == True:
        break      
        
 
cv2.destroyAllWindows()


# In[13]:


cv2.destroyAllWindows()


# In[16]:


2//2


# In[ ]:





# In[42]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


def showpics(chrlist, orglist, lastpics=10):

    assert lastpics <= len(chrlist)
    assert len(chrlist) == len(orglist)
    chrtail = chrlist[len(chrlist)-lastpics:]
    orgtail = orglist[len(orglist)-lastpics:]

    columns = 5
    rows = lastpics//columns
    fig=plt.figure(figsize=(15, 2*rows))
    for i in range(columns*rows):
        
        #ret,thresh1 = cv2.threshold(chrtail[i],1,255,cv2.THRESH_BINARY)
        #ret,thresh2 = cv2.threshold(chrtail[i],0,2,cv2.THRESH_BINARY)
        thresh1 = (chrtail[i]%2)*254
        thresh2 = (chrtail[i]//2)*254


        
        imgresult = np.zeros((imgsize, imgsize, 3), np.uint8)
        imgresult[:,:,0]=thresh1
        imgresult[:,:,1]=thresh2   
        fig.add_subplot(rows, columns, i+1)
        #plt.imshow(chrtail[i]*127)
        plt.imshow(cv2.addWeighted(imgresult, 0.2, orgtail[i], 0.8, 0))
        #plt.imshow(imgresult)
        


# In[43]:


imglist = []
orglist = []

i = 0
imgsize = 128

first = 300
last =  340

assert imgcount == len(tellist)
assert len(tellist) == len(chrlist)
assert first < last
assert last< imgcount

for i in range(first, last):

    
        chrimg = cv2.imread(join(chrmaskpath,chrlist[i]),1)
        telimg = cv2.imread(join(telmaskpath,tellist[i]),1)
        chrdest = cv2.imread(join(chrdestpath,chrlist[i]),1)
        teldest = cv2.imread(join(teldestpath,tellist[i]),1)
        
        telimg_inv = cv2.bitwise_not(telimg)
        chrimg_and = cv2.bitwise_and(chrimg, telimg_inv)
        
        telgray = cv2.cvtColor(telimg, cv2.COLOR_BGR2GRAY)
        chrgray = cv2.cvtColor(chrimg_and, cv2.COLOR_BGR2GRAY)
        
        telgray = (telgray) // 127
        chrgray = (chrgray) // 254
        
        #imgresult = np.zeros((imgsize, imgsize,1), np.uint8)
        #imgresult[0:imgsize, 0:imgsize,1] = chrgray
        #imgresult[0:imgsize, 0:imgsize,0] = telgray
        
        imgresult = cv2.bitwise_or(chrgray, telgray)
        #imgresult = chrgray
        
        #print(imgresult.shape)
        
        #print(chrimg.shape)
        assert(chrimg.shape == telimg.shape)
           
        
        if i > last:
            break
            
        imglist.append(imgresult)
        orglist.append(makecolor(chrdest, teldest))
        i = i + 1


# In[44]:


showpics(imglist, orglist, 40)


# In[57]:


#np.set_printoptions(threshold=np.inf)
#print(imglist[0])


# In[ ]:




