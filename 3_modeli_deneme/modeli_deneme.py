# -*- coding: utf-8 -*-

from keras.models import load_model
model = load_model('../2_modeli_olusturma/basarili_model.h5py')

model.summary()

import cv2
import numpy as np
import matplotlib.pyplot as plt
#%%
basliklar = ['5arka','5on','10arka','10on','20arka','20on',
             '50arka','50on','100arka','100on','200arka','200on']
def tahminYap(imgNumber):
    img = cv2.imread('./testVerileri/'+str(imgNumber)+'.jpg')
    img = cv2.resize(img, (200, 92)) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
   
    img = np.reshape(img,(-1,92,200,1))
    sonuc = model.predict(img)
    sonuc = basliklar[np.argmax(sonuc)]
    
    return sonuc
#%%
def goruntule(imgNumber):
    img = cv2.imread('testVerileri/'+str(imgNumber)+'.jpg')
    img = cv2.resize(img, (200, 92)) 
    
    plt.imshow(img,cmap='gray')
    plt.title('Banknot')
    plt.axis('off')
    plt.show()
#%%
goruntule(25)
#%%
tahminYap(25)

