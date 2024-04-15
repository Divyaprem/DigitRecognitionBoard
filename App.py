#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install pygame


# In[3]:


import pygame , sys , os
from pygame.locals import *
from pygame import mixer
import numpy as np


# In[4]:


from keras.models import load_model
import cv2


# Initialize the pygame

# In[5]:


pygame.init()


# In[6]:


pygame.mixer.init()


# In[7]:


mixer.music.set_volume(0.9)

soundeffect0 = pygame.mixer.Sound("Desktop/Sound(Digits)/zero.wav")
soundeffect1 = pygame.mixer.Sound("Desktop/Sound(Digits)/one.wav")
soundeffect2 = pygame.mixer.Sound("Desktop/Sound(Digits)/two.wav")
soundeffect3 = pygame.mixer.Sound("Desktop/Sound(Digits)/three.wav")
soundeffect4 = pygame.mixer.Sound("Desktop/Sound(Digits)/four.wav")
soundeffect5 = pygame.mixer.Sound("Desktop/Sound(Digits)/five.wav")
soundeffect6 = pygame.mixer.Sound("Desktop/Sound(Digits)/six.wav")
soundeffect7 = pygame.mixer.Sound("Desktop/Sound(Digits)/seven.wav")
soundeffect8 = pygame.mixer.Sound("Desktop/Sound(Digits)/eight.wav")
soundeffect9 = pygame.mixer.Sound("Desktop/Sound(Digits)/nine.wav")


# In[8]:


WINDOWSIZEX = 640
WINDOWSIZEY = 480
BOUNDRYINC = 5
WHITE=(255,255,255)
BLACK =(0,0,0)
RED =(255,0,0)

IMAGESAVE = False

MODEL = load_model("bestmodel.h5")

LABELS = {0:"Zero" ,1:"One",2:"Two",3:"Three",4:"Four",5:"Five",6:"Six",7:"Seven",8:"Eight",9:"Nine"}
        
FONT = pygame.font.Font(None,18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX,WINDOWSIZEY))
##WHILE_INT= DISPLAYSURF.mp_rgb(WHITE)
pygame.display.set_caption("Digit Recognition Board")
icon = pygame.image.load("Desktop/eye-scanner.png")
pygame.display.set_icon(icon)


# In[1]:


iswriting= False

PREDICT = True

number_xcord = []
number_ycord = []
image_cnt = 1

while True:
    
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit
        if event.type == MOUSEMOTION and iswriting:
            xcord,ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4 ,0)
            
            number_xcord.append(xcord)
            number_ycord.append(ycord)
            
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True
        
        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)
            
            rect_min_x , rect_max_x= max(number_xcord[0]- BOUNDRYINC, 0) , min(WINDOWSIZEX, number_xcord[-1]+BOUNDRYINC)
            rect_min_Y , rect_max_Y= max(number_ycord[0]- BOUNDRYINC, 0) , min(number_ycord[-1]+BOUNDRYINC,WINDOWSIZEX)
            
            number_xcord = []
            number_ycord = []
            
            ing_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x , rect_min_Y:rect_max_Y].T.astype(np.float32)
            if IMAGESAVE:
                cv2.imwrite("image.png")
                image_cnt +=1 
                
            if PREDICT:
                image= cv2.resize(ing_arr,(28,28))
                image = np.pad(image, (10,10), "constant", constant_values=0)
                image = cv2.resize(image, (28,28))/255
                
                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])
                
                textSurface = FONT.render(label ,True , BLACK ,WHITE)
                textRecObj = textSurface.get_rect()
                textRecObj.left ,textRecObj.bottom = rect_min_x, rect_min_Y
                
                DISPLAYSURF.blit(textSurface, textRecObj)
                
                if label == "Zero":
                    soundeffect0.play()
                elif label == "One":
                    soundeffect1.play()
                elif label == "Two":
                    soundeffect2.play()
                elif label == "Three":
                    soundeffect3.play()
                elif label == "Four":
                    soundeffect4.play()
                elif label == "Five":
                    soundeffect5.play()
                elif label == "Six":
                    soundeffect6.play()
                elif label == "Seven":
                    soundeffect7.play()
                elif label == "Eight":
                    soundeffect8.play() 
                elif label == "Nine":
                    soundeffect9.play()    
            if event.type == KEYDOWN:
                if event.unicode == "n":
                    DISPLAYSURF.fill(BLACK)
    pygame.display.update()             


# In[ ]:





# In[ ]:





# In[ ]:




