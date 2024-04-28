import pygame
from pygame.locals import *
import numpy as np
import keras
from keras.models import load_model
import sys
import cv2

wx=640
wy=480
bound=5
white=(255,255,255)
black=(0,0,0)
red=(255,0,0)
model=load_model("C://Users//swapn//Downloads//Handwritten Digit Recognition//dr.keras")
labels={0:"Zeros",1:"One",2:"Two",3:"Three",4:"Four",5:"Five",6:"Six",7:"Seven",8:"Eight",9:"Nine"} # Output by model converted to corresponding number in the text form

pygame.init()

font = pygame.font.Font(None, 18)

ds=pygame.display.set_mode((wx,wy))
pygame.display.set_caption("Drawing Board")

predict=True
imgsave=False
iswriting=False
xcn=[]
ycn=[]
imgcnt=0

while True:
    for event in pygame.event.get():

        if event.type==QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type==MOUSEMOTION and iswriting:
            xc,yc=event.pos
            pygame.draw.circle(ds,white,(xc,yc),4,0)

            xcn.append(xc)
            ycn.append(yc)

        if event.type==MOUSEBUTTONDOWN:
            iswriting=True

        if event.type==MOUSEBUTTONUP:
            iswriting=False

            xcn=sorted(xcn)
            ycn=sorted(ycn)

            rectminx,rectmaxx=max(xcn[0]-bound,0),min(wx,xcn[-1]+bound)
            rectminy,rectmaxy=max(ycn[0]-bound,0),min(ycn[-1]+bound,wx)

            xcn=[]
            ycn=[]

            imgarr=np.array(pygame.PixelArray(ds))[rectminx:rectmaxx,rectminy:rectmaxy].T.astype(np.float32)

            if predict:
                image=cv2.resize(imgarr,(28,28))
                image=np.pad(image,(10,10),'constant', constant_values=0)
                image=cv2.resize(image,(28,28))/255
                
                label = str(labels[np.argmax(model.predict(image.reshape(1,28,28,1)))])

                ts=font.render(label,True, red, white)
                recttext=ts.get_rect()
                recttext.left,recttext.bottom=rectminx,rectmaxy
                ds.blit(ts, (recttext.left, recttext.bottom))

        pygame.display.update()