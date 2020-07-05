import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import math
import cv2


def get_real_coordinates(ratio, x1, y1, x2, y2):

    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2 ,real_y2)



path = 'show/6e83b58f-7446-4d94-81da-ab17c093f47d_0056_roi0x_38.png'
direction = 'x'

img = cv2.imread(path)
x1,x2,y1,y2,key, new_probs, ratio = 16,272,64,320,10.0,0.195180132985,9.375
(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)


#gtx1,gty1,gtz1, gtx2,gty2,gtz2, label = 0.623582368001,3.55429015758,9.90384921523,12.623582368,15.5542901576,23.9038492152,6.0
gtx1,gty1,gtz1, gtx2,gty2,gtz2, label = 27.4743763726,8.47437637262,4.47437637262,50.5256236274,31.5256236274,27.5256236274,2.0

#cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (255,0,0),1)
#cv2.imshow('img', img)
#cv2.rectangle(img,(int(round(gty1)), int(round(gtz1))), (int(round(gty2)), int(round(gtz2))), (0,0,255),1)
if direction == 'x':
    cv2.rectangle(img,(real_x1,real_y1), ( real_x2,real_y2), (255,0,0),1)
    cv2.rectangle(img,(int(round(gtz1)), int(round(gty1))), (int(round(gtz2)), int(round(gty2))), (0,0,255),1)
elif direction == 'y':
    cv2.rectangle(img,(real_x1,real_y1), ( real_x2,real_y2), (255,0,0),1)
    cv2.rectangle(img,(int(round(gtz1)), int(round(gtx1))), (int(round(gtz2)), int(round(gtx2))), (0,0,255),1)
elif direction == 'z':
    cv2.rectangle(img,(real_x1,real_y1), ( real_x2,real_y2), (255,0,0),1)
    cv2.rectangle(img,(int(round(gty1)), int(round(gtx1))), (int(round(gty2)), int(round(gtx2))), (0,0,255),1)




x1,x2,y1,y2,key, new_probs, ratio = 96,320,320,544,2.0,0.186601877213,9.375
(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
gtx1,gty1,gtz1, gtx2,gty2,gtz2, label = 27.4743763726,8.47437637262,4.47437637262,50.5256236274,31.5256236274,27.5256236274,2.0
if direction == 'x':
    cv2.rectangle(img,(real_x1,real_y1), ( real_x2,real_y2), (255,0,0),1)
    cv2.rectangle(img,(int(round(gtz1)), int(round(gty1))), (int(round(gtz2)), int(round(gty2))), (0,0,255),1)
elif direction == 'y':
    cv2.rectangle(img,(real_x1,real_y1), ( real_x2,real_y2), (255,0,0),1)
    cv2.rectangle(img,(int(round(gtz1)), int(round(gtx1))), (int(round(gtz2)), int(round(gtx2))), (0,0,255),1)
elif direction == 'z':
    cv2.rectangle(img,(real_x1,real_y1), ( real_x2,real_y2), (255,0,0),1)
    cv2.rectangle(img,(int(round(gty1)), int(round(gtx1))), (int(round(gty2)), int(round(gtx2))), (0,0,255),1)

x1,x2,y1,y2,key, new_probs, ratio = 400,592,272,464,9.0,0.195500463247,9.375
(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
gtx1,gty1,gtz1, gtx2,gty2,gtz2, label = 29.2501162797,28.2501162797,45.2501162797,50.7498837203,49.7498837203,63.0,9.0
if direction == 'x':
    cv2.rectangle(img,(real_x1,real_y1), ( real_x2,real_y2), (255,0,0),1)
    cv2.rectangle(img,(int(round(gtz1)), int(round(gty1))), (int(round(gtz2)), int(round(gty2))), (0,0,255),1)
elif direction == 'y':
    cv2.rectangle(img,(real_x1,real_y1), ( real_x2,real_y2), (255,0,0),1)
    cv2.rectangle(img,(int(round(gtz1)), int(round(gtx1))), (int(round(gtz2)), int(round(gtx2))), (0,0,255),1)
elif direction == 'z':
    cv2.rectangle(img,(real_x1,real_y1), ( real_x2,real_y2), (255,0,0),1)
    cv2.rectangle(img,(int(round(gty1)), int(round(gtx1))), (int(round(gty2)), int(round(gtx2))), (0,0,255),1)

x1,x2,y1,y2,key, new_probs, ratio = 400,592,256,480,4.0,0.224642470479,9.375
(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
gtx1,gty1,gtz1, gtx2,gty2,gtz2, label = 32.8657509503,12.8657509503,26.8657509503,57.1342490497,37.1342490497,51.1342490497,6.0
if direction == 'x':
    cv2.rectangle(img,(real_x1,real_y1), ( real_x2,real_y2), (255,0,0),1)
    cv2.rectangle(img,(int(round(gtz1)), int(round(gty1))), (int(round(gtz2)), int(round(gty2))), (0,0,255),1)
elif direction == 'y':
    cv2.rectangle(img,(real_x1,real_y1), ( real_x2,real_y2), (255,0,0),1)
    cv2.rectangle(img,(int(round(gtz1)), int(round(gtx1))), (int(round(gtz2)), int(round(gtx2))), (0,0,255),1)
elif direction == 'z':
    cv2.rectangle(img,(real_x1,real_y1), ( real_x2,real_y2), (255,0,0),1)
    cv2.rectangle(img,(int(round(gty1)), int(round(gtx1))), (int(round(gty2)), int(round(gtx2))), (0,0,255),1)





#cv2.imshow('img', img)
#cv2.imwrite('./results_imgs/{}.png'.format('0191_roi1x_28'),img)


#plt.imshow(img[int(round(gtx1)):int(round(gtx2)), int(round(gtz1)):int(round(gtz2))],cmap='gray')
#plt.imshow(img[int(round(gtz1)):int(round(gtz2)), int(round(gtx1)):int(round(gtx2))],cmap='gray')
plt.imshow(img,cmap='gray')
plt.show()

