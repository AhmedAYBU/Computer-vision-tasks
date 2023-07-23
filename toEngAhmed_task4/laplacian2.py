# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 22:22:07 2023

@author: Ahmed KASABASHI
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

laplacian_kernel = np.array([[0, -1, 0],
                              [-1, 4, -1],
                              [0, -1, 0]])

# for kernel_size=5 for our function

# laplacian_kernel = np.array([[0, 0, -1, 0, 0],
#                               [0, -1, -2, -1, 0],
#                               [-1, -2, 16, -2, -1],
#                               [0, -1, -2, -1, 0],
#                               [0, 0, -1, 0, 0]])


print(laplacian_kernel)

def mine_laplacian(img):
    
    # for more precision during of convolution we convert the image to float32
    img=np.float64(img) #you can try to use np,float32(img)
    
    #GET THE dimensions of the image and the kernal size
    img_height,img_width=img.shape[:2]
    kernel_size=laplacian_kernel.shape[0]
    print("kernel_size=",kernel_size)
    
    # creat new empty image
    output=np.zeros_like(image)
    # print(output)
    
    # Apply the Laplacian kernel using convolution
    
    for y in range(1,img_height-1):
        for x in range(1,img_width-1):
            region=img[y-1:y+2, x-1:x+2]
            conv_result=np.sum(region*laplacian_kernel)
    
    # for kernel_size=5 for our function
    # for y in range(kernel_size // 2, img_height - kernel_size // 2):
    #     for x in range(kernel_size // 2, img_width - kernel_size // 2):
    #         region=img[y - kernel_size // 2:y + kernel_size // 2 + 1, x - kernel_size // 2:x + kernel_size // 2 + 1]
    #         conv_result=np.sum(region*laplacian_kernel)

            
            output[y,x]=conv_result
            
    #convert the output back to uint8
    output=np.uint8(np.clip(output,0,255))
    
# In the line output = np.uint8(np.clip(output, 0, 255)), the values 0 and 255 are used as the lower and upper bounds for clipping the pixel values in the output image.

# The np.clip() function is used to limit the range of pixel values in the output image to the specified lower and upper bounds. Any pixel value below the lower bound will be set to the lower bound, and any pixel value above the upper bound will be set to the upper bound.

# In this case:

# 0 is the lower bound, indicating that any negative pixel values resulting from the Laplacian filtering will be clipped and set to 0 (minimum intensity, black).
# 255 is the upper bound, indicating that any pixel values above 255 resulting from the Laplacian filtering will be clipped and set to 255 (maximum intensity, white).
# The purpose of this clipping operation is to ensure that the output image remains within the valid range of pixel values for an 8-bit grayscale image, which ranges from 0 to 255. Clipping prevents any potential underflow or overflow in pixel values that may occur during the Laplacian filtering process.

# By using np.uint8, the data type of the output image is set to an 8-bit unsigned integer (uint8), which enforces the valid range of 0 to 255 for the pixel values.

# In summary, the line output = np.uint8(np.clip(output, 0, 255)) ensures that the output image contains valid pixel values within the 0 to 255 range, regardless of the results obtained from the Laplacian filtering.    

    return output
    


image=cv2.imread("bird.jpg",0)
lap1=cv2.Laplacian(image,cv2.CV_64F,ksize=1)
lap2=np.uint8(np.absolute(lap1))

mine=mine_laplacian(image)

titles=["image","lap1","lap2","mine"]
images=[image,lap1,lap2,mine]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i],"gray")
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
    
plt.show()
