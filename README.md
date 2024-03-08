# Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step 1
Import the required libraries.

### Step2
Convert the image from BGR to RGB.

### Step 3
Apply the required filters for the image separately.

### Step 4
Plot the original and filtered image by using matplotlib.pyplot.

### Step 5
End the program.


## Program:
### Developed By   :MANOGARAN S
### Register Number:212223240081
</br>

### 1. Smoothing Filters

i) Using Averaging Filter
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
image1 = cv2.imread('tiger.jpeg')
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
kernel = np.ones((11,11), np.float32)/121
image3 = cv2.filter2D(image2,-1, kernel)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title('Orignal')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title('Filtered')
plt.axis('off')



```
ii) Using Weighted Averaging Filter
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt
image1 = cv2.imread('tiger.jpeg')
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
kernel2 = np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image3 = cv2.filter2D(image2,-1, kernel2)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title('Origna`l')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title('Filtered')
plt.axis('off')




```
iii) Using Gaussian Filter
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image1 = cv2.imread('tiger.jpeg')
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
image3 = cv2.GaussianBlur(src=image2, ksize=(11,11), sigmaX=0, sigmaY=0)

plt.figure(figsize=(9, 9))
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image3)
plt.title('Filtered')
plt.axis('off')

plt.show()





```

iv) Using Median Filter
```Python



import cv2
import numpy as np
import matplotlib.pyplot as plt

image1 = cv2.imread('tiger.jpeg')
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
median = cv2.medianBlur(src=image2, ksize=11)
plt.figure(figsize=(9, 9))
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image3)
plt.title('Filtered')
plt.axis('off')

plt.show()





```

### 2. Sharpening Filters
i) Using Laplacian Kernal
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image1 = cv2.imread('tiger.jpeg')
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
kernel3=np.array([[0,1,0],[1,-4,1],[0,1,0]])
image3=cv2.filter2D(image2,-1,kernel3)

plt.figure(figsize=(9, 9))
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image3)
plt.title('Filtered')
plt.axis('off')

plt.show()





```
ii) Using Laplacian Operator
```Python
import cv2
import numpy as np
import matplotlib.pyplot as plt

image1 = cv2.imread('tiger.jpeg')
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
new_image=cv2.Laplacian(image2,cv2.CV_64F)

plt.figure(figsize=(9, 9))
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image3)
plt.title('Filtered')
plt.axis('off')

plt.show()





```

## OUTPUT:
### 1. Smoothing Filters
</br>

i) Using Averaging Filter
![alt text](<Screenshot 2024-03-08 171137.png>)

ii) Using Weighted Averaging Filter
![alt text](<Screenshot 2024-03-08 171149.png>)

iii) Using Gaussian Filter
![alt text](<Screenshot 2024-03-08 171459.png>)

iv) Using Median Filter
![alt text](<Screenshot 2024-03-08 171510.png>)


### 2. Sharpening Filters
</br>

i) Using Laplacian Kernal
![alt text](<Screenshot 2024-03-08 171519.png>)

ii) Using Laplacian Operator
![alt text](<Screenshot 2024-03-08 171530.png>)

## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
