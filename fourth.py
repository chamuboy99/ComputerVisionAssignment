import cv2
import numpy as np
from matplotlib import pyplot as plt

my_img = cv2.imread('testing.jpg', 0)

def block_average(img, block_size):
    height, width = img.shape
    output = img.copy()
    
    for i in range(0, height - block_size + 1, block_size):
        for j in range(0, width - block_size + 1, block_size):
            block = img[i:i+block_size, j:j+block_size]
            avg_value = np.mean(block, dtype=np.uint8)
            output[i:i+block_size, j:j+block_size] = avg_value
    return output

avg_3 = block_average(my_img, 3)
avg_5 = block_average(my_img, 5)
avg_7 = block_average(my_img, 7)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Original")
plt.imshow(my_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("3x3 Block Avg")
plt.imshow(avg_3, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("5x5 Block Avg")
plt.imshow(avg_5, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("7x7 Block Avg")
plt.imshow(avg_7, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
