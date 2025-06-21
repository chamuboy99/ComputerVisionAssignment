import cv2
import numpy as np
import matplotlib.pyplot as plt

def block_average(my_img, block_size):
    height, width = my_img.shape
    h_trim, w_trim = height - height % block_size, width - width % block_size
    img_cropped = my_img[:h_trim, :w_trim]
    reshaped = img_cropped.reshape(h_trim // block_size, block_size,
                                   w_trim // block_size, block_size)
    block_avg = reshaped.mean(axis=(1, 3)).astype(np.uint8)
    return np.kron(block_avg, np.ones((block_size, block_size), dtype=np.uint8))

img = cv2.imread('testing.jpg', cv2.IMREAD_GRAYSCALE)

result_3 = block_average(img, 3)
result_5 = block_average(img, 5)
result_7 = block_average(img, 7)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray'), plt.title("Original"), plt.axis('off')
plt.subplot(2, 2, 2), plt.imshow(result_3, cmap='gray'), plt.title("3×3 Block Averaging"), plt.axis('off')
plt.subplot(2, 2, 3), plt.imshow(result_5, cmap='gray'), plt.title("5×5 Block Averaging"), plt.axis('off')
plt.subplot(2, 2, 4), plt.imshow(result_7, cmap='gray'), plt.title("7×7 Block Averaging"), plt.axis('off')
plt.tight_layout()
plt.show()
