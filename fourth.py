import cv2
import numpy as np
import matplotlib.pyplot as plt

my_img = cv2.imread('testing.jpg', cv2.IMREAD_GRAYSCALE)

height, width = my_img.shape

block_sizes = [3, 5, 7]

results = {}

for b in block_sizes:
    h_trim, w_trim = height - height % b, width - width % b
    cropped = my_img[:h_trim, :w_trim]
    reshaped = cropped.reshape(h_trim // b, b, w_trim // b, b)
    avg = reshaped.mean(axis=(1, 3)).astype(np.uint8)
    upscaled = np.kron(avg, np.ones((b, b), dtype=np.uint8))
    results[b] = upscaled

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.title("Original")
plt.imshow(my_img, cmap='gray')
plt.axis('off')

for i, b in enumerate(block_sizes, start=2):
    plt.subplot(2, 2, i)
    plt.title(f"{b}x{b} Block Average")
    plt.imshow(results[b], cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()
