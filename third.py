import cv2
import numpy as np
from matplotlib import pyplot as plt

my_img = cv2.imread('testing.jpg')
(h, w) = my_img.shape[:2]

center = (w // 2, h // 2)  # Image center

rotation_matrix_45 = cv2.getRotationMatrix2D(center, 45, 1.0)

cos = np.abs(rotation_matrix_45[0, 0])
sin = np.abs(rotation_matrix_45[0, 1])
new_width = int((h * sin) + (w * cos))
new_height = int((h * cos) + (w * sin))

rotation_matrix_45[0, 2] += (new_width / 2) - center[0]
rotation_matrix_45[1, 2] += (new_height / 2) - center[1]

rotated_45 = cv2.warpAffine(my_img, rotation_matrix_45, (new_width, new_height))

rotated_90 = cv2.rotate(my_img, cv2.ROTATE_90_CLOCKWISE)

image_rgb = cv2.cvtColor(my_img, cv2.COLOR_BGR2RGB)
rotated_45_rgb = cv2.cvtColor(rotated_45, cv2.COLOR_BGR2RGB)
rotated_90_rgb = cv2.cvtColor(rotated_90, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 6))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Rotated 45°")
plt.imshow(rotated_45_rgb)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Rotated 90°")
plt.imshow(rotated_90_rgb)
plt.axis('off')

plt.tight_layout()
plt.show()
