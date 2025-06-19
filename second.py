import cv2
from matplotlib import pyplot as plt

my_img = cv2.imread('testing.jpg')

avg_3x3 = cv2.blur(my_img, (3, 3))
avg_10x10 = cv2.blur(my_img, (10, 10))
avg_20x20 = cv2.blur(my_img, (20, 20))

my_img_rgb = cv2.cvtColor(my_img, cv2.COLOR_BGR2RGB)
avg_3x3_rgb = cv2.cvtColor(avg_3x3, cv2.COLOR_BGR2RGB)
avg_10x10_rgb = cv2.cvtColor(avg_10x10, cv2.COLOR_BGR2RGB)
avg_20x20_rgb = cv2.cvtColor(avg_20x20, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title('Original')
plt.imshow(my_img_rgb)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('3x3 Average')
plt.imshow(avg_3x3_rgb)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('10x10 Average')
plt.imshow(avg_10x10_rgb)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('20x20 Average')
plt.imshow(avg_20x20_rgb)
plt.axis('off')

plt.tight_layout()
plt.show()
