import cv2
from matplotlib import pyplot as plt

def reduce_intensity(image_path, levels):
    my_img = cv2.imread(image_path, 0)
    if (levels & (levels - 1)) != 0 or levels <= 1:
        raise ValueError("Number of levels must be a power of 2 and greater than 1")
    factor = 256 // levels
    reduced_img = (my_img // factor) * factor + factor // 2

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title('Original Image')
    plt.imshow(my_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title(f'Image with {levels} intensity levels')
    plt.imshow(reduced_img, cmap='gray')
    plt.axis('off')

    plt.show()

levels = int(input("Enter number of intensity levels (power of 2): "))
reduce_intensity('testing.jpg', levels)