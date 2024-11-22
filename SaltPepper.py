import cv2
import copy
from random import randint

def SaltPepper(img):
    # Getting the dimensions of the image
    if img.ndim > 2:  # color
        height, width, _ = img.shape
    else:  # gray scale
        height, width = img.shape

    result = copy.deepcopy(img)

    # Randomly pick some pixels in the image
    # Pick a random number between height*width/80 and height*width/10
    number_of_pixels = randint(int(height * width / 100), int(height * width / 10))

    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = randint(0, height - 1)

        # Pick a random x coordinate
        x_coord = randint(0, width - 1)

        if result.ndim > 2:
            result[y_coord][x_coord] = [randint(0, 255), randint(0, 255), randint(0, 255)]
        else:
            # Color that pixel to white
            result[y_coord][x_coord] = 255

    # Randomly pick some pixels in image
    # Pick a random number between height*width/80 and height*width/10
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = randint(0, height - 1)

        # Pick a random x coordinate
        x_coord = randint(0, width - 1)

        if result.ndim > 2:
            result[y_coord][x_coord] = [randint(0, 255), randint(0, 255), randint(0, 255)]
        else:
            # Color that pixel to white
            result[y_coord][x_coord] = 0

    return result
