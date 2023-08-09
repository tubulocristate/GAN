import torch
import config
import matplotlib.pyplot as plt

def make_square():
    blank = torch.zeros(size=(config.WIDTH, config.HEIGHT))
    storona = 20
    min_row = int(config.HEIGHT/2 - storona)
    max_row = int(config.HEIGHT/2 + storona)
    min_col = int(config.WIDTH/2 - storona)
    max_col = int(config.WIDTH/2 + storona)

    for row in range(min_row, max_row):
        for col in range(min_col, max_col):
            blank[row][col] = 1.
    return blank

def make_circle():
    blank = torch.zeros(size=(config.WIDTH, config.HEIGHT))
    radius = 10;

    for x in range(-radius, radius):
        for y in range(-radius, radius):
            if x*x + y*y <= radius*radius:
                blank[x+32][y+32] = 1.
    return blank

def make_sinusoidal_image():
    blank = torch.zeros(size=(config.WIDTH, config.HEIGHT))
    value = 0
    increment = 2*3.1415/(config.HEIGHT*config.WIDTH)
    for row in range(config.HEIGHT):
        for col in range(config.WIDTH):
            blank[row][col] = value
            value += increment
    return torch.sin(blank)


def display_image(image):
    plt.imshow(image, cmap="Greys_r")
    plt.show()


