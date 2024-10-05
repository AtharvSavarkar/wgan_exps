import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

data = np.load('DermaMNIST.npz')

try:
    os.mkdir('total_data')
except FileExistsError:
    pass

try:
    os.mkdir('total_data/mel')
except FileExistsError:
    pass

try:
    os.mkdir('total_data/nv')
except FileExistsError:
    pass

try:
    os.mkdir('lake')
except FileExistsError:
    pass

train_images = data['train_images']
train_labels = data['train_labels']

for i in tqdm(range(len(train_labels))):

    img = train_images[i]

    if train_labels[i] == 0:
        plt.imsave(f'total_data/mel/mel_{i}.jpg', img)
        plt.imsave(f'lake/mel_{i}.jpg', img)
    else:
        plt.imsave(f'total_data/nv/nv_{i}.jpg', img)
        plt.imsave(f'lake/nv_{i}.jpg', img)

print('Done !')
