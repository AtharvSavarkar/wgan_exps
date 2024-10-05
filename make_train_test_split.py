import os
import shutil
from tqdm import tqdm
import numpy as np

train_test_split = [0.7, 0.3]
data_folder_path = 'total_data'

assert np.sum(train_test_split) == 1, 'Sum of fractions in train_test_split should be equal to 1'
classes = os.listdir(data_folder_path)
num_cls = len(classes)

try:
    os.mkdir('test_data')
except FileExistsError:
    pass

try:
    os.mkdir('train_data')
except FileExistsError:
    pass

# Making folders
for i in range(num_cls):
    try:
        os.mkdir(f'train_data/{classes[i]}')
    except FileExistsError:
        pass

    try:
        os.mkdir(f'test_data/{classes[i]}')
    except FileExistsError:
        pass


num_imgs_per_class = []
num_train_per_class = []
num_test_per_class = []
for i in range(num_cls):

    num_imgs_per_class.append(len(os.listdir(os.path.join(data_folder_path, classes[i]))))
    num_train_per_class.append(int(num_imgs_per_class[i] * train_test_split[0]))
    num_test_per_class.append(int(num_imgs_per_class[i] - num_train_per_class[i]))

print(f'Number of images per class - {num_imgs_per_class}')
print(f'Number of train images per class - {num_train_per_class}')
print(f'Number of test images per class - {num_test_per_class}')

print('Copying files ...')
for i in range(num_cls):
    for j in tqdm(range(num_imgs_per_class[i])):

        images = os.listdir(os.path.join(data_folder_path, classes[i]))

        img_source = os.path.join(data_folder_path, classes[i], images[j])

        if j < num_train_per_class[i]:
            img_destination = os.path.join('train_data', classes[i], images[j])
        else:
            img_destination = os.path.join('test_data', classes[i], images[j])

        shutil.copy(img_source, img_destination)

print('Done !')
