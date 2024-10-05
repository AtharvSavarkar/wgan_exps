import os
import shutil
from tqdm import tqdm


def combine_folders(folder1_path, folder2_path, final_folder_path):
    try:
        shutil.rmtree(final_folder_path)
    except FileNotFoundError:
        pass

    assert not os.path.exists(final_folder_path), 'Delete final_folder_path before running this code'

    assert len(os.listdir(folder1_path)) == len(
        os.listdir(folder2_path)), 'Number of classes not same in folder1 and folder2'
    assert os.listdir(folder1_path) == os.listdir(folder2_path), 'Classes not same in folder1 and folder2'

    classes = os.listdir(folder1_path)
    num_cls = len(classes)

    try:
        os.mkdir(final_folder_path)
    except FileExistsError:
        pass

    for i in range(num_cls):
        try:
            os.mkdir(os.path.join(final_folder_path, classes[i]))
        except FileExistsError:
            pass

    per_cls_imgs_1 = []
    per_cls_imgs_2 = []

    for i in range(num_cls):
        per_cls_imgs_1.append(len(os.listdir(os.path.join(folder1_path, classes[i]))))
        per_cls_imgs_2.append(len(os.listdir(os.path.join(folder2_path, classes[i]))))

    print(f'Per cls imgs for folder1 - {per_cls_imgs_1}')
    print(f'Per cls imgs for folder2 - {per_cls_imgs_2}')

    for i in range(num_cls):

        f1_class_imgs = os.listdir(os.path.join(folder1_path, classes[i]))

        for j in tqdm(range(len(f1_class_imgs))):
            img_source = os.path.join(folder1_path, classes[i], f1_class_imgs[j])
            img_dest = os.path.join(final_folder_path, classes[i], f1_class_imgs[j])

            shutil.copy(img_source, img_dest)

        f2_class_imgs = os.listdir(os.path.join(folder2_path, classes[i]))

        for j in tqdm(range(len(f2_class_imgs))):
            img_source = os.path.join(folder2_path, classes[i], f2_class_imgs[j])
            img_dest = os.path.join(final_folder_path, classes[i], f2_class_imgs[j])

            shutil.copy(img_source, img_dest)

    print('Done !')
