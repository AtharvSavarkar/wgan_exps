import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader
from torch.utils import data
from torchvision import datasets
from torchvision import transforms

plt.style.use('ggplot')


def get_data(train_folder_path, test_folder_path, batch_size=64):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # dataset_train, dataset_valid = data.random_split(dataset_class, train_test_split)
    dataset_train = datasets.ImageFolder(train_folder_path, transform=transform)
    dataset_valid = datasets.ImageFolder(test_folder_path, transform=transform)

    print(f'Classes - {dataset_train.class_to_idx} \n\n')

    # Create data loaders.
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset_valid,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, valid_loader


def save_plots(train_acc, valid_acc, train_loss, valid_loss, name=None):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='tab:blue', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='tab:red', linestyle='-',
        label='validation accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join('outputs', name + '_accuracy.png'))

    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-',
        label='validation loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join('outputs', name + '_loss.png'))
