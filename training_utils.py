import torch
from tqdm import tqdm
# from tqdm.notebook import tqdm_notebook
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, classification_report


# def per_class_accuracy(labels, predictions):
#     """
#     Calculates the per-class accuracy given two arrays of labels and predictions
#     """
#
#     labels = labels.tolist()
#     predictions = predictions.tolist()
#
#     class_report = classification_report(labels, predictions, output_dict=True, zero_division=1)
#     accuracy_per_class = {}
#     for class_name in class_report.keys():
#         if class_name.isdigit():
#             class_id = int(class_name)
#             # accuracy_per_class[class_id] = class_report[class_name]['recall']
#             accuracy_per_class[class_id] = class_report[class_name]['precision']
#             # print(class_report[class_name])
#     return np.array(list(accuracy_per_class.values()), dtype='float64')


def get_per_cls_imgs(labels, num_cls):

    labels = labels.tolist()

    per_cls_imgs = np.zeros(num_cls)

    for i in range(num_cls):
        per_cls_imgs[i] = labels.count(int(i))

    # print(f'Labels - {labels}')
    # print(f'per_cls_imgs - {per_cls_imgs}')

    return np.array(per_cls_imgs)


def get_per_cls_correct(labels, predictions, num_cls):
    """
    Calculates number of images correctly classified per class count
    """

    labels = labels.tolist()
    predictions = predictions.tolist()

    per_cls_correct = np.zeros(num_cls)

    batch_size = len(labels)

    for i in range(batch_size):
        class_id = labels[i]

        if labels[i] == predictions[i]:
            per_cls_correct[class_id] += 1
        else:
            pass

    # print(f'Predictions - {predictions}')
    # print(f'per_cls_correct - {per_cls_correct}')

    return np.array(per_cls_correct)


# def get_per_cls_acc(preds, labels, num_cls):
#
#     print(f'Preds - {preds.tolist()}')
#     print(f'Labels - {labels.tolist()}')
#
#     preds = np.array(preds)
#     labels = np.array(labels)
#
#     num_images = len(labels)
#
#     assert len(labels) == len(preds), 'Number of labels and predictions not matching !'
#
#     per_cls_acc = np.zeros(num_cls)
#     for i in range(num_cls):
#         per_cls_acc[i] =
#
#     return per_cls_acc


# Training function.
def train(model, trainloader, optimizer, criterion, device, num_cls):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    per_cls_correct = np.zeros(num_cls, dtype='float64')
    per_cls_imgs = np.zeros(num_cls, dtype='float64')
    counter = 0

    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)

        # # Added softmax layer before computing loss
        # outputs = F.softmax(outputs, dim=1)

        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()

        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()

        per_cls_correct += get_per_cls_correct(labels, preds, num_cls)
        per_cls_imgs += get_per_cls_imgs(labels, num_cls)

    # print(f'per_cls_correct - {per_cls_correct}')
    # print(f'per_cls_imgs - {per_cls_imgs}')

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    # epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    per_cls_acc = per_cls_correct * 100 / per_cls_imgs

    return epoch_loss, epoch_acc, model, per_cls_acc


# Validation function.
def validate(model, testloader, criterion, device, num_cls):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    per_cls_correct = np.zeros(num_cls, dtype='float64')
    per_cls_imgs = np.zeros(num_cls, dtype='float64')
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data

            # print(f'Image shape - {image.shape}')
            # print(f'Image type - {type(image)}')
            # print(f'Labels - {labels}')
            # print(f'Labels shape - {labels.shape}')

            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

            per_cls_correct += get_per_cls_correct(labels, preds, num_cls)
            per_cls_imgs += get_per_cls_imgs(labels, num_cls)

    # print(f'per_cls_correct - {per_cls_correct}')
    # print(f'per_cls_imgs - {per_cls_imgs}')

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))

    per_cls_acc = per_cls_correct * 100 / per_cls_imgs

    return epoch_loss, epoch_acc, model, per_cls_acc
