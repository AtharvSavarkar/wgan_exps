import json
import os
import subprocess
import time

import torch
import torch.optim as optim

from resnet18 import ResNet, BasicBlock
from torchvision.models import mobilenet_v2
from training_utils import train, validate
from utils import save_plots, get_data
from combine_train_folders import *
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Provide path to config file for setting up experiment')

args = parser.parse_args()

config_file_path = args.config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# STEP 1 - Parsing dataset from .npz file to folders
print(f'\n\n Parsing dataset from .npz file to folders ...')
from parse_data_from_npz import *

# STEP 2 - Splitting data into train and test split
print(f'\n\n Splitting data into train and test split ...')
from make_train_test_split import *

# STEP 3 - Reading exp_config.json file for setting up experiment
print(f'\n\n Setting up experiment ...')

with open(config_file_path, 'r') as file:
    exp_config = json.loads((file.read()))

dataset_npz_path = exp_config["dataset_npz_path"]
exp_name = exp_config["exp_name"]
subset_selection_algo = exp_config["subset_selection_algo"]
wgan_epochs = exp_config["wgan_epochs"]
wgan_img_size = exp_config["wgan_img_size"]
cnn_epochs = exp_config["cnn_epochs"]
cnn_batch_size = exp_config["cnn_batch_size"]
cnn_lr = exp_config["cnn_lr"]
optimizer_weight_decay = exp_config["optimizer_weight_decay"]
cnn_optimizer = exp_config["cnn_optimizer"]
wgan_budget = exp_config["budget_list"]
feature_extraction_model = exp_config["feature_extraction_model"]
train_test_model = exp_config["train_test_model"]
subset_selection_budget = exp_config["subset_selection_budget"]

# STEP 4 - Feature Extraction
if feature_extraction_model == "resnet18":
    print("Using ResNet18 model for feature extraction")
    from feature_extractor_batch_resnet18 import *
elif feature_extraction_model == "mv2":
    print("Using MV2 model for feature extraction")
    from feature_extractor_batch_mv2 import *
else:
    print('Warning - Using default feature extraction model - ResNet18')
    from feature_extractor_batch_resnet18 import *


# STEP 5 - Subset Selection
print(f'\n\n Starting subset selection algorithm ...')
from sub_selection_and_split_data import *

select_subset_and_split(subset_selection_budget, subset_selection_algo)

# STEP 6 - Making directories for storing wGAN generated images
from make_dirs import *

for i in wgan_budget:
    make_dirs_for_wgan_generate(i, subset_selection_algo)

# STEP 7 - Training wGAN models
wgan_train_command_1 = ["python3", "main.py", "--dataset", "folder", "--dataroot",
                        f"gan_train_{subset_selection_algo}/mel", "--niter", f"{wgan_epochs + 5}",
                        "--experiment", f"samples_{subset_selection_algo}_mel", "--imageSize", f"{wgan_img_size}",
                        "--cuda"]
wgan_train_command_2 = ["python3", "main.py", "--dataset", "folder", "--dataroot",
                        f"gan_train_{subset_selection_algo}/nv", "--niter", f"{wgan_epochs + 5}",
                        "--experiment", f"samples_{subset_selection_algo}_nv", "--imageSize", f"{wgan_img_size}",
                        "--cuda"]

subprocess.run(wgan_train_command_1)
subprocess.run(wgan_train_command_2)

for i in range(len(wgan_budget)):
    wgan_generate_command_1 = ["python3", "generate.py", "--config",
                               f"samples_{subset_selection_algo}_mel/generator_config.json",
                               "--weights", f"samples_{subset_selection_algo}_mel/netG_epoch_{wgan_epochs}.pth",
                               "--output_dir", f"generated_imgs_{subset_selection_algo}_{wgan_budget[i]}/mel",
                               "--nimages", f"{wgan_budget[i]}", "--cuda"]
    wgan_generate_command_2 = ["python3", "generate.py", "--config",
                               f"samples_{subset_selection_algo}_nv/generator_config.json",
                               "--weights", f"samples_{subset_selection_algo}_nv/netG_epoch_{wgan_epochs}.pth",
                               "--output_dir", f"generated_imgs_{subset_selection_algo}_{wgan_budget[i]}/nv",
                               "--nimages", f"{wgan_budget[i]}", "--cuda"]

    subprocess.run(wgan_generate_command_1)
    subprocess.run(wgan_generate_command_2)


# STEP 8 - Defining ResNet18 training functions

def train_resnet18(train_folder_path, test_folder_path, epochs, batch_size, learning_rate, weight_decay, optimizer_str):
    try:
        os.mkdir('resnet18_models')
    except FileExistsError:
        pass

    # try:
    #     os.mkdir('outputs')
    # except FileExistsError:
    #     pass

    # Set seed
    seed = 42
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    # np.random.seed(seed)
    # random.seed(seed)
    torch.random.manual_seed(42)
    num_classes = 2

    train_loader, valid_loader = get_data(train_folder_path, test_folder_path, batch_size=batch_size)

    # Define model based on the argument parser string.
    # if args['model'] == 'scratch':
    print('[INFO]: Training ResNet18 built from scratch...')
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=num_classes).to(device)
    plot_name = 'resnet_scratch'

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Optimizer.
    if optimizer_str == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_str == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_str == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        print('Optimizer not defined ...')
        exit()

    # Loss function.
    criterion = nn.CrossEntropyLoss()

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.

    max_val_acc = 0

    start_time = time.time()

    # Code will analyse last es_epochs to decide on early stopping of model
    es_epochs = 20

    # If max change in valid acc of last es_epochs fall below es_delta then training will stop
    es_delta = 0.01

    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_acc, model, per_cls_train_acc = train(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            num_cls=num_classes
        )
        valid_epoch_loss, valid_epoch_acc, model, per_cls_test_acc = validate(
            model,
            valid_loader,
            criterion,
            device,
            num_cls=num_classes
        )

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)

        if valid_epoch_acc > max_val_acc:
            print('Saving best.pth to working directory ... ')
            # print(f'Per class test acc for this model is - {per_cls_test_acc}')
            torch.save(model.state_dict(), 'cnn_models/best.pth')
            max_val_acc = valid_epoch_acc
        else:
            print('Saving last.pth to working directory ... ')
            torch.save(model.state_dict(), 'cnn_models/last.pth')

        # Early Stopping Code
        # if epoch < es_epochs:
        #     pass
        # else:
        #     # std_of_last_es_epochs = np.std(valid_acc[::-es_epochs])
        #     max_delta_of_last_es_epochs = np.round(max(valid_acc[::-es_epochs]) - min(valid_acc[::-es_epochs]), 2)
        #     print(f'Max delta of validation acc for last {es_epochs} epochs - {max_delta_of_last_es_epochs}')
        #
        #     if max_delta_of_last_es_epochs < es_delta and train_acc[-1] > 96:
        #         print(f'\n\nNo significant improvement in model found for last {es_epochs}')
        #         print('Early stopping model training ... \n\n')
        #         break
        #     else:
        #         pass

        # Change round off parameters to see exact accuracies (e.g. :.3f round off to 3 decimal places)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.2f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.2f}")
        # Printing per class accuracy after round off to 1 decimal place
        print(f'Per class test accuracy - {np.round_(per_cls_test_acc, decimals=2)}')
        print('-' * 50, '\n')

    end_time = time.time()

    print('\n\nSummary')
    print(f'Max training accuracy - {np.round(max(train_acc), 2)}')
    print(f'Max validation accuracy - {np.round(max(valid_acc), 2)}')
    print(f'Time required for training model - {np.round((end_time - start_time) / 60, 2)} \n\n')

    # Save the loss and accuracy plots.
    # save_plots(
    #     train_acc,
    #     valid_acc,
    #     train_loss,
    #     valid_loss,
    #     name=plot_name
    # )
    # print('TRAINING COMPLETE')

    return np.max(train_acc), np.max(valid_acc)


def train_mv2(train_folder_path, test_folder_path, epochs, batch_size, learning_rate, weight_decay, optimizer_str):

    train_loader, valid_loader = get_data(train_folder_path, test_folder_path, batch_size=batch_size)

    # Initialize the MobileNetV2 model
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, len(train_loader.dataset.classes))  # Adjusting for the number of classes
    model = model.to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer.
    if optimizer_str == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_str == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_str == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        print('Optimizer not defined ...')
        exit()

    # Lists to keep track of losses and accuracies.
    train_loss_array, valid_loss_array = [], []
    train_acc_array, valid_acc_array = [], []

    # Training and validation loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training phase
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Calculate average loss and accuracy for training
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train


        train_loss_array.append(train_loss)
        train_acc_array.append(train_accuracy)


        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader, desc=f"Validating Epoch {epoch+1}/{epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Calculate statistics
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        # Calculate average loss and accuracy for validation
        val_loss = running_val_loss / len(valid_loader)
        val_accuracy = correct_val / total_val

        valid_loss_array.append(val_loss)
        valid_acc_array.append(val_accuracy)

        # Print statistics for the epoch
        print(f"Epoch [{epoch+1}/{epochs}]: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f} \n")

    return np.max(train_acc_array), np.max(valid_acc_array)






wgan_budget = [0] + wgan_budget

print(wgan_budget)

training_accuracies = []
validation_accuracies = []
budget_till_now = []
for i in range(len(wgan_budget)):

    print(f'\n\n wGAN generated images - {wgan_budget[i]}')
    print('Training CNN ...')

    current_budget = wgan_budget[i]
    budget_till_now.append(current_budget)

    test_folder_path = 'test_data'

    if current_budget == 0:
        train_folder_path = f'train_fe_{subset_selection_algo}'
    else:
        combine_folders(f'train_fe_{subset_selection_algo}',
                        f'generated_imgs_{subset_selection_algo}_{wgan_budget[i]}',
                        'to_train')
        train_folder_path = 'to_train'


    # Training model as per config json file

    if train_test_model == "resnet18":
        print("Using ResNet18 model for training and testing")
        cnn_train_acc, cnn_valid_acc = train_resnet18(train_folder_path, test_folder_path, cnn_epochs,
                                                            cnn_batch_size, cnn_lr, optimizer_weight_decay,
                                                            cnn_optimizer)

    elif train_test_model == "mv2":
        print("Using MV2 model for training and testing")
        cnn_train_acc, cnn_valid_acc = train_mv2(train_folder_path, test_folder_path, cnn_epochs,
                                                            cnn_batch_size, cnn_lr, optimizer_weight_decay,
                                                            cnn_optimizer)

    else:
        print('Warning - Using default feature extraction model - ResNet18')
        print("Using ResNet18 model for training and testing")
        cnn_train_acc, cnn_valid_acc = train_resnet18(train_folder_path, test_folder_path, cnn_epochs,
                                                            cnn_batch_size, cnn_lr, optimizer_weight_decay,
                                                            cnn_optimizer)



    training_accuracies.append(cnn_train_acc)
    validation_accuracies.append(cnn_valid_acc)

    results_df = pd.DataFrame()
    results_df['budget'] = budget_till_now
    results_df['train_accs'] = training_accuracies
    results_df['test_accs'] = validation_accuracies


    results_df.to_excel(f'{exp_name}_{subset_selection_algo}.xlsx', index=False)


# STEP 9 - Deleting generated folders to clear for next experiment

files_generated = os.listdir()
for i in range(len(files_generated)):

    temp_file = files_generated[i]
    if os.path.isdir(temp_file) and temp_file != 'models' and temp_file != '.git':
        shutil.rmtree(temp_file)

# os.remove('img_features_lake.csv')

