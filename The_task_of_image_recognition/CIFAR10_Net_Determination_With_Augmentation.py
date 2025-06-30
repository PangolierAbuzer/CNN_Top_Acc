import torch
import random
import numpy as np
import torchvision.datasets
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms  # for augmentation


# for the visualization function
CIFAR10_train = torchvision.datasets.CIFAR10('./', download=True, train=True)

# implementing augmentation in the training dataset
transform_Train = transforms.Compose([
    #  random cropping with added padding
    transforms.RandomCrop(32, padding=4),
    # random horizontal reflection
    transforms.RandomHorizontalFlip(),
    # transformation to a tensor
    transforms.ToTensor(),
    # normalization to the range [-1, 1])
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# transformations for the test (only tensor formatting and data normalization)
transform_Test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# loading the CIFAR-10 dataset using transformations
Train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_Train)
Test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_Test)

# distribution of the new set by batch
batch_size = 128
Train_loader = torch.utils.data.DataLoader(Train_set, batch_size=batch_size, shuffle=True, num_workers=2)
Test_loader = torch.utils.data.DataLoader(Test_set, batch_size=100, shuffle=False, num_workers=2)


"""
creating seeds for testing training on other devices or comparing it with other neural networks,
as well as transferring calculations to the GPU (video card)
"""
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


# a function for drawing graphs for analytics
def plot_history(list_history, text_title, ord_y_text_lable, color):

    epochs = range(1, len(list_history) + 1)

    plt.figure(figsize=(5, 5))
    plt.plot(epochs, list_history, marker='o', linestyle='-', color=color, markerfacecolor="purple")

    plt.title(text_title, size=20)
    plt.xlabel('Epoch', size=15)
    plt.ylabel(ord_y_text_lable, size=15)
    plt.grid(True)

    plt.xticks(epochs)
    plt.tight_layout()
    plt.show()


# a function for rendering images with neural network predictions
def visualize_predictions(correct_predictions, incorrect_predictions, class_names, num_to_show):

    def show_images(predictions, title):

        plt.figure(figsize=(12, 4))
        plt.suptitle(title, fontsize=16)

        for i in range(min(num_to_show, len(predictions))):

            image, true_label, predicted_label = predictions[i]
            plt.subplot(1, num_to_show, i + 1)

            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).numpy()

            plt.imshow(image)
            plt.title(f"True: {class_names[true_label]}\nPred: {class_names[predicted_label]}")
            plt.axis('off')
        plt.show()

    if correct_predictions:
        show_images(correct_predictions, "Correct predictions")
    else:
        print("There are no such images")

    if incorrect_predictions:
        show_images(incorrect_predictions, "Incorrect predictions")
    else:
        print("There are no such images")


# network training function with augmentation
# it's similar to the training function without augmentation, but the data format changes
# I won't comment much
def train_with_aug(net, Train_loader, Test_loader, epochs, lr, weight_decay=0.0):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)

    train_accuracy_history = []
    train_loss_history = []
    test_accuracy_history = []
    test_loss_history = []
    correct_predictions = []
    incorrect_predictions = []

    best_data = []
    test_loss_min = 10

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, data in enumerate(Train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # forward pass
            outputs = net(inputs)
            loss_value = criterion(outputs, labels)

            # backward pass
            loss_value.backward()
            # optimization
            optimizer.step()

            # metrics
            running_loss += loss_value.item()
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        epoch_train_accuracy = 100 * correct_train / total_train
        epoch_train_loss = running_loss / len(Train_loader)

        train_accuracy_history.append(epoch_train_accuracy)
        train_loss_history.append(epoch_train_loss)


        net.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0.0

        with torch.no_grad():
            for images, labels in Test_loader:
                images, labels = images.to(device), labels.to(device)

                # forward pass
                outputs = net(images)
                loss_value_test = criterion(outputs, labels)

                test_loss += loss_value_test.item()

                _, predicted_test = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted_test == labels).sum().item()

                if epoch == epochs - 1:
                    for i in range(len(labels)):
                        image = images[i].cpu()
                        true_label = labels[i].cpu().item()
                        predicted_label = predicted_test[i].cpu().item()

                        if predicted_label == true_label:
                            correct_predictions.append((image, true_label, predicted_label))
                        else:
                            incorrect_predictions.append((image, true_label, predicted_label))

        # calculating metrics on a test sample
        accuracy_test = 100 * correct_test / total_test
        avg_test_loss = test_loss / len(Test_loader)

        test_accuracy_history.append(accuracy_test)
        test_loss_history.append(avg_test_loss)

        if avg_test_loss < test_loss_min:
            test_loss_min = avg_test_loss
            best_data = (epoch + 1, accuracy_test, avg_test_loss)

        # step
        scheduler.step(accuracy_test)

        # output of results
        print(f"Epoch № {epoch + 1}/{epochs} -> "
              f"Train Acc: {epoch_train_accuracy:.2f}%, Train Loss: {epoch_train_loss:.4f} | "
              f"Test Acc: {accuracy_test:.2f}%, Test Loss: {avg_test_loss:.4f}")

    print(f"The best condition based on the loss on the test -> Epoch: {best_data[0]}/{epochs},"
          f" Accuracy:  {best_data[1]:.2f}%, Loss: {best_data[2]:.4f}")

    print('Training completed')

    return (train_accuracy_history, train_loss_history, test_accuracy_history, test_loss_history,
            correct_predictions, incorrect_predictions)

