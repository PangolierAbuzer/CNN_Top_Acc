import torch
import random
import numpy as np
import torchvision.datasets
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler


# loading the dataset into the train and test sets
CIFAR10_train = torchvision.datasets.CIFAR10('./', download=True, train=True)
CIFAR10_test = torchvision.datasets.CIFAR10('./', download=True, train=False)

# distribution of features and classes for each of the samples
X_train = torch.FloatTensor(CIFAR10_train.data)
y_train = torch.LongTensor(CIFAR10_train.targets)  # to be compatible with cross-entropy
X_test = torch.FloatTensor(CIFAR10_test.data)
y_test = torch.LongTensor(CIFAR10_test.targets)


"""
creating seeds for testing training on other devices or comparing it with other neural networks,
as well as transferring calculations to the GPU (video card)
"""
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


# a function for drawing ten random images from the training set
def print_images():
    print('\n')
    plt.figure(figsize=(10, 2))

    num_images = len(X_train)
    random_indices = random.sample(range(num_images), 10)

    for i in range(10):
        random_index = random_indices[i]
        plt.subplot(1, 10, i + 1)
        plt.imshow(X_train[random_index].numpy())
        plt.title(CIFAR10_train.classes[y_train[random_index]])
        # plt.axis('off')
        print(y_train[random_index], end=' ')

    plt.tight_layout()
    plt.show()
    print('\n')


# a function for displaying objects (to understand how it should look)
def print_obj(obj, m):
    print('\n')
    k = 0
    for i in obj:
        if k == m:
            break
        print(i, '\n')
        k += 1


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


# network training function
"""
the training function takes as input a neural network model, training and test samples, the number of epochs, 
the learning step, and the regularization coefficient of the network
"""
def train(net, X_train, y_train, X_test, y_test, epochs, lr, weight_decay=0.0):

    # selecting a computing device (graphics card or processor) and transferring the network model to the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    # choosing the loss function
    loss = torch.nn.CrossEntropyLoss()

    # choosing an optimizer
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr) - used it before
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    # enabling the smart step
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)

    # batch processing
    batch_size = 128

    # storing metrics
    train_accuracy_history = []
    train_loss_history = []
    test_accuracy_history = []
    test_loss_history = []
    correct_predictions = []
    incorrect_predictions = []

    # a variable for storing data about the best epoch in terms of learning efficiency
    best_data = []
    # a variable for finding the minimum error on the test
    test_loss_min = 10

    # connecting the test and training samples to the device
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # for each epoch, we perform
    for epoch in range(epochs):
        # we shuffle the training data indices at the beginning of each epoch
        order = np.random.permutation(len(X_train))

        # auxiliary variables for working with metrics
        total_train_loss_this_epoch = 0.0
        num_batches_this_epoch = 0
        total_correct_train_preds_this_epoch = 0
        total_train_samples_this_epoch = 0

        for start_index in range(0, len(X_train), batch_size):
            # we zero the gradients of the model parameters
            optimizer.zero_grad()

            # we tell the program that the network is in the process of learning
            net.train()

            # select the indices of the current batch from the mixed order
            batch_indexes = order[start_index:start_index + batch_size]

            # connect batch features and classes to the device
            X_batch = X_train[batch_indexes].to(device)
            y_batch = y_train[batch_indexes].to(device)

            # performing a forward pass:
            # we feed the batch of data to the model's input to obtain predictions
            train_preds = net.forward(X_batch)

            # calculate the loss function value for the current batch
            loss_value = loss(train_preds, y_batch)

            # counting common errors
            total_train_loss_this_epoch += loss_value.item()
            num_batches_this_epoch += 1

            # calculation of correct predictions and the number of examples in the current batch
            predicted_labels = train_preds.argmax(dim=1)
            correct_in_batch = (predicted_labels == y_batch).sum()
            samples_in_batch = y_batch.size(0)

            # accumulation of total values over an epoch
            total_correct_train_preds_this_epoch += correct_in_batch.item()
            total_train_samples_this_epoch += samples_in_batch

            # performing a backward pass:
            # we calculate the gradients of the loss function with respect to all model parameters
            loss_value.backward()

            # performing the optimization step
            # we perform the step of updating the model parameters based on the calculated optimization gradients
            optimizer.step()

        # calculation of the total error on the training set per epoch
        average_train_loss_this_epoch = total_train_loss_this_epoch / num_batches_this_epoch
        train_loss_history.append(average_train_loss_this_epoch)

        # calculating the overall accuracy on the training set per epoch
        overall_train_accuracy_this_epoch = (
                (total_correct_train_preds_this_epoch / total_train_samples_this_epoch) * 100)
        train_accuracy_history.append(overall_train_accuracy_this_epoch)

        # we tell the program that the network is now in testing mode
        net.eval()

        with torch.no_grad():
            # we perform a forward pass on the entire test set to obtain predictions
            test_preds = net.forward(X_test)

            # we calculate the value of the loss function on the test set
            test_loss = loss(test_preds, y_test).data.cpu().item()
            test_loss_history.append(test_loss)

            # we calculate the accuracy on the test set
            test_accuracy = (test_preds.argmax(dim=1) == y_test).float().mean().item() * 100

            # we keep the accuracy value
            test_accuracy_history.append(test_accuracy)

            # getting predictions for rendering images
            predicted_labels = test_preds.argmax(dim=1)

            # for the last epoch, we select images
            if epoch == epochs - 1:

                # comparing predicted and actual labels for saving
                for i in range(len(y_test)):
                    image = X_test[i].cpu()
                    true_label = y_test[i].cpu().item()
                    predicted_label = predicted_labels[i].cpu().item()

                    if predicted_label == true_label:
                        correct_predictions.append((image, true_label, predicted_label))
                    else:
                        incorrect_predictions.append((image, true_label, predicted_label))

            # finding the minimum error
            if test_loss < test_loss_min:
                test_loss_min = test_loss
                best_data = (epoch + 1, test_accuracy, test_loss)

        # transmitting the step accuracy data
        scheduler.step(test_accuracy)


        # data output
        print(f"Epoch № {epoch + 1}/{epochs} -> "
              f"Train Acc: {overall_train_accuracy_this_epoch:.2f}%, Train Loss: {average_train_loss_this_epoch:.4f} | "
              f"Test Acc: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}")

        # manual step change
        """
        if epoch + 1 == 6: 
            new_lr = lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        """

    print(f"The best condition based on the loss on the test -> Epoch: {best_data[0]}/{epochs},"
          f" Accuracy:  {best_data[1]:.2f}%, Loss: {best_data[2]:.4f}")

    print('Training completed')

    # returning the metrics
    return (train_accuracy_history, train_loss_history, test_accuracy_history, test_loss_history,
            correct_predictions, incorrect_predictions)


















































































