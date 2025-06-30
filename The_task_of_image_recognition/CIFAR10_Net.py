from CIFAR10_Net_Determination import *
from Net_Models import *

# model training without augmentation

if __name__ == "__main__":
    print("\nProgram data\n\ny_train size:", len(y_train), "\ny_test size:", len(y_test))

    """
    function for displaying object elements
    (the input is the object being considered and the number of its elements to be displayed)
    """
    # print_obj(X_test, 2)

    print("\nMinimum and maximum value for the X_train sample" 
          "(obviously, the same applies to X_test):", X_train.min(), X_train.max())

    # compress the features for convenience by finding the minimum and maximum values
    # now the X_train and X_test elements store probability values (from 0 to 1)
    X_train /= 255.
    X_test /= 255.
    # we can check it out
    # print_obj(X_test, 2)

    print("\nThe classes (images) considered in the task:", CIFAR10_train.classes)

    """
    if we consider the above in the tensor structure, it will be understood as follows: the sample is represented by 
    50,000 objects. Each object stores 32 two-dimensional arrays. Each two-dimensional array stores 32 arrays. 
    Each array stores the color intensity of a single pixel in the FloatTensor format (32-bit floating-point number).
    The first two-dimensional array contains 32 pixels horizontally, and the next array goes down, 
    and so on until the last array. The result is a size of 32x32 pixels. The color intensity is always between 
    0 and 255. In such cases, it is recommended to narrow the feature boundary to 1, which allows for normalization 
    and faster training, which I have done.
    y_train - mapping the class to each of the images in the training set
    """
    print("\nDimensions of the training sample:", X_train.shape, y_train.shape)


    # let's put the image component in the 2nd position in the dimension for working with matplotlib
    X_train = X_train.permute(0, 3, 1, 2)
    X_test = X_test.permute(0, 3, 1, 2)
    print("\nDimensions of the training sample after permute:", X_train.shape, y_train.shape)

    # drawing images
    # print_images()

    print("\nThe learning process...")
    # model = Stepik_CIFAR10_Net()
    # model = My_CIFAR10_Net_First()
    model = My_CIFAR10_Net_Second()
    # model = My_CIFAR10_Net_Third()
    # model = models.resnet18(weights=None)

    # applying the learning function to the selected model
    (train_accuracy_history, train_loss_history, test_accuracy_history, test_loss_history,
     correct_predictions, incorrect_predictions) = \
        train(model, X_train, y_train, X_test, y_test, 25, 0.1, 5e-4)

    # drawing metrics
    plot_history(train_accuracy_history, "The history of accuracy during training",
                 "Accuracy (%)", "cyan")
    plot_history(train_loss_history, "The history of loss during training",
                 "Loss (CrossEntropyLoss)", "pink")
    plot_history(test_accuracy_history, "Accuracy history during test execution",
                 "Accuracy (%)", "green")
    plot_history(test_loss_history, "Loss history during test execution",
                 "Loss (CrossEntropyLoss)", "red")

    # the function of visualizing some of the results
    # (10 images with correct predictions and 10 with incorrect predictions)
    visualize_predictions(correct_predictions, incorrect_predictions, CIFAR10_train.classes, 10)




             


