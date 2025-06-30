from CIFAR10_Net_Determination_With_Augmentation import *
from Net_Models import *

# model training with augmentation

if __name__ == "__main__":
    print("\nProgram data\n\ny_train size:", len(Train_set), "\ny_test size:", len(Test_set))

    print("\nThe classes (images) considered in the task:", Train_set.classes)

    # model = My_CIFAR10_Net_Second()
    model = My_CIFAR10_Net_Third()
    print("\nThe learning process...")

    # applying the learning function to the selected model
    (train_accuracy_history, train_loss_history, test_accuracy_history, test_loss_history,
     correct_predictions, incorrect_predictions) = train_with_aug(
        net=model,
        Train_loader=Train_loader,
        Test_loader=Test_loader,
        epochs=100,
        lr=0.1,
        weight_decay=5e-4
    )

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

