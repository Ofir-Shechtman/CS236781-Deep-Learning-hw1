import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss
import cs236781.dataloader_utils as dataloader_utils


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = torch.distributions.normal.Normal(0, weight_std).rsample((n_features, n_classes))

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:T
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        class_scores = torch.matmul(x, self.weights)
        _, y_pred = torch.max(class_scores, dim=1)
        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Use the predict function above and compare the predicted class
        #  labels to the ground truth labels to obtain the accuracy (in %).
        #  Do not use an explicit loop.

        acc = torch.sum(y == y_pred).item() / y.shape[0]
        return acc * 100

    @staticmethod
    def _epoch(self,
               dl: DataLoader,
               loss_fn: ClassifierLoss,
               weight_decay,
               to_update):
        x, y = dataloader_utils.flatten(dl)
        y_pred, x_scores = self.predict(x)
        loss = loss_fn(x, y, x_scores, y_pred, to_update) + weight_decay / 2 * self.weights.norm() ** 2
        total_correct = self.evaluate_accuracy(y, y_pred)
        return total_correct, loss

    def train(
            self,
            dl_train: DataLoader,
            dl_valid: DataLoader,
            loss_fn: ClassifierLoss,
            learn_rate=0.1,
            weight_decay=0.001,
            max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            train_accuracy, train_loss = self._epoch(self, dl_train, loss_fn, weight_decay, True)
            valid_accuracy, valid_loss = self._epoch(self, dl_valid, loss_fn, weight_decay, False)
            train_res.accuracy.append(train_accuracy)
            train_res.loss.append(train_loss)
            valid_res.accuracy.append(valid_accuracy)
            valid_res.loss.append(valid_loss)

            grad = loss_fn.grad()
            self.weights = self.weights - learn_rate * grad

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).
        return self.weights[1:].T.reshape(-1, *img_shape)


def hyperparams():
    hp = dict(weight_std=0.001, learn_rate=0.1, weight_decay=0.01)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.

    return hp
