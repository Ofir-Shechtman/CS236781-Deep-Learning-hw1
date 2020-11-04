import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}
        self.M = None
        self.x = None
        self.y = None
        self.x_scores = None

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).
        self.x = x
        self.y = y
        self.x_scores = x_scores
        N = self.x_scores.size(0)
        N_range = torch.arange(N)
        y_indices = self.y.to(torch.long)
        s_yi = x_scores[N_range, y_indices]
        self.M = x_scores - s_yi[:, None] + self.delta
        self.M[N_range, y_indices] = 0
        self.M = torch.max(self.M, torch.tensor([0.]))

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        # raise NotImplementedError()
        # ========================

        return sum(sum(self.M)) / N

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.
        y_indices = self.y.to(torch.long)

        N = self.x_scores.size(0)
        N_range = torch.arange(N)
        self.M[self.M > 0] = 1
        y_sums = y_indices.clone()
        y_sums.apply_(lambda x: {k: v.item() for k, v in enumerate(sum(self.M))}.get(x))
        self.M[N_range, y_indices] = y_sums.to(torch.float)
        return torch.matmul(self.x.T, self.M)
