import torch
import torch.nn as nn


class FtnmtLoss(nn.Module):
    """
    This function calculates the average fractal tanimoto similarity for d = 0...depth
    """

    def __init__(self, depth=5, axis=[1, 2, 3], smooth=1.0e-5, batch_axis=0, weight=None, **kwargs):
        super().__init__()

        assert depth >= 0, ValueError("depth must be >= 0, aborting...")

        self.smooth = smooth
        self.axis = axis
        self.depth = depth

        if depth == 0:
            self.depth = 1
            self.scale = 1.
        else:
            self.depth = depth
            self.scale = 1. / depth

    def inner_prod(self, prob, label):
        prod = torch.mul(prob, label)
        prod = torch.sum(prod, dim=self.axis)

        return prod

    def tnmt_base(self, preds, labels):

        tpl = self.inner_prod(preds, labels)
        tpp = self.inner_prod(preds, preds)
        tll = self.inner_prod(labels, labels)

        num = tpl + self.smooth
        scale = 1. / self.depth
        denum = 0.0
        for d in range(self.depth):
            a = 2. ** d
            b = -(2. * a - 1.)

            denum = denum + torch.reciprocal(torch.add(a * (tpp + tll), b * tpl) + self.smooth)

        result = torch.mul(num, denum) * scale
        return result

    def forward(self, preds, labels):

        l1 = self.tnmt_base(preds, labels)
        l2 = self.tnmt_base(1. - preds, 1. - labels)
        result = 0.5 * (l1 + l2)
        return torch.mean(1. - result)


class MtskLoss(nn.Module):
    def __init__(self, depth=0):
        super().__init__()
        self.ftnmt = FtnmtLoss(depth=depth)

    def forward(self, _predictions, targets):
        loss_segm = self.ftnmt(_predictions[0], targets[0])
        loss_bound = self.ftnmt(_predictions[1], targets[1])
        loss_dist = self.ftnmt(_predictions[2], targets[2])
        return loss_segm, loss_bound, loss_dist


if __name__ == "__main__":
    labels = torch.randint(low=0, high=2, size=(10, 1, 32, 32))
    sigmoid = torch.nn.Sigmoid()
    pred = sigmoid(torch.randn(10, 1, 32, 32))
    loss_0 = MtskLoss(0)
    loss_10 = MtskLoss(10)
    loss_20 = MtskLoss(20)
    # loss expects predictions for segmentation mask, boundaries and distance maps
    # loss
    print(loss_0([pred, pred, pred], [labels, labels, labels]))
    print(loss_10([pred, pred, pred], [labels, labels, labels]))
    print(loss_20([pred, pred, pred], [labels, labels, labels]))