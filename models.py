from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class BrainAgeCNN(nn.Module):
    """
    The BrainAgeCNN predicts the age given a brain MR-image.
    """
    def __init__(self) -> None:
        super().__init__()

        self.num_features = 1

        # Feel free to also add arguments to __init__ if you want.
        # ----------------------- ADD YOUR CODE HERE --------------------------
        self.conv1_1 = nn.Conv3d(self.num_features, self.num_features*4, kernel_size = 3, stride = 1, padding = 0)
        self.conv2_1 = nn.Conv3d(self.num_features*4, self.num_features*4, kernel_size = 3, stride = 1, padding = 0)
        self.bnn1_1 = nn.BatchNorm3d(self.num_features * 4)
        self.maxp1_1 = nn.MaxPool3d(kernel_size = 2, stride=2, padding=0)

        self.conv1_2 = nn.Conv3d(self.num_features*4, self.num_features*8, kernel_size = 3, stride = 1, padding = 0)
        self.conv2_2 = nn.Conv3d(self.num_features*8, self.num_features*8, kernel_size = 3, stride = 1, padding = 0)
        self.bnn1_2 = nn.BatchNorm3d(self.num_features*8)
        self.maxp1_2 = nn.MaxPool3d(kernel_size = 2, stride=2, padding=0)

        self.conv1_3 = nn.Conv3d(self.num_features*8, self.num_features*16, kernel_size = 3, stride = 1, padding = 0)
        self.conv2_3 = nn.Conv3d(self.num_features*16, self.num_features*16, kernel_size = 3, stride = 1, padding = 0)
        self.bnn1_3 = nn.BatchNorm3d(self.num_features*16)
        self.maxp1_3 = nn.MaxPool3d(kernel_size = 2, stride=2, padding=0)

        self.conv1_4 = nn.Conv3d(self.num_features*16, self.num_features*32, kernel_size = 3, stride = 1, padding = 0)
        self.conv2_4 = nn.Conv3d(self.num_features*32, self.num_features*32, kernel_size = 3, stride = 1, padding = 0)
        self.bnn1_4 = nn.BatchNorm3d(self.num_features*32)
        self.maxp1_4 = nn.MaxPool3d(kernel_size = 2, stride=2, padding=0)

        # self.conv1_5 = nn.Conv3d(self.num_features*64, self.num_features*128, kernel_size = 3, stride = 1, padding = 0)
        # self.conv2_5 = nn.Conv3d(self.num_features*128, self.num_features*128, kernel_size = 3, stride = 1, padding = 0)
        # self.bnn1_5 = nn.BatchNorm3d(self.num_features*128)
        # self.maxp1_5 = nn.MaxPool3d(kernel_size = 2, stride=2, padding=0)

        self.fc1 = nn.Linear(1024, 4)
        # ------------------------------- END ---------------------------------

    def forward(self, imgs: Tensor) -> Tensor:
        """
        Forward pass of your model.

        :param imgs: Batch of input images. Shape (N, 1, H, W, D)
        :return pred: Batch of predicted ages. Shape (N)
        """
        # ----------------------- ADD YOUR CODE HERE --------------------------
        imgs = imgs.type(torch.cuda.FloatTensor)
        print(imgs.shape)
        m = nn.ReLU()

        x = m(self.conv1_1(imgs))
        x = self.maxp1_1(self.bnn1_1(m(self.conv2_1(x))))

        x = m(self.conv1_2(x))
        x = self.maxp1_2(self.bnn1_2(m(self.conv2_2(x))))

        x = m(self.conv1_3(x))
        x = self.maxp1_3(self.bnn1_3(m(self.conv2_3(x))))

        x = m(self.conv1_4(x))
        x = self.maxp1_4(self.bnn1_4(m(self.conv2_4(x))))

        # x1_b5 = m(self.conv1_5(x2_b4))
        # x2_b5 = self.maxp1_5(self.bnn1_5(m(self.conv2_5(x1_b5))))
        #print(x.shape)
        #print(x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4])
        x = x.view(-1, x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4])
        pred = self.fc1(x)
        
        # ------------------------------- END ---------------------------------
        return pred

    def train_step(
        self,
        imgs: Tensor,
        labels: Tensor,
        return_prediction: Optional[bool] = False
    ):
        """Perform a training step. Predict the age for a batch of images and
        return the loss.

        :param imgs: Batch of input images (N, 1, H, W, D)
        :param labels: Batch of target labels (N)
        :return loss: The current loss, a single scalar.
        :return pred
        """
        pred = self(imgs)  # (N)

        # ----------------------- ADD YOUR CODE HERE --------------------------
        print(pred)
        print(labels)
        loss = F.mse_loss(labels.float(), pred.float())
        print(loss)

        # ------------------------------- END ---------------------------------

        if return_prediction:
            return loss, pred
        else:
            return loss