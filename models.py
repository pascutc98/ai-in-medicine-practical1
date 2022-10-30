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

        self.loss = torch.nn.L1Loss()


        # Feel free to also add arguments to __init__ if you want.
        # ----------------------- ADD YOUR CODE HERE --------------------------
        self.conv1_1 = nn.Conv3d(in_channels = 1, out_channels = 4, kernel_size = 3, stride = 1, padding = 0)
        self.relu1_1 = nn.ReLU()
        self.conv2_1 = nn.Conv3d(in_channels = 4, out_channels = 4, kernel_size = 3, stride = 1, padding = 0)
        self.bnn1_1 = nn.BatchNorm3d(num_features = 4)
        self.relu2_1 = nn.ReLU()
        self.maxp1_1 = nn.MaxPool3d(kernel_size = 2, stride=2, padding=0)

        self.conv1_2 = nn.Conv3d(in_channels = 4, out_channels = 8, kernel_size = 3, stride = 1, padding = 0)
        self.relu1_2 = nn.ReLU()
        self.conv2_2 = nn.Conv3d(in_channels = 8, out_channels = 8, kernel_size = 3, stride = 1, padding = 0)
        self.bnn1_2 = nn.BatchNorm3d(num_features = 8)
        self.relu2_2 = nn.ReLU()
        self.maxp1_2 = nn.MaxPool3d(kernel_size = 2, stride=2, padding=0)

        self.conv1_3 = nn.Conv3d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 1, padding = 0)
        self.relu1_3 = nn.ReLU()
        self.conv2_3 = nn.Conv3d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 0)
        self.bnn1_3 = nn.BatchNorm3d(num_features = 16)
        self.relu2_3 = nn.ReLU()
        self.maxp1_3 = nn.MaxPool3d(kernel_size = 2, stride=2, padding=0)

        self.conv1_4 = nn.Conv3d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 0)
        self.relu1_4 = nn.ReLU()
        self.conv2_4 = nn.Conv3d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 0)
        self.bnn1_4 = nn.BatchNorm3d(num_features = 32)
        self.relu2_4 = nn.ReLU()
        self.maxp1_4 = nn.MaxPool3d(kernel_size = 2, stride=2, padding=0)

        self.fc1 = nn.Linear(2048, 1024)
        #self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 8)
        self.relu1_5 = nn.ReLU()

       
        # ------------------------------- END ---------------------------------

    def forward(self, imgs: Tensor) -> Tensor:
        """
        Forward pass of your model.

        :param imgs: Batch of input images. Shape (N, 1, H, W, D)
        :return pred: Batch of predicted ages. Shape (N)
        """
        # ----------------------- ADD YOUR CODE HERE --------------------------
       

        x = self.relu1_1(self.conv1_1(imgs))
        x = self.maxp1_1(self.relu2_1(self.bnn1_1(self.conv2_1(x))))

        x = self.relu1_2(self.conv1_2(x))
        x = self.maxp1_2(self.relu2_2(self.bnn1_2(self.conv2_2(x))))

        x = self.relu1_3(self.conv1_3(x))
        x = self.maxp1_3(self.relu2_3(self.bnn1_3(self.conv2_3(x))))

        x = self.relu1_4(self.conv1_4(x))
        x = self.maxp1_4(self.relu2_4(self.bnn1_4(self.conv2_4(x))))

        x = x.view(-1, x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4])
        pred = self.relu1_5(self.fc2(self.fc1(x))
        
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
        pred = torch.squeeze(self.forward(imgs.float()))  # (N)
        #print(pred)
        # ----------------------- ADD YOUR CODE HERE --------------------------
        
        loss = self.loss(pred, labels.float())
        #pred = pred.float()
        #print(loss)
        # ------------------------------- END ---------------------------------

        if return_prediction:
            return loss, pred
        else:
            return loss