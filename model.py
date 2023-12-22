import torch 
import torch.nn as nn
from typing import Optional

class ConvNet3D(nn.Module):
    def __init__(self, num_kernels: Optional[int] = 8, hidden_size: Optional[int] = 16):
        super(ConvNet3D, self).__init__()

        self.num_kernels = num_kernels
        self.hidden_size = hidden_size

        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(1, num_kernels, kernel_size=3, stride=1, padding="same")
        self.conv2 = nn.Conv3d(num_kernels, num_kernels * 2, kernel_size=3, stride=1, padding="same")
        self.conv3 = nn.Conv3d(num_kernels * 2, num_kernels * 4, kernel_size=3, stride=1, padding="same")

        # Max pooling layers
        self.pool = nn.MaxPool3d(kernel_size=(4,2,2))

        # Activation function
        self.lrelu = nn.LeakyReLU()

        # Batch Norm layer
        self.bnorm1 = nn.BatchNorm3d(num_kernels)
        self.bnorm2 = nn.BatchNorm3d(num_kernels * 2)
        self.bnorm3 = nn.BatchNorm3d(num_kernels * 4)

        # Fully connected layers
        self.fc1 = nn.Linear(num_kernels * 4 * 8 * 12 * 12, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Input shape: (batch_size(B), channels(C), window_size(W'), height(H), depth(D))

        # Convolutional layers
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.lrelu(x)
        x = self.bnorm1(x)
        x = self.pool(x)
        # print(x.shape)

        x = self.conv2(x)
        # print(x.shape)
        x = self.lrelu(x)
        x = self.bnorm2(x)
        x = self.pool(x)
        # print(x.shape)

        x = self.conv3(x)
        # print(x.shape)
        x = self.lrelu(x)
        x = self.bnorm3(x)
        x = self.pool(x)
        # print(x.shape)

        # Flatten before fully connected layers
        x = x.view(-1, self.num_kernels * 4 * 8 * 12 * 12)
        # print(x.shape)

        # Fully connected layers
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.lrelu(x)

        x = self.fc2(x)

        return x


if __name__ == "__main__":

    # Create an instance of the Custom3DConvNet model
    conv3d_model = ConvNet3D(8, 32)

    # Device to CUDA 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    conv3d_model.to(device)

    # # Example input tensor
    input_tensor = torch.randn((4, 1, 512, 100, 100)).to(torch.float32).to(device)

    # Forward pass
    output = conv3d_model(input_tensor)

    # Print the output shape
    print("Output shape:", output.shape)
