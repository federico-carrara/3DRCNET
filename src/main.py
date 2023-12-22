from torch.utils.data import DataLoader
from torchsummary import summary
from datasets import OCTDataset
from model import ConvNet3D
from train import train_model

### Define hyperparameters ###
batch_sz = 4
n_workers = 4
n_kernels = 8
hidden_sz = 32
n_epochs = 50
lr = 1e-4
##############################

# Create datasets
print("Loading datasets ...")
train_dataset = OCTDataset(path_to_data="./Preprocessed_Data", split="train", patch_size=100)
val_dataset = OCTDataset(path_to_data="./Preprocessed_Data", split="val", patch_size=100)
# test_dataset = OCTDataset(path_to_data="./Preprocessed_Data", split="test")
print("DONE!!")

# Create Dataloaders
print("Creating DataLoaders ...")
train_dataloader = DataLoader(train_dataset, batch_size=batch_sz, num_workers=n_workers, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_sz, num_workers=n_workers, shuffle=False)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_sz, num_workers=n_workers, shuffle=False)
print("DONE!!")

# Build the Model
print("Instantiating the model ...")
conv3d_model = ConvNet3D(num_kernels=n_kernels, hidden_size=hidden_sz)
print("DONE!!")

# Train the model
print("Started training ...")
train_model(conv3d_model, train_dataloader, val_dataloader, num_epochs=n_epochs, learning_rate=lr)
