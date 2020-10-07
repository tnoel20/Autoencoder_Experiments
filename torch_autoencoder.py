import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import scipy
import numpy as np
import matplotlib.pyplot as plt

# Implement autoencoder with several different latent
# layer sizes. Train on CIFAR-10.

class Autoencoder(nn.Module):
    def __init__(self, **kwargs):
        # If causes issues, remove "Autoencoder, self" args below
        super(Autoencoder, self).__init__()
        self.encoder_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128 
            # TODO: Should pass "hidden_units" as a kwargs key
        )
        self.decoder_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        encoded = self.encoder_layer(features)
        encoded = torch.relu(encoded)
        decoded = self.decoder_layer(encoded)
        reconstructed = torch.relu(decoded)
        return reconstructed


def main():
    CIFAR10_DIM = 32*32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 20

    # DROOT = '/nfs/hpc/share/noelt/data'
    # Load the dataset and transform it
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root='data', train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='data', train=False, transform=transform, download=True
    )

    # Get data loaders for train and test sets
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    # Use the GPU
    device = torch.device("cuda")

    # Create an instance of Autoencoder
    model = Autoencoder(input_shape=CIFAR10_DIM).to(device)
    
    # Creating an optimizer object, using learning rate of 1e-3
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Specifying MSE as our loss function
    criterion = nn.MSELoss()

    # Train the autoencoder
    for epoch in range(NUM_EPOCHS):
        loss = 0
        for batch_features, _ in train_loader:
            # Reshape mini-batch data to [N, 32*32] matrix
            # Load it to the active device
            batch_features = batch_features.view(-1, 32*32).to(device)

            # reset the gradients
            optimizer.zero_grad()

            # compute reconstructions 
            outputs = model(batch_features)

            # compute training reconstruction loss 
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
        
        # Compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch+1,NUM_EPOCHS,loss))

    
    
if __name__ == '__main__':
    main()
