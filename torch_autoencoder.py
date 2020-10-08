import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import scipy
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Implement autoencoder with several different latent
# layer sizes. Train on CIFAR-10.

class Autoencoder(nn.Module):
    def __init__(self, **kwargs):
        super(Autoencoder, self).__init__()
        '''
        self.encoder = nn.Sequential(
            # Try in_channels (first arg) as 3 if issues arise
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        '''
        self.encoder_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=kwargs["hid_units"]
            #128 
            # TODO: Should pass "hidden_units" as a kwargs key
        )
        self.decoder_layer = nn.Linear(
            in_features=kwargs["hid_units"], out_features=kwargs["input_shape"]
        )
        

    def forward(self, features): # , x):
        '''
        x = self.encoder(x)
        x = self.decoder(x)
        '''
        encoded = self.encoder_layer(features)
        encoded = torch.relu(encoded)
        decoded = self.decoder_layer(encoded)
        reconstructed = torch.relu(decoded)
        return reconstructed
    
def train_model(model, data_loader, device, num_epochs=20, learning_rate=1e-3):
    # Creating an optimizer object, using learning rate of 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    outputs = []

    # Train the autoencoder
    for epoch in range(num_epochs):
        loss = 0
        for batch_features, _ in data_loader:
            img = batch_features
            # Reshape mini-batch data to [N, 32*32] matrix
            # Load it to the active device
            batch_features = batch_features.view(-1, 32*32).to(device)

            # reset the gradients
            optimizer.zero_grad()

            # compute reconstructions
            reconstruction = model(batch_features)

            # compute training reconstruction loss
            train_loss = criterion(reconstruction, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # Compute the epoch training loss
        loss = loss / len(data_loader)
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch+1,num_epochs,loss))
        outputs.append((epoch,img,reconstruction),)
    
    return outputs, loss


def training_progression(outputs):
    num_epochs = len(outputs)
    for k in range(0, num_epochs, 5):
        plt.figure(figsize=(9,2))
        imgs = outputs[k][1].detach().cpu().numpy()
        recon = outputs[k][2].detach().cpu().numpy()
        for i, item in enumerate(imgs):
            if i >= 9: break
            plt.subplot(2,9,i+1)
            plt.imshow(item[0])
        for i, item in enumerate(imgs):
            if i >= 9: break
            plt.subplot(2,9,9+i+1)
            plt.imshow(item[0])
    plt.show()


def main():
    CIFAR10_DIM = 32*32
    NUM_EPOCHS = 20

    # DROOT = '/nfs/hpc/share/noelt/data'
    # Load the dataset and transform it
        
    train_dataset = datasets.CIFAR10(
        root='data', train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = datasets.CIFAR10(
        root='data', train=False, transform=transforms.ToTensor(), download=True
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

    hidden_units = [4, 8, 10, 16, 20, 30, 32, 40, 50, 64, 80, 128]
    losses = []

    for hidden_dim in hidden_units:
        # Create an instance of Autoencoder
        model = Autoencoder(input_shape=CIFAR10_DIM, hid_units=hidden_dim).to(device)

        # Note that loss is MSE
        outputs, loss = train_model(model, train_loader, device)

        #training_progression(outputs)

        losses.append(loss)

    plt.plot(hidden_units, losses)
    plt.xlabel('Hidden Units')
    plt.ylabel('Loss, MSE')
    plt.title('Autoencoder Reconstruction Error')
    plt.show()
     
    
if __name__ == '__main__':
    main()
