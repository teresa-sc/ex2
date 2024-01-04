import torch 
from torchvision import transforms

def mnist():
    """Return train and test dataloaders for MNIST."""
    train_data, train_labels = [ ], [ ]
    for i in range(5):
        train_data.append(torch.load(f"data/raw/train_images_{i}.pt"))

    train_data = torch.cat(train_data, dim=0)

    test_data = torch.load("data/raw/test_images.pt")

    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    normalize = transforms.Normalize(0,1)

    train_data = normalize(train_data)
    test_data = normalize(test_data)

    torch.save(train_data, "data/processed/train_images.pt")
    torch.save(test_data, "data/processed/test_images.pt")





if __name__ == '__main__':
    # Get the data and process it
    mnist()

