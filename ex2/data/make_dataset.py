import torch
from torchvision import transforms

def mnist():
    """Return train and test dataloaders for MNIST."""
    train_data, train_labels = [ ], [ ]
    for i in range(5):
        train_data.append(torch.load(f"data/raw/train_images_{i}.pt"))
        train_labels.append(torch.load(f"data/raw/train_target_{i}.pt"))

    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    test_data = torch.load("data/raw/test_images.pt")
    test_labels = torch.load("data/raw/test_target.pt")

    normalize = transforms.Normalize(0,1)

    train_data = normalize(train_data)
    test_data = normalize(test_data)

    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    torch.save(torch.utils.data.TensorDataset(train_data, train_labels), "data/processed/train_images.pt")
    torch.save(torch.utils.data.TensorDataset(test_data, test_labels), "data/processed/test_images.pt")





if __name__ == '__main__':
    # Get the data and process it
    mnist()
