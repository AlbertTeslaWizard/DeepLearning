import torch 
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 16,
            kernel_size = 3,
            padding = 1
        )

        self.pool = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )

        self.conv2 = nn.Conv2d(
            in_channels = 16,
            out_channels = 32,
            kernel_size = 3,
            padding = 1
        )

        self.fc = nn.Linear(32 * 7 * 7, 10)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)

        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
        
if __name__ == '__main__':
    torch.manual_seed(42)
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root = './data',
        train = True,
        download = True,
        transform = transform
    )

    test_dataset = datasets.MNIST(
        root = './data',
        train = False,
        download = True,
        transform = transform
    )

    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = 64,
        shuffle = True 
    )

    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = 64,
        shuffle = False
    )
    
    model = SimpleCNN().to(device)
    model.train()
    
    epochs = 10
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr = 0.001)
    
    for epoch in range(epochs):
        train_loss = 0.0

        for X, y in train_dataloader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(X)
        
        avg_loss = train_loss / len(train_dataset)
        print(f"epoch: {epoch + 1} | loss: {avg_loss: .4f}")

    model.eval()
    with torch.no_grad():
        corrects = 0
        for X, y in test_dataloader:
            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)
            preds = torch.argmax(y_pred, dim = 1)
            corrects += (preds == y).sum().item()
        
        accuracy = corrects / len(test_dataset) * 1.0
        print(f"accuracy: {accuracy: .4f}")

