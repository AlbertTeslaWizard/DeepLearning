import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = datasets.MNIST(
    root = "./data",
    train = True,
    download = True,
    transform = transform
)

test_dataset = datasets.MNIST(
    root = "./data",
    train = False,
    download = True,
    transform = transform
)

train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 64,
    shuffle = True
)

test_loader = DataLoader(
    dataset = test_dataset,
    batch_size = 64,
    shuffle = False
)

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
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x 

model = SimpleCNN().to(device)
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.001
)

for epoch in range(5):
    model.train()

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} | Loss: {loss.item(): .4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        predicted_class = torch.argmax(pred, dim=1)
        correct += (predicted_class == y).sum().item()
        total += len(y)

accuracy = correct / total
print(f"Test Accuracy: {accuracy: .4f}")
