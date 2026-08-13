import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader, random_split 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_loss = float('inf')
print(f"Using device is {device}")

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class QuadricDataset(Dataset):
    def __init__(self, num_samples=1000):
        super().__init__()
        self.x = torch.randn(num_samples, 1)
        self.y = self.x ** 2 + 1

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


if __name__ == '__main__':
    torch.manual_seed(42)
    model = MyModel().to(device)
    
    full_dataset = QuadricDataset(num_samples=2500)
    train_dataset_size = int(len(full_dataset) * 0.8)
    val_dataset_size = int(len(full_dataset) - train_dataset_size)
    
    train_dataset, val_dataset = random_split(full_dataset, [train_dataset_size, val_dataset_size])
    
    epochs = 200
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=32,
        shuffle=False        
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(batch_x)

        avg_train_loss = train_loss / len(train_dataset)

        val_loss = 0.0
        model.eval()

        with torch.no_grad():
            for batch_x, batch_y in val_dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                y_pred = model(batch_x)
                loss = loss_fn(y_pred, batch_y)

                val_loss += loss.item() * len(batch_x)

        avg_val_loss = val_loss / len(val_dataset)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model_epoch")
            print("最佳模型已保存")

        if (epoch + 1) % 10 == 0:
            print(f"epoch: {epoch + 1} | train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f}")
