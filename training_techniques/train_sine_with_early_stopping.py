import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader, random_split
import math 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device is {device}")

class RegularizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x

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
    def __init__(self, num_samples=1000, noise=0):
        super().__init__()
        self.x = torch.rand(num_samples, 1)
        self.y = torch.sin(2 * math.pi * self.x) + 1 + noise * torch.randn_like(self.x)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class EarlyStop:
    def __init__(self, patience=15, delta=1e-4, save_path="best_model.pth", verbose=False):
        self.patience = patience
        self.counter = 0
        self.is_earlystop = False
        self.best_loss = float('inf')
        self.verbose = verbose 
        self.delta = delta 
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            
            if self.verbose:
                print(f"val_loss 降至 {val_loss: .4f}，最佳模型已经保存")
        else:
            self.counter += 1 
            if self.verbose:
                print(f"val loss {val_loss: .4f} 未显著下降，{self.counter} | {self.patience}") 
            
            if self.counter >= self.patience:
                self.is_earlystop = True



if __name__ == '__main__':
    torch.manual_seed(42)
    model = RegularizedModel().to(device)
    
    full_dataset = QuadricDataset(num_samples=2500)
    train_dataset_size = int(len(full_dataset) * 0.8)
    val_dataset_size = int(len(full_dataset) - train_dataset_size)
    
    train_dataset, val_dataset = random_split(full_dataset, [train_dataset_size, val_dataset_size])
    
    epochs = 100 
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

    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    loss_fn = nn.MSELoss()

    earlystop = EarlyStop()

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
        scheduler.step()

        earlystop(avg_val_loss, model)
        if earlystop.is_earlystop:
            print(f"\n连续 {earlystop.patience} 轮 val_loss 无显著改善，触发早停")
            break
        
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"epoch: {epoch + 1} | train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f} | current_lr: {current_lr}")

    
    model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
    model.eval()
    test_data_x = torch.rand(10, 1)
    test_data_y_true = torch.sin(2 * math.pi * test_data_x) + 1

    with torch.no_grad():
        for x, y in zip(test_data_x, test_data_y_true):
            x = x.to(device)
            test_data_y_pred = model(x.unsqueeze(0))
            print(f"x: {x.item(): .4f} | y_pred: {test_data_y_pred.item(): .4f} | test_data_y_true: {y.item(): .4f}")
        
        
        


        






