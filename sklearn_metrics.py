import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 2)

    def forward(self, x):
        out = self.fc1(x)
        return out

class ClassifierDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.X = torch.randn(1000, 5)
        self.y = torch.randint(0, 2, [1000,])
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == '__main__':
    torch.manual_seed(42)

    test_dataset = ClassifierDataset()
    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = 32,
        shuffle = False
    )

    model = ClassifierModel().to(device)
    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for X, y in test_dataloader:
            X = X.to(device)
            y = y.to(device)
            
            y_pred = model(X)
            preds = torch.argmax(y_pred, dim = 1)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y.cpu().numpy())
    
    accuracy = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds)

    print(f"accuracy: {accuracy: .4f}")
    print(f"f1: {f1: .4f}")




        
