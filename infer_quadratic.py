import torch 
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

if __name__ == '__main__':
    torch.manual_seed(42)
    model = MyModel().to(device)
    model.load_state_dict(torch.load("best_model_epoch", map_location=device, weights_only=True))

    x = torch.Tensor([[2.0]]).to(device)
    model.eval()
    with torch.no_grad():
        y = model(x)
        print(y.item())
