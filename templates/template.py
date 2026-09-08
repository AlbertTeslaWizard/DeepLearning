import torch 
import torch.nn as nn 
import torch.optim as optim

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(torch.relu(x))

        return x

if __name__ == '__main__':
    torch.manual_seed(42)

    X = torch.randn(1000, 2)
    Y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 1).float().unsqueeze(1)

    model = MyModel()
    epochs = 200
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)

    for epoch in range(epochs):
        y_pred = model(X)
        loss = loss_fn(y_pred, Y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if((epoch + 1) % 10 == 0):
            acc = ((torch.sigmoid(y_pred) > 0.5).float() == Y).float().mean()
            print(f"epoch: {epoch + 1: 3d} | loss: {loss.item(): .4f} | acc: {acc: .2f}")
    
    
    model.eval()
    with torch.no_grad():
        test_data = torch.Tensor([[0.5, 0.5], [0.2, 0.99]])
        test_pred = model(test_data)
        test_pred_label = [torch.sigmoid(test_pred) > 0.5]

        print(test_pred_label)

