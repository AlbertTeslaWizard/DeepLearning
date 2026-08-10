import torch 
import torch.nn as nn
import torch.optim as optim

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)
    
    def forward(self, x):
        x = self.fc(x)
        return x

if __name__ == '__main__':
    torch.manual_seed(42)
    model = MyModel()

    X = torch.randn(1000, 1)
    Y = 2 * X + 2

    epochs = 50
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        y_pred = model(X)
        loss = loss_fn(y_pred, Y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if((epoch + 1) % 5 == 0):
            print(f"epoch: {epoch + 1: 2d} | loss: {loss.item(): .4f}")
    
    test_x_data = torch.randn(10, 1)
    test_y = 2 * test_x_data + 2

    for x, y in zip(test_x_data, test_y):
        test_y_pred = model(x)
        print(f"test_x: {x.item(): .4f} | test_y_pred: {test_y_pred.item(): .4f} | test_y: {y.item(): .4f}")

    w = model.fc.weight
    b = model.fc.bias

    print(f"w: {w.item(): .4f}, b: {b.item(): .4f}")


