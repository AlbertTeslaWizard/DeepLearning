import torch

class EarlyStop:
    def __init__(self, patience=15, delta=1e-4, save_path="best_model.pth", verbose=False):
        self.patience = patience
        self.counter = 0
        self.delta = delta
        self.best_loss = float('inf')
        self.verbose = verbose
        self.save_path = save_path
        self.is_earlystop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta: 
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(f"val_loss 降至 {val_loss:.4f}, 最佳模型已保存")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.is_earlystop = True

          
