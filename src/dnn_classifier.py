import torch
import torch.nn as nn

class DNNClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, layers=2):
        super(DNNClassifier, self).__init__()
        
        modules = []
        modules.append(nn.Linear(input_dim, hidden_dim))
        modules.append(nn.ReLU())
        
        for _ in range(layers - 1):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.ReLU())
            
        modules.append(nn.Linear(hidden_dim, 1))
        modules.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*modules)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

    def predict(self, x):
        """
        x: tensor of shape (batch_size, input_dim)
        Returns: probabilities mapped between 0 and 1
        """
        self.eval()
        with torch.no_grad():
            preds = self.model(x)
        return preds.squeeze(-1)
        
    def update(self, x, y):
        self.train()
        self.optimizer.zero_grad()
        predictions = self.model(x).squeeze(-1)
        loss = self.loss_fn(predictions, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
