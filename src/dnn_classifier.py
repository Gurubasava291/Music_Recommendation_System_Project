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
    
    def fit(self, X, y, epochs=10, batch_size=32, verbose=True):
        """
        Train the model using X (features) and y (targets).
        X, y can be numpy arrays or torch tensors.
        """
        from torch.utils.data import DataLoader, TensorDataset
        
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
            
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_X, batch_y in dataloader:
                loss = self.update(batch_X, batch_y)
                total_loss += loss
                
            if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")


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

    def score(self, X, y):
        """
        Evaluate the model and return the accuracy score.
        X, y can be numpy arrays or torch tensors.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
            
        preds = self.predict(X)
        # Assuming binary classification with 0.5 threshold
        predicted_classes = (preds >= 0.5).float()
        accuracy = (predicted_classes == y).float().mean().item()
        return accuracy

