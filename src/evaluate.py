import torch

def evaluate_model(model, data_loader, device):
    model.eval()  # Evaluation mode
    y_true, y_pred = [], []

    with torch.no_grad():  # No gradients needed
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)  # Forward pass
            predicted = torch.argmax(outputs, dim=1)  # Predicted class

            y_true.extend(torch.argmax(y_batch, dim=1).cpu().numpy())  # True class
            y_pred.extend(predicted.cpu().numpy())  # Store prediction
   
    return y_true, y_pred