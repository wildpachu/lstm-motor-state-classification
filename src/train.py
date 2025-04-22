import torch

def evaluate_model(model, data_loader, loss_function, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = loss_function(outputs, y_batch)
            running_loss += loss.item()

            # Accuracy calculation
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == torch.argmax(y_batch, dim=1)).sum().item()
            total += y_batch.size(0)
    
    avg_loss = running_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, loss_function, optimizer, num_epochs, patience, device, best_model_path):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float("inf")
    counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_function(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Accuracy calculation for training
            predicted = torch.argmax(outputs, dim=1)
            correct_train += (predicted == torch.argmax(y_batch, dim=1)).sum().item()
            total_train += y_batch.size(0)

        # Average training loss and accuracy
        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Evaluation on validation set
        val_loss, val_acc = evaluate_model(model, val_loader, loss_function, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Print epoch results
        print(f'Epoch {epoch + 1}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Early stopping and saving the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"--------Improved model saved at {best_model_path}--------")
            counter = 0  # Reset counter on improvement
        else:
            counter += 1  # Increment counter when no improvement

        if counter >= patience:
            print(f"-----Early stopping triggered after {patience} epochs without improvement.-----")
            break

    return train_losses, val_losses, train_accuracies, val_accuracies