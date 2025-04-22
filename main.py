import config

from src.model import LSTMClassifier
from src.data_preparation import transform_raw_data
from src.preprocessing import data_preprocessing
from src.sliding_windows import sliding_windows_dataloader
from src.reporting import generate_pdf_report
from src.train import train_model
from src.evaluate import evaluate_model

import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":

    # Transform raw data
    transform_raw_data(db_path=config.RAW_DATA_DIR, output_dir=config.PROCESSED_DATA_DIR)

    # Preprocess transformed data (scaling, encoding, splitting)
    X_train, X_val, X_test, y_train, y_val, y_test, enc = data_preprocessing()

    # Create DataLoaders using sliding windows
    print("Creating dataloaders...")
    train_loader = sliding_windows_dataloader(X_train, y_train, window_size=config.WINDOW_SIZE, stride=config.STRIDE, batch_size=config.BATCH_SIZE)
    val_loader = sliding_windows_dataloader(X_val, y_val, window_size=config.WINDOW_SIZE, stride=config.STRIDE, batch_size=config.BATCH_SIZE)
    test_loader = sliding_windows_dataloader(X_test, y_test, window_size=config.WINDOW_SIZE, stride=config.STRIDE, batch_size=config.BATCH_SIZE)
    print("Dataloaders created successfully!")
    print("----------------------------------------------------------------")

    # Model parameters
    num_epochs = config.EPOCHS
    patience = config.PATIENCE
    best_model_path = config.BEST_MODEL_PATH
    input_dim = config.INPUT_DIM
    hidden_dim = config.HIDDEN_DIM
    num_layers = config.NUM_LAYERS
    output_dim = config.OUTPUT_DIM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create and move model to device
    model = LSTMClassifier(input_dim, hidden_dim, num_layers, output_dim)
    model.to(device)

    # Loss and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Train model
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, loss_function, optimizer,
        num_epochs, patience, device, best_model_path
    )

    # Load best saved model
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

    # Evaluate on test set
    y_true, y_pred = evaluate_model(model, test_loader, device)

    # Get class labels from encoder
    class_labels = enc.get_feature_names_out()

    # Generate PDF report
    report_path = config.REPORT_PATH
    print("Generating PDF report...")
    generate_pdf_report(train_losses, val_losses, train_accuracies, val_accuracies, y_true, y_pred, class_labels, filename=report_path)
    print("Report successfully generated: reporte_modelo.pdf")
