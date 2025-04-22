from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_pdf_report(train_losses, val_losses, train_accuracies, val_accuracies, y_true, y_pred, class_labels, filename="reports/training_report.pdf"):
    # --- Create output folders ---
    output_dir_report = os.path.dirname(filename)
    output_dir_plots = os.path.join(output_dir_report, "plots")
    os.makedirs(output_dir_plots, exist_ok=True)

    # --- Calculate Metrics ---
    accuracy = accuracy_score(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred, target_names=class_labels)

    # --- Save Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_labels, 
                yticklabels=class_labels)
    plt.xlabel("Prediction")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(output_dir_plots, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # --- Save Loss Plot ---
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(output_dir_plots, "loss_plot.png")
    plt.savefig(loss_path)
    plt.close()

    # --- Save Accuracy Plot ---
    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    acc_path = os.path.join(output_dir_plots, "accuracy_plot.png")
    plt.savefig(acc_path)
    plt.close()

    # --- Create PDF ---
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Page 1: Training Plots
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 50, "Model Performance Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 80, f"Model Accuracy: {accuracy:.4f}")

    c.drawImage(loss_path, 100, height - 400, width=400, height=250)
    c.drawImage(acc_path, 100, height - 700, width=400, height=250)
    c.showPage()

    # Page 2: Confusion Matrix
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 50, "Confusion Matrix")
    c.drawImage(cm_path, 100, height - 500, width=400, height=400)
    c.showPage()

    # Page 3: Classification Report
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 50, "Classification Report")
    text = c.beginText(100, height - 80)
    text.setFont("Helvetica", 10)
    for line in clf_report.split('\n'):
        text.textLine(line)
    c.drawText(text)
    c.save()
