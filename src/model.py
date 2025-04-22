import torch.nn as nn

# Arquitectura de la red
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Capa final que transforma la última salida oculta en la predicción
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)


    def forward(self, x):
        # x: (batch_size, sequence_length, input_dim)
        out, (hn, cn) = self.lstm(x)
        # Se toma la salida del último paso de la secuencia
        last_time_step = out[:, -1, :]  # (batch_size, hidden_dim)
        out = self.fc1(last_time_step)   # (batch_size, output_dim)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc4(out)
        return out