import torch
from torch.utils.data import Dataset, DataLoader

# Algoritmo de ventanas deslizantes
class SlidingWindowDataset(Dataset):
    def __init__(self, X, y, window_size=4200, stride=2100, id_column='id_Serie'):
        """
        X: DataFrame con las variables predictoras.
        y: DataFrame con las etiquetas one-hot (8 columnas). No contiene id_Serie.
        window_size: número de filas que tendrá cada ventana.
        stride: número de filas a saltar para la siguiente ventana.
        id_column: nombre de la columna en X que identifica cada serie.
        """
        assert len(X) == len(y), "X e y deben tener el mismo número de filas"
        self.X = X
        self.y = y
        self.window_size = window_size
        self.stride = stride
        self.id_column = id_column

        # Crea un diccionario que mapea cada serie a la lista de índices
        self.series_indices = {}
        # Se almacena la lista de ventanas, donde cada elemento es una tupla (serie_id, start)
        # 'start' es la posición donde inicia la ventana.
        self.windows = []

        for serie in self.X[self.id_column].unique():
            # Obtiene la lista de índices para cada serie.
            indices = self.X.index[self.X[self.id_column] == serie].tolist()
            self.series_indices[serie] = indices
            n_rows = len(indices)

            # Si la serie tiene suficientes filas para al menos una ventana:
            if n_rows >= window_size:
                for start in range(0, n_rows - window_size + 1, stride):
                    self.windows.append((serie, start))
                    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        # Obtiene la información de la ventana
        serie_id, start_in_series = self.windows[idx]
        # Recupera la lista de índices para esta serie
        indices = self.series_indices[serie_id]
        # Calcula los índices globales de la ventana
        window_indices = indices[start_in_series: start_in_series + self.window_size]
        
        # Extrae la ventana de X eliminando la columna id_Serie (ya que no es necesaria para el modelo)
        window_X = self.X.loc[window_indices].drop(columns=[self.id_column]).values  
        # Extrae la etiqueta (se toma la última fila de la ventana en y)
        window_y = self.y.loc[window_indices[-1]].values 
        
        # Convertir a tensores de PyTorch
        window_tensor = torch.tensor(window_X, dtype=torch.float32)
        label_tensor = torch.tensor(window_y, dtype=torch.float32)
        return window_tensor, label_tensor
    
def sliding_windows_dataloader(X, y, window_size, stride, batch_size):
    dataset = SlidingWindowDataset(X, y, window_size, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader