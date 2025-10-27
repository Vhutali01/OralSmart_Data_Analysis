
import torch
import torch.nn as nn
import joblib
import numpy as np

class OralHealthMLPWrapper:
    """
    Wrapper class for the OralHealth MLP model for easy deployment.
    """

    def __init__(self, model_path, scaler_path, feature_names_path):
        """
        Initialize the model wrapper.

        Args:
            model_path: Path to the saved model (.pkl, .pickle, .joblib, or .pth)
            scaler_path: Path to the saved scaler (.pkl or .joblib)
            feature_names_path: Path to the saved feature names (.pkl)
        """
        # Load feature names
        self.feature_names = joblib.load(feature_names_path)
        self.n_features = len(self.feature_names)

        # Load scaler
        self.scaler = joblib.load(scaler_path)

        # Load model
        if model_path.endswith('.pth'):
            self.model_dict = torch.load(model_path, map_location='cpu')
        else:
            self.model_dict = joblib.load(model_path)

        # Recreate the model architecture
        self.model = self._create_model()
        self.model.load_state_dict(self.model_dict['model_state_dict'])
        self.model.eval()

        # Class names
        self.class_names = ['Low Risk', 'Medium Risk', 'High Risk']

    def _create_model(self):
        """Recreate the model architecture."""
        hidden_sizes = self.model_dict['params']['hidden_sizes']
        dropout = self.model_dict['params']['dropout']

        layers = []
        input_size = self.n_features

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_size = hidden_size

        # Output layer
        layers.append(nn.Linear(input_size, 3))  # 3 classes

        return nn.Sequential(*layers)

    def predict(self, X):
        """
        Make predictions on input data.

        Args:
            X: Input data (pandas DataFrame or numpy array)

        Returns:
            predictions: Array of predicted class indices
        """
        # Ensure X has the correct features
        if hasattr(X, 'columns'):
            # DataFrame - select and reorder features
            X = X[self.feature_names]

        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values

        # Scale the data
        X_scaled = self.scaler.transform(X)

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled)

        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)

        return predictions.numpy()

    def predict_proba(self, X):
        """
        Get prediction probabilities.

        Args:
            X: Input data (pandas DataFrame or numpy array)

        Returns:
            probabilities: Array of prediction probabilities
        """
        # Ensure X has the correct features
        if hasattr(X, 'columns'):
            X = X[self.feature_names]

        if hasattr(X, 'values'):
            X = X.values

        # Scale the data
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)

        # Get probabilities
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        return probabilities.numpy()

    def predict_with_labels(self, X):
        """
        Get predictions with class labels.

        Args:
            X: Input data

        Returns:
            tuple: (predictions, class_labels, probabilities)
        """
        pred_indices = self.predict(X)
        pred_labels = [self.class_names[i] for i in pred_indices]
        probabilities = self.predict_proba(X)

        return pred_indices, pred_labels, probabilities

# Usage example:
# model = OralHealthMLPWrapper(
#     model_path='best_pytorch_model.pkl',
#     scaler_path='scaler.pkl', 
#     feature_names_path='feature_names.pkl'
# )
# predictions = model.predict(your_data)
# pred_indices, pred_labels, probabilities = model.predict_with_labels(your_data)
