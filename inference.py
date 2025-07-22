import os
import json
import torch
import torch.nn as nn
import joblib
import numpy as np

# This class definition is copied directly from aws_train_model.py to ensure a perfect match
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        # hidden_size * 2 for bidirectional LSTM
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Get output of the last time step from the forward and backward passes
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def model_fn(model_dir):
    """
    Loads the saved model, scaler, and configuration from the model_dir.
    The model is now loaded dynamically based on the saved config.
    """
    print("--- Loading model, scaler, and config for inference ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model configuration
    config_path = os.path.join(model_dir, "model_config.json")
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    # Instantiate the model with the loaded configuration
    model = LSTMModel(
        input_size=model_config['input_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        output_size=model_config['output_size'],
        dropout_rate=model_config['dropout_rate']
    )

    # Load the model's learned weights
    model_path = os.path.join(model_dir, "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # Load the scaler
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    scaler = joblib.load(scaler_path)

    print("--- Model, scaler, and config loaded successfully ---")
    return {"model": model, "scaler": scaler, "device": device}

def input_fn(request_body, request_content_type):
    """
    Deserializes the input request body.
    """
    if request_content_type == "application/json":
        data = json.loads(request_body)
        # Ensure input is a numpy array for the scaler
        return np.array(data['inputs'], dtype=np.float32)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_artifacts):
    """
    Makes a prediction with the loaded model and scaler.
    """
    model = model_artifacts["model"]
    scaler = model_artifacts["scaler"]
    device = model_artifacts["device"]

    # The scaler expects a 2D array (n_samples, n_features), which input_data already is.
    scaled_input = scaler.transform(input_data)
    
    # Add a batch dimension for the LSTM model and send to the correct device
    input_tensor = torch.from_numpy(scaled_input).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)

    return prediction

def output_fn(prediction, response_content_type):
    """
    Serializes the prediction output.
    Returns only the probability, leaving the final signal decision to the client.
    """
    if response_content_type == "application/json":
        probability = torch.sigmoid(prediction).item()
        return json.dumps({
            "probability": probability
        })
    raise ValueError(f"Unsupported content type: {response_content_type}")