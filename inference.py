# inference.py

import os
import json
import torch
import torch.nn as nn
import joblib
import numpy as np

# This class definition must match the one in your training script
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        # Ensure bidirectional=True matches the training definition if it was used there
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, output_size) # hidden_size * 2 for bidirectional

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def model_fn(model_dir):
    """
    Loads the saved model and scaler from the model_dir.
    This function is called once when the endpoint starts.
    """
    print("--- Loading model and scaler for inference ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # These now match the parameters from your aws_train_model.py script
    model = LSTMModel(input_size=17, hidden_size=128, num_layers=3, output_size=1, dropout_rate=0.5)
    
    model_path = os.path.join(model_dir, "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    scaler = joblib.load(scaler_path)
    
    print("--- Model and scaler loaded successfully ---")
    return {"model": model, "scaler": scaler, "device": device}

def input_fn(request_body, request_content_type):
    """
    Deserializes the input request body. The scaler now expects a numpy array.
    """
    if request_content_type == "application/json":
        data = json.loads(request_body)
        # Return a numpy array, as the scaler expects it
        return np.array(data['inputs'], dtype=np.float32)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_artifacts):
    """
    Makes a prediction with the loaded model and scaler.
    """
    model = model_artifacts["model"]
    scaler = model_artifacts["scaler"]
    device = model_artifacts["device"]
    
    # Scale the numpy input
    scaled_input = scaler.transform(input_data)
    
    # Convert to a tensor for the model, adding the batch dimension
    input_tensor = torch.from_numpy(scaled_input).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)
    
    return prediction

def output_fn(prediction, response_content_type):
    """
    Serializes the prediction output into a JSON response.
    """
    if response_content_type == "application/json":
        probability = torch.sigmoid(prediction).item()
        return json.dumps({
            "probability": probability,
            "signal": 1 if probability > 0.5 else 0
        })
    raise ValueError(f"Unsupported content type: {response_content_type}")