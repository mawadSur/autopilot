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
    Loads the saved model and scaler from the model_dir.
    """
    print("--- Loading model and scaler for inference ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    Deserializes the input request body.
    """
    if request_content_type == "application/json":
        data = json.loads(request_body)
        inputs = data.get("inputs", None)
        if inputs is None:
            raise ValueError("JSON must have a key 'inputs'")
        # Expecting shape: (window, features)
        arr = np.array(inputs, dtype=np.float32)
        return arr
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_artifacts):
    """
    Makes a prediction with the loaded model and scaler.
    """
    model = model_artifacts["model"]
    scaler = model_artifacts["scaler"]
    device = model_artifacts["device"]

    scaled_input = scaler.transform(input_data)
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
            "signal": 1 if probability > float(os.getenv("PROB_THRESHOLD", "0.5")) else 0
        })
    raise ValueError(f"Unsupported content type: {response_content_type}")