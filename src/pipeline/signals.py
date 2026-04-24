#!/usr/bin/env python3
"""
Signal generation from model predictions with confidence scoring and threshold management.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

class SignalGenerator:
    """
    Generates trading signals from model predictions.
    
    Handles:
    - Model inference via SageMaker or local endpoint
    - Confidence threshold management
    - Position tracking
    """
    
    def __init__(
        self,
        endpoint_name: Optional[str] = None,
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.6,
        history_size: int = 192,
    ):
        self.endpoint_name = endpoint_name or os.getenv("ENDPOINT_NAME")
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.history_size = history_size
        
        self.history: list = []
        self._last_prediction: Optional[float] = None
    
    @property
    def threshold(self) -> float:
        return self.buy_threshold
    
    def add_bar(self, bar: Dict[str, Any]) -> None:
        self.history.append(bar)
        if len(self.history) > self.history_size:
            self.history.pop(0)
    
    def get_signal(self, bar: Dict[str, Any]) -> Dict[str, Any]:
        self.add_bar(bar)
        
        if len(self.history) < self.history_size:
            return {
                "signal": 0,
                "confidence": None,
                "reason": "warming_up",
            }
        
        if not self.endpoint_name:
            return {
                "signal": 0,
                "confidence": None,
                "reason": "no_endpoint",
            }
        
        try:
            import boto3
            from botocore.config import Config
            import sagemaker
            from sagemaker.predictor import Predictor
            from sagemaker.serializers import JSONSerializer
            from sagemaker.deserializers import JSONDeserializer
            
            config = Config(read_timeout=30, connect_timeout=30, retries={"max_attempts": 3})
            sm_runtime = boto3.client("sagemaker-runtime", config=config)
            session = sagemaker.Session(sagemaker_runtime_client=sm_runtime)
            predictor = Predictor(
                endpoint_name=self.endpoint_name,
                sagemaker_session=session,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer(),
            )
            
            inputs = self._prepare_inputs()
            result = predictor.predict({"inputs": inputs})
            
            prob = result.get("probability", 0.5)
            self._last_prediction = prob
            
            if prob >= self.buy_threshold:
                return {"signal": 1, "confidence": prob, "reason": "buy"}
            elif prob <= (1.0 - self.sell_threshold):
                return {"signal": -1, "confidence": 1.0 - prob, "reason": "sell"}
            else:
                return {"signal": 0, "confidence": prob, "reason": "hold"}

        except Exception as e:
            return {
                "signal": 0,
                "confidence": None,
                "reason": f"error: {e}",
            }

    def _prepare_inputs(self) -> list:
        from utils import build_windows_from_flat, DEFAULT_FEATURE_COLS
        
        feature_cols = self._get_feature_cols()
        
        rows = []
        for bar in self.history[-self.history_size:]:
            row = [float(bar.get(c, 0.0)) for c in feature_cols]
            rows.append(row)
        
        return rows
    
    def _get_feature_cols(self) -> list:
        from utils import DEFAULT_FEATURE_COLS
        
        return DEFAULT_FEATURE_COLS.copy()


class LocalSignalGenerator:
    """
    Local signal generator using ONNX or PyTorch model.
    """
    
    def __init__(
        self,
        model_path: str,
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.6,
        history_size: int = 192,
        use_onnx: bool = False,
    ):
        self.model_path = model_path
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.history_size = history_size
        self.use_onnx = use_onnx
        
        self.history: list = []
        self._model = None
        self._init_model()
    
    @property
    def threshold(self) -> float:
        return self.buy_threshold
    
    def _init_model(self):
        if self.use_onnx:
            try:
                from pipeline.onnx_inference import ONNXInference
                self._model = ONNXInference(self.model_path)
            except ImportError as e:
                print(f"[signal] ONNX not available: {e}")
                self.use_onnx = False
        
        if not self.use_onnx:
            try:
                import torch
                from models import LSTMClassifier
                
                state = torch.load(self.model_path, map_location="cpu")
                
                from utils import load_meta
                meta = load_meta(os.path.dirname(self.model_path) or ".")
                
                self._model = LSTMClassifier(
                    input_size=int(meta.get("input_size", 15)),
                    hidden_size=int(meta.get("hidden_size", 512)),
                    num_layers=int(meta.get("num_layers", 3)),
                    dropout=float(meta.get("dropout", 0.2)),
                    bidirectional=bool(meta.get("bidirectional", True)),
                    num_classes=2,
                )
                self._model.load_state_dict(state)
                self._model.eval()
            except Exception as e:
                print(f"[signal] Model load error: {e}")
    
    def add_bar(self, bar: Dict[str, Any]) -> None:
        self.history.append(bar)
        if len(self.history) > self.history_size:
            self.history.pop(0)
    
    def get_signal(self, bar: Dict[str, Any]) -> Dict[str, Any]:
        self.add_bar(bar)
        
        if len(self.history) < self.history_size:
            return {
                "signal": 0,
                "confidence": None,
                "reason": "warming_up",
            }
        
        try:
            import numpy as np
            import torch
            
            inputs = self._prepare_inputs()
            x = np.asarray(inputs, dtype=np.float32)[None, ...]
            
            if self.use_onnx:
                probs, _ = self._model.predict(x)
                prob = float(probs[0, 1])
            else:
                with torch.no_grad():
                    logits = self._model(torch.from_numpy(x))
                    probs = torch.softmax(logits, dim=-1)
                    prob = float(probs[0, 1])
            
            if prob >= self.buy_threshold:
                return {"signal": 1, "confidence": prob, "reason": "buy"}
            elif prob <= (1.0 - self.sell_threshold):
                return {"signal": -1, "confidence": 1.0 - prob, "reason": "sell"}
            else:
                return {"signal": 0, "confidence": prob, "reason": "hold"}

        except Exception as e:
            return {
                "signal": 0,
                "confidence": None,
                "reason": f"error: {e}",
            }

    def _prepare_inputs(self) -> list:
        from utils import DEFAULT_FEATURE_COLS

        feature_cols = DEFAULT_FEATURE_COLS.copy()
        
        rows = []
        for bar in self.history[-self.history_size:]:
            row = [float(bar.get(c, 0.0)) for c in feature_cols]
            rows.append(row)
        
        return rows


__all__ = [
    "SignalGenerator",
    "LocalSignalGenerator",
]
