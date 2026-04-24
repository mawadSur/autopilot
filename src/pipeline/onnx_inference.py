#!/usr/bin/env python3
"""
ONNX conversion and inference utilities for faster local inference.

Converts PyTorch LSTMClassifier to ONNX for ~10x faster inference without GPU.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    onnx = None
    ort = None


def pytorch_to_onnx(
    model: nn.Module,
    output_path: str,
    input_size: int,
    seq_len: int = 192,
    hidden_size: int = 512,
    num_layers: int = 3,
    dropout: float = 0.2,
    bidirectional: bool = True,
    num_classes: int = 2,
) -> bool:
    """
    Convert PyTorch LSTMClassifier to ONNX format.
    
    Args:
        model: PyTorch LSTMClassifier instance
        output_path: Path to save ONNX model
        input_size: Number of input features
        seq_len: Sequence/window length
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        bidirectional: Use bidirectional LSTM
        num_classes: Number of output classes
        
    Returns:
        True if conversion successful
    """
    if onnx is None:
        raise ImportError("onnx and onnxruntime required. Install: pip install onnx onnxruntime")
    
    model.eval()
    
    dummy_input = torch.randn(1, seq_len, input_size, dtype=torch.float32)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "seq_len"},
            "output": {0: "batch_size"}
        },
    )
    
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"[onnx] Model saved to {output_path}")
    return True


class ONNXInference:
    """
    Fast ONNX Runtime inference wrapper with optional caching.
    """
    
    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = None,
        cache_predictions: bool = True,
        cache_ttl: int = 60,
    ):
        """
        Initialize ONNX inference session.
        
        Args:
            model_path: Path to ONNX model file
            providers: List of ONNX Runtime providers (CPU, CUDA, TensorRT)
            cache_predictions: Enable prediction caching with Redis
            cache_ttl: Cache time-to-live in seconds
        """
        if ort is None:
            raise ImportError("onnxruntime required. Install: pip install onnxruntime")
        
        if providers is None:
            providers = ["CPUExecutionProvider"]
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        self.cache_enabled = cache_predictions
        self.cache_ttl = cache_ttl
        
        if cache_predictions:
            self._init_redis_cache()
    
    def _init_redis_cache(self):
        """Initialize Redis connection for caching."""
        try:
            import redis
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            self.redis = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
            )
            self.redis.ping()
            print(f"[cache] Redis connected at {redis_host}:{redis_port}")
        except Exception as e:
            print(f"[cache] Redis not available: {e}. Caching disabled.")
            self.redis = None
            self.cache_enabled = False
    
    def _get_cache_key(self, inputs: np.ndarray) -> str:
        """Generate cache key from input hash."""
        arr_hash = hash(inputs.tobytes())
        return f"onnx:pred:{arr_hash}"
    
    def predict(
        self,
        inputs: np.ndarray,
        use_cache: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        Run inference with optional caching.
        
        Args:
            inputs: Input array [batch, seq_len, features]
            use_cache: Whether to use prediction cache
            
        Returns:
            Tuple of (probabilities, inference_time_ms)
        """
        import time
        
        orig_shape = inputs.shape
        
        if inputs.ndim == 3:
            inputs = inputs.reshape(1, *inputs.shape)
        
        if use_cache and self.cache_enabled and self.redis is not None:
            cache_key = self._get_cache_key(inputs)
            cached = self.redis.get(cache_key)
            if cached is not None:
                return np.array([float(cached)]), 0.0
        
        start = time.perf_counter()
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: inputs.astype(np.float32)}
        )[0]
        elapsed = (time.perf_counter() - start) * 1000
        
        probs = 1 / (1 + np.exp(-outputs))
        
        if use_cache and self.cache_enabled and self.redis is not None:
            cache_key = self._get_cache_key(inputs)
            self.redis.setex(
                cache_key,
                self.cache_ttl,
                str(probs[0, 1] if probs.ndim > 1 else probs[0])
            )
        
        return probs, elapsed
    
    def predict_batch(
        self,
        inputs: np.ndarray,
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Run batched inference for multiple windows.
        
        Args:
            inputs: Input array [n_windows, seq_len, features]
            batch_size: Processing batch size
            
        Returns:
            Tuple of (probabilities array, list of inference times)
        """
        n_windows = inputs.shape[0]
        results = []
        times = []
        
        for i in range(0, n_windows, batch_size):
            batch = inputs[i:i + batch_size]
            probs, t = self.predict(batch, use_cache=False)
            results.append(probs)
            times.append(t)
        
        return np.vstack(results), times


def convert_pytorch_to_onnx(
    model_dir: str,
    output_dir: Optional[str] = None,
) -> str:
    """
    Convert model.pt to model.onnx in the same directory.
    
    Args:
        model_dir: Directory with model.pt and model_meta.json
        output_dir: Output directory (defaults to model_dir)
        
    Returns:
        Path to created ONNX model
    """
    from models import LSTMClassifier
    
    model_dir = Path(model_dir)
    if output_dir is None:
        output_dir = model_dir
    else:
        output_dir = Path(output_dir)
    
    meta_path = model_dir / "model_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    
    model = LSTMClassifier(
        input_size=int(meta.get("input_size", 15)),
        hidden_size=int(meta.get("hidden_size", 512)),
        num_layers=int(meta.get("num_layers", 3)),
        dropout=float(meta.get("dropout", 0.2)),
        bidirectional=bool(meta.get("bidirectional", True)),
        num_classes=int(meta.get("num_classes", 2)),
    )
    
    state_path = model_dir / meta.get("model_state_path", "model.pt")
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    
    output_path = output_dir / "model.onnx"
    
    pytorch_to_onnx(
        model,
        str(output_path),
        input_size=int(meta.get("input_size", 15)),
        seq_len=int(meta.get("window_size", 192)),
        hidden_size=int(meta.get("hidden_size", 512)),
        num_layers=int(meta.get("num_layers", 3)),
        dropout=float(meta.get("dropout", 0.2)),
        bidirectional=bool(meta.get("bidirectional", True)),
        num_classes=int(meta.get("num_classes", 2)),
    )
    
    return str(output_path)


__all__ = [
    "pytorch_to_onnx",
    "ONNXInference",
    "convert_pytorch_to_onnx",
]