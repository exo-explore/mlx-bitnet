import unittest
import mlx.core as mx
import torch
from torch import nn as torch_nn
import numpy as np
from mlx_bitnet import BitLinear as MLXBitLinear
from mlx_bitnet import BitnetRMSNorm as MLXBitnetRMSNorm
from mlx_bitnet import BitnetRotaryEmbedding as MLXBitnetRotaryEmbedding
from minimal_bitnet import BitLinear as TorchBitLinear
from minimal_bitnet import BitnetRMSNorm as TorchBitnetRMSNorm
from minimal_bitnet import BitnetRotaryEmbedding as TorchBitnetRotaryEmbedding

class TestBitLinearInterop(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.cpu)
        self.mlx_bit_linear = MLXBitLinear(3, 2, bias=False, weight_bits=1, input_bits=8)
        self.weights = [("weight", mx.random.uniform(shape=(2, 3)))]
        self.mlx_bit_linear.load_weights(self.weights)
        self.input_tensor = mx.random.uniform(shape=(2, 3))

        self.torch_bit_linear = TorchBitLinear(3, 2, bias=False, weight_bits=1, input_bits=8).cpu()
        torch_weights = [("weight", torch_nn.Parameter(torch.tensor(np.array(self.weights[0][1])).cpu()))]
        self.torch_bit_linear.load_state_dict(dict(torch_weights))
        self.torch_input_tensor = torch.tensor(np.array(self.input_tensor)).cpu()

    def test_output_tensor_comparison(self):
        mlx_output_tensor = self.mlx_bit_linear.forward(self.input_tensor)
        torch_output_tensor = self.torch_bit_linear.forward(self.torch_input_tensor)
        
        # Check if the output tensors are close enough
        self.assertTrue(torch.allclose(torch_output_tensor, torch.tensor(np.array(mlx_output_tensor)), atol=1e-6), "Output tensors do not match.")
class TestBitnetRMSNormInterop(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.cpu)
        self.hidden_size = 2048
        self.eps = 1e-6
        self.mlx_rms_norm = MLXBitnetRMSNorm(self.hidden_size, self.eps)
        self.input_tensor = mx.random.uniform(shape=(2, self.hidden_size))

        self.torch_rms_norm = TorchBitnetRMSNorm(self.hidden_size, self.eps).cpu()
        torch_input_tensor = torch.tensor(np.array(self.input_tensor)).cpu()
        self.torch_input_tensor = torch_input_tensor.float()

    def test_output_tensor_comparison(self):
        mlx_output_tensor = self.mlx_rms_norm.forward(self.input_tensor)
        torch_output_tensor = self.torch_rms_norm.forward(self.torch_input_tensor)
        
        # Check if the output tensors are close enough
        self.assertTrue(torch.allclose(torch_output_tensor, torch.tensor(np.array(mlx_output_tensor)), atol=1e-6), "Output tensors do not match.")
class TestBitnetRotaryEmbeddingInterop(unittest.TestCase):
    def setUp(self):
        self.dim = 64
        self.max_position_embeddings = 512
        self.base = 10000
        self.scaling_factor = 1.0

        self.mlx_rotary_embedding = MLXBitnetRotaryEmbedding(self.dim, self.max_position_embeddings, self.base, scaling_factor=self.scaling_factor)
        self.torch_rotary_embedding = TorchBitnetRotaryEmbedding(self.dim, self.max_position_embeddings, self.base, scaling_factor=self.scaling_factor)

        self.position_ids = mx.arange(0, self.max_position_embeddings)

        self.torch_position_ids = torch.arange(0, self.max_position_embeddings)

    def test_rotary_embedding_output_comparison(self):
        mlx_cos, mlx_sin = self.mlx_rotary_embedding.forward(None, self.position_ids)
        torch_cos, torch_sin = self.torch_rotary_embedding.forward(None, self.torch_position_ids)

        # Check if the cosine embeddings are close enough
        self.assertTrue(torch.allclose(torch_cos, torch.tensor(np.array(mlx_cos)), atol=1e-6), "Cosine embeddings do not match.")

        # Check if the sine embeddings are close enough
        self.assertTrue(torch.allclose(torch_sin, torch.tensor(np.array(mlx_sin)), atol=1e-6), "Sine embeddings do not match.")
class TestTensorExpandInterop(unittest.TestCase):
    def test_tensor_expand(self):
        # Create a tensor in both frameworks
        torch_tensor = torch.tensor([[1], [2], [3]])
        mlx_tensor = mx.array([[1], [2], [3]])

        # Expand the tensors
        torch_expanded = torch_tensor.expand(3, 4)
        mlx_expanded = mx.broadcast_to(mlx_tensor, (3, 4))

        # Check if the expanded tensors are equal
        self.assertTrue(torch.equal(torch_expanded, torch.tensor(np.array(mlx_expanded))), "Expanded tensors do not match.")
class TestTensorNegativeExpandInterop(unittest.TestCase):
    def test_tensor_negative_expand(self):
        # Create a tensor in both frameworks
        torch_tensor = torch.tensor([[1], [2], [3]])
        mlx_tensor = mx.array([[1], [2], [3]])

        # Expand the tensors with -1 indicating copying the existing dimension
        torch_expanded = torch_tensor.expand(-1, 4)  # -1 means not changing the dimension
        mlx_expanded = mx.broadcast_to(mlx_tensor, (-1, 4))  # MXNet does not support -1 in broadcast_to directly

        # Check if the expanded tensors are equal
        self.assertTrue(torch.equal(torch_expanded, torch.tensor(np.array(mlx_expanded))), "Negative expanded tensors do not match.")


if __name__ == '__main__':
    unittest.main()
