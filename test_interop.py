import unittest
import mlx.core as mx
import mlx.nn as mx_nn
import torch
from torch import nn as torch_nn
import numpy as np
from mlx_bitnet import BitLinear as MLXBitLinear
from mlx_bitnet import BitnetRMSNorm as MLXBitnetRMSNorm
from mlx_bitnet import BitnetRotaryEmbedding as MLXBitnetRotaryEmbedding
from mlx_bitnet import BitnetMLP as MLXBitnetMLP
from mlx_bitnet import BitnetAttention as MLXBitnetAttention
from mlx_bitnet import MinimalBitnetConfig
from minimal_bitnet import BitLinear as TorchBitLinear
from minimal_bitnet import BitnetRMSNorm as TorchBitnetRMSNorm
from minimal_bitnet import BitnetRotaryEmbedding as TorchBitnetRotaryEmbedding
from minimal_bitnet import BitnetMLP as TorchBitnetMLP
from minimal_bitnet import BitnetAttention as TorchBitnetAttention
from transformers.activations import silu as torch_silu

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

        # Initialize BitLinear with bias
        self.mlx_bit_linear_with_bias = MLXBitLinear(3, 2, bias=True, weight_bits=1, input_bits=8)
        weights_with_bias = [("weight", mx.random.uniform(shape=(2, 3))), ("bias", mx.random.uniform(shape=(2,)))]
        self.mlx_bit_linear_with_bias.load_weights(weights_with_bias)

        self.torch_bit_linear_with_bias = TorchBitLinear(3, 2, bias=True, weight_bits=1, input_bits=8).cpu()
        torch_weights_with_bias = [("weight", torch_nn.Parameter(torch.tensor(np.array(weights_with_bias[0][1])).cpu())), ("bias", torch_nn.Parameter(torch.tensor(np.array(weights_with_bias[1][1])).cpu()))]
        self.torch_bit_linear_with_bias.load_state_dict(dict(torch_weights_with_bias))

    def test_output_tensor_comparison(self):
        mlx_output_tensor = self.mlx_bit_linear.forward(self.input_tensor)
        torch_output_tensor = self.torch_bit_linear.forward(self.torch_input_tensor)

        # Check if the output tensors are close enough
        self.assertTrue(torch.allclose(torch_output_tensor, torch.tensor(np.array(mlx_output_tensor)), atol=1e-6), "Output tensors do not match.")

    def test_output_tensor_comparison_with_bias(self):
        mlx_output_tensor_with_bias = self.mlx_bit_linear_with_bias.forward(self.input_tensor)
        torch_output_tensor_with_bias = self.torch_bit_linear_with_bias.forward(self.torch_input_tensor)

        # Check if the output tensors with bias are close enough
        self.assertTrue(torch.allclose(torch_output_tensor_with_bias, torch.tensor(np.array(mlx_output_tensor_with_bias)), atol=1e-6), "Output tensors with bias do not match.")
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

        # Adjusting the shape of position_ids to match the expected input in mlx_bitnet.py
        self.position_ids = mx.arange(0, self.max_position_embeddings).reshape(-1, 1)

        # Adjusting the shape of torch_position_ids to match the expected input in mlx_bitnet.py
        self.torch_position_ids = torch.arange(0, self.max_position_embeddings).reshape(-1, 1)

        # Creating dummy input tensors for both frameworks to pass into the forward method along with position_ids
        self.dummy_input_mx = mx.zeros((1, self.dim))
        self.dummy_input_torch = torch.zeros((1, self.dim))

    def test_rotary_embedding_output_comparison(self):
        torch_cos, torch_sin = self.torch_rotary_embedding.forward(self.dummy_input_torch, self.torch_position_ids)
        mlx_cos, mlx_sin = self.mlx_rotary_embedding.forward(self.dummy_input_mx, self.position_ids)

        # Check if the cosine embeddings are close enough
        self.assertTrue(torch.allclose(torch_cos, torch.tensor(np.array(mlx_cos)), atol=1e-4), "Cosine embeddings do not match.")

        # Check if the sine embeddings are close enough
        self.assertTrue(torch.allclose(torch_sin, torch.tensor(np.array(mlx_sin)), atol=1e-4), "Sine embeddings do not match.")
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
        mlx_expanded = mx.broadcast_to(mlx_tensor, (mlx_tensor.shape[0], 4))  # MXNet does not support -1 in broadcast_to directly

        # Check if the expanded tensors are equal
        self.assertTrue(torch.equal(torch_expanded, torch.tensor(np.array(mlx_expanded))), "Negative expanded tensors do not match.")
class TestTensorTransposeInterop(unittest.TestCase):
    def test_tensor_transpose(self):
        # Create a tensor in both frameworks with shape (8, 2, 1)
        torch_tensor = torch.randn(8, 2, 1)
        mlx_tensor = mx.array(torch_tensor.cpu().numpy())

        # Transpose the tensors
        torch_transposed = torch_tensor.transpose(1, 2)
        mlx_transposed = mlx_tensor.transpose((0, 2, 1))

        # Check if the transposed tensors have the same shape
        self.assertEqual(torch_transposed.shape, tuple(mlx_transposed.shape), "Transposed tensor shapes do not match.")

        # Check if the transposed tensors are equal
        self.assertTrue(torch.allclose(torch_transposed, torch.tensor(np.array(mlx_transposed))), "Transposed tensors do not match.")
class TestMLXTorchBitnetMLPInterop(unittest.TestCase):
    def setUp(self):
        self.config = MinimalBitnetConfig(
            hidden_size=128,
            intermediate_size=512,
            weight_bits=8,
            input_bits=8,
            rms_norm_eps=1e-6,
        )
        # Initialize both MLPs with the same configuration
        self.torch_mlp = TorchBitnetMLP(self.config)
        self.mlx_mlp = MLXBitnetMLP(self.config)

        gate_proj_weights = [
            ("weight", mx.array(self.torch_mlp.gate_proj.weight.detach().numpy()))
        ]
        if self.torch_mlp.gate_proj.bias:
            gate_proj_weights.append(("bias", mx.array(self.torch_mlp.gate_proj.bias.detach().numpy())))
        self.mlx_mlp.gate_proj.load_weights(gate_proj_weights)

        up_proj_weights = [
            ("weight", mx.array(self.torch_mlp.up_proj.weight.detach().numpy()))
        ]
        if self.torch_mlp.up_proj.bias:
            up_proj_weights.append(("bias", mx.array(self.torch_mlp.up_proj.bias.detach().numpy())))
        self.mlx_mlp.up_proj.load_weights(up_proj_weights)

        down_proj_weights = [
            ("weight", mx.array(self.torch_mlp.down_proj.weight.detach().numpy()))
        ]
        if self.torch_mlp.down_proj.bias:
            down_proj_weights.append(("bias", mx.array(self.torch_mlp.down_proj.bias.detach().numpy())))
        self.mlx_mlp.down_proj.load_weights(down_proj_weights)

        self.dummy_input_torch = torch.randn(1, 128)
        self.dummy_input_mlx = mx.array(self.dummy_input_torch.cpu().detach().numpy())

    def test_mlp_output_comparison(self):
        torch_output = self.torch_mlp(self.dummy_input_torch)
        mlx_output = self.mlx_mlp.forward(self.dummy_input_mlx)

        # Check if the outputs are close enough
        self.assertTrue(torch.allclose(torch_output, torch.tensor(np.array(mlx_output)), atol=1e-4), "MLP outputs do not match.")
class TestSiluInterop(unittest.TestCase):
    def test_silu_function(self):
        # Create a tensor with random values
        torch_tensor = torch.randn(10, 10)
        mlx_tensor = mx.array(torch_tensor.cpu().numpy())

        # Apply SiLU activation function using both frameworks
        torch_silu_result = torch_silu(torch_tensor)
        mlx_silu_result = mx_nn.silu(mlx_tensor)

        # Convert MXNet result to Torch tensor for comparison
        mlx_silu_result_torch = torch.tensor(np.array(mlx_silu_result))

        # Check if the results are close enough
        self.assertTrue(torch.allclose(torch_silu_result, mlx_silu_result_torch, atol=1e-6), "SiLU results do not match.")
class TestBitnetAttentionInterop(unittest.TestCase):
    def setUp(self):
        self.config = MinimalBitnetConfig(
            hidden_size=128,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=512,
            attention_bias=True,
            weight_bits=1,
            input_bits=8,
            intermediate_size=512,
            pad_token_id=0,
            rms_norm_eps=1e-6,
            rope_theta=1.0,
        )
        self.torch_attention = TorchBitnetAttention(self.config)
        self.mlx_attention = MLXBitnetAttention(self.config)

        q_proj_weights = [
            ("weight", mx.array(self.torch_attention.q_proj.weight.detach().numpy()))
        ]
        if hasattr(self.torch_attention.q_proj, 'bias') and self.torch_attention.q_proj.bias is not None:
            q_proj_weights.append(("bias", mx.array(self.torch_attention.q_proj.bias.detach().numpy())))
        self.mlx_attention.q_proj.load_weights(q_proj_weights)
        k_proj_weights = [
            ("weight", mx.array(self.torch_attention.k_proj.weight.detach().numpy()))
        ]
        if hasattr(self.torch_attention.k_proj, 'bias') and self.torch_attention.k_proj.bias is not None:
            k_proj_weights.append(("bias", mx.array(self.torch_attention.k_proj.bias.detach().numpy())))
        self.mlx_attention.k_proj.load_weights(k_proj_weights)

        v_proj_weights = [
            ("weight", mx.array(self.torch_attention.v_proj.weight.detach().numpy()))
        ]
        if hasattr(self.torch_attention.v_proj, 'bias') and self.torch_attention.v_proj.bias is not None:
            v_proj_weights.append(("bias", mx.array(self.torch_attention.v_proj.bias.detach().numpy())))
        self.mlx_attention.v_proj.load_weights(v_proj_weights)

        o_proj_weights = [
            ("weight", mx.array(self.torch_attention.o_proj.weight.detach().numpy()))
        ]
        if hasattr(self.torch_attention.o_proj, 'bias') and self.torch_attention.o_proj.bias is not None:
            o_proj_weights.append(("bias", mx.array(self.torch_attention.o_proj.bias.detach().numpy())))
        self.mlx_attention.o_proj.load_weights(o_proj_weights)

        self.dummy_input_torch = torch.randn(1, 64, 128)
        self.dummy_input_mlx = mx.array(self.dummy_input_torch.cpu().detach().numpy())
        self.dummy_position_ids = torch.arange(0, 64).unsqueeze(0)

    def test_attention_output_comparison(self):
        torch_output, _, _ = self.torch_attention(
            self.dummy_input_torch,
            position_ids=self.dummy_position_ids
        )
        mlx_output, _, _ = self.mlx_attention.forward(
            self.dummy_input_mlx,
            position_ids=mx.array(self.dummy_position_ids.cpu().numpy())
        )

        # Check if the outputs are close enough
        self.assertTrue(torch.allclose(torch_output, torch.tensor(np.array(mlx_output)), atol=1e-4), "Attention outputs do not match.")






if __name__ == '__main__':
    unittest.main()
