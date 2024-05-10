import unittest
import mlx.core as mx
import mlx.nn as mx_nn
import torch
from torch import nn as torch_nn
import numpy as np
from mlx_bitnet import weight_quant as mlx_weight_quant
from mlx_bitnet import BitLinear as MLXBitLinear
from mlx_bitnet import BitnetRMSNorm as MLXBitnetRMSNorm
from mlx_bitnet import BitnetRotaryEmbedding as MLXBitnetRotaryEmbedding
from mlx_bitnet import BitnetMLP as MLXBitnetMLP
from mlx_bitnet import BitnetAttention as MLXBitnetAttention
from mlx_bitnet import MinimalBitnetConfig
from mlx_bitnet import BitnetDecoderLayer as MLXBitnetDecoderLayer
from mlx_bitnet import load_model, load_causal_model
from mlx_bitnet import BitnetModel as MLXBitnetModel
from mlx_bitnet import BitnetForCausalLM as MLXBitnetForCausalLM
from mlx_bitnet import BitnetTokenizer
from torch_bitnet import weight_quant as torch_weight_quant
from torch_bitnet import BitLinear as TorchBitLinear
from torch_bitnet import BitnetRMSNorm as TorchBitnetRMSNorm
from torch_bitnet import BitnetRotaryEmbedding as TorchBitnetRotaryEmbedding
from torch_bitnet import BitnetMLP as TorchBitnetMLP
from torch_bitnet import BitnetAttention as TorchBitnetAttention
from torch_bitnet import BitnetModel as TorchBitnetModel
from torch_bitnet import BitnetForCausalLM as TorchBitnetForCausalLM
from torch_bitnet import BitnetDecoderLayer as TorchBitnetDecoderLayer
from transformers.activations import silu as torch_silu
from training.bit_linear import weight_quant as bit_linear_weight_quant

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

    @unittest.skip("skip this test normally because its slow")
    def test_model_loading_comparison(self):
        model_name = "1bitLLM/bitnet_b1_58-large"
        torch_model = TorchBitnetModel.from_pretrained(model_name)
        mlx_model, _ = load_model(model_name)

        torch_embed_tokens_weight = torch_model.embed_tokens.weight.cpu().detach().numpy()
        mlx_embed_tokens_weight = np.array(mlx_model.embed_tokens.weight)
        self.assertTrue(np.allclose(torch_embed_tokens_weight, mlx_embed_tokens_weight, atol=1e-6), "Embed tokens weights do not match.")

        for layer_index in range(len(torch_model.layers)):
            torch_layer = torch_model.layers[layer_index]
            mlx_layer = mlx_model.layers[layer_index]

            # Check MLP up_proj weights
            torch_up_proj_weight = torch_layer.mlp.up_proj.weight.cpu().detach().numpy()
            mlx_up_proj_weight = np.array(mlx_layer.mlp.up_proj.weight)
            self.assertTrue(np.allclose(torch_up_proj_weight, mlx_up_proj_weight, atol=1e-6), f"Layer {layer_index} MLP up_proj weights do not match.")

            # Check MLP down_proj weights
            torch_down_proj_weight = torch_layer.mlp.down_proj.weight.cpu().detach().numpy()
            mlx_down_proj_weight = np.array(mlx_layer.mlp.down_proj.weight)
            self.assertTrue(np.allclose(torch_down_proj_weight, mlx_down_proj_weight, atol=1e-6), f"Layer {layer_index} MLP down_proj weights do not match.")

            # Check self attention q_proj weights
            torch_q_proj_weight = torch_layer.self_attn.q_proj.weight.cpu().detach().numpy()
            mlx_q_proj_weight = np.array(mlx_layer.self_attn.q_proj.weight)
            self.assertTrue(np.allclose(torch_q_proj_weight, mlx_q_proj_weight, atol=1e-6), f"Layer {layer_index} self attention q_proj weights do not match.")

            # Check self attention k_proj weights
            torch_k_proj_weight = torch_layer.self_attn.k_proj.weight.cpu().detach().numpy()
            mlx_k_proj_weight = np.array(mlx_layer.self_attn.k_proj.weight)
            self.assertTrue(np.allclose(torch_k_proj_weight, mlx_k_proj_weight, atol=1e-6), f"Layer {layer_index} self attention k_proj weights do not match.")

            # Check self attention v_proj weights
            torch_v_proj_weight = torch_layer.self_attn.v_proj.weight.cpu().detach().numpy()
            mlx_v_proj_weight = np.array(mlx_layer.self_attn.v_proj.weight)
            self.assertTrue(np.allclose(torch_v_proj_weight, mlx_v_proj_weight, atol=1e-6), f"Layer {layer_index} self attention v_proj weights do not match.")

            # Check self attention o_proj weights
            torch_o_proj_weight = torch_layer.self_attn.o_proj.weight.cpu().detach().numpy()
            mlx_o_proj_weight = np.array(mlx_layer.self_attn.o_proj.weight)
            self.assertTrue(np.allclose(torch_o_proj_weight, mlx_o_proj_weight, atol=1e-6), f"Layer {layer_index} self attention o_proj weights do not match.")

    @unittest.skip("skip this test normally because its slow")
    def test_bitnet_decoder_layer_interop(self):
        layer_norm_eps = 1e-6

        # Initialize MLXBitnetDecoderLayer
        config = MinimalBitnetConfig(
            hidden_size=2048,
            num_attention_heads=32,
            num_key_value_heads=32,
            rms_norm_eps=layer_norm_eps,
        )
        mlx_decoder_layer = MLXBitnetDecoderLayer(config=config, layer_idx=0)
        # Initialize TorchBitnetDecoderLayer
        torch_decoder_layer = TorchBitnetDecoderLayer(config=config, layer_idx=0).cpu()

        # Load weights and biases (if they exist) from MLX layer to Torch layer for comparison
        torch_decoder_layer.self_attn.q_proj.weight = torch.nn.Parameter(torch.tensor(np.array(mlx_decoder_layer.self_attn.q_proj.weight)).float().cpu())
        if hasattr(mlx_decoder_layer.self_attn.q_proj, 'bias') and mlx_decoder_layer.self_attn.q_proj.bias is not None:
            torch_decoder_layer.self_attn.q_proj.bias = torch.nn.Parameter(torch.tensor(np.array(mlx_decoder_layer.self_attn.q_proj.bias)).float().cpu())

        torch_decoder_layer.self_attn.k_proj.weight = torch.nn.Parameter(torch.tensor(np.array(mlx_decoder_layer.self_attn.k_proj.weight)).float().cpu())
        if hasattr(mlx_decoder_layer.self_attn.k_proj, 'bias') and mlx_decoder_layer.self_attn.k_proj.bias is not None:
            torch_decoder_layer.self_attn.k_proj.bias = torch.nn.Parameter(torch.tensor(np.array(mlx_decoder_layer.self_attn.k_proj.bias)).float().cpu())

        torch_decoder_layer.self_attn.v_proj.weight = torch.nn.Parameter(torch.tensor(np.array(mlx_decoder_layer.self_attn.v_proj.weight)).float().cpu())
        if hasattr(mlx_decoder_layer.self_attn.v_proj, 'bias') and mlx_decoder_layer.self_attn.v_proj.bias is not None:
            torch_decoder_layer.self_attn.v_proj.bias = torch.nn.Parameter(torch.tensor(np.array(mlx_decoder_layer.self_attn.v_proj.bias)).float().cpu())

        torch_decoder_layer.self_attn.o_proj.weight = torch.nn.Parameter(torch.tensor(np.array(mlx_decoder_layer.self_attn.o_proj.weight)).float().cpu())
        if hasattr(mlx_decoder_layer.self_attn.o_proj, 'bias') and mlx_decoder_layer.self_attn.o_proj.bias is not None:
            torch_decoder_layer.self_attn.o_proj.bias = torch.nn.Parameter(torch.tensor(np.array(mlx_decoder_layer.self_attn.o_proj.bias)).float().cpu())

        torch_decoder_layer.mlp.gate_proj.weight = torch.nn.Parameter(torch.tensor(np.array(mlx_decoder_layer.mlp.gate_proj.weight)).float().cpu())
        if hasattr(mlx_decoder_layer.mlp.gate_proj, 'bias') and mlx_decoder_layer.mlp.gate_proj.bias is not None:
            torch_decoder_layer.mlp.gate_proj.bias = torch.nn.Parameter(torch.tensor(np.array(mlx_decoder_layer.mlp.gate_proj.bias)).float().cpu())

        torch_decoder_layer.mlp.up_proj.weight = torch.nn.Parameter(torch.tensor(np.array(mlx_decoder_layer.mlp.up_proj.weight)).float().cpu())
        if hasattr(mlx_decoder_layer.mlp.up_proj, 'bias') and mlx_decoder_layer.mlp.up_proj.bias is not None:
            torch_decoder_layer.mlp.up_proj.bias = torch.nn.Parameter(torch.tensor(np.array(mlx_decoder_layer.mlp.up_proj.bias)).float().cpu())

        torch_decoder_layer.mlp.down_proj.weight = torch.nn.Parameter(torch.tensor(np.array(mlx_decoder_layer.mlp.down_proj.weight)).float().cpu())
        if hasattr(mlx_decoder_layer.mlp.down_proj, 'bias') and mlx_decoder_layer.mlp.down_proj.bias is not None:
            torch_decoder_layer.mlp.down_proj.bias = torch.nn.Parameter(torch.tensor(np.array(mlx_decoder_layer.mlp.down_proj.bias)).float().cpu())


        # Prepare input tensors
        input_tensor = mx.random.uniform(shape=(2, 50, 2048))
        torch_input_tensor = torch.tensor(np.array(input_tensor)).float().cpu()

        # Prepare attention mask
        attention_mask = mx.random.uniform(shape=(2, 1, 50, 50))
        torch_attention_mask = torch.tensor(np.array(attention_mask)).float().cpu()

        # Prepare position ids
        position_ids = mx.broadcast_to(mx.expand_dims(mx.arange(50), axis=0), (2, 50))
        torch_position_ids = torch.tensor(np.array(position_ids)).long().cpu()

        # Forward pass
        mlx_output = mlx_decoder_layer.forward(input_tensor, attention_mask=attention_mask, position_ids=position_ids)
        torch_output = torch_decoder_layer.forward(torch_input_tensor, attention_mask=torch_attention_mask, position_ids=torch_position_ids)

        # Check if the outputs are close enough
        self.assertTrue(torch.allclose(torch.tensor(np.array(mlx_output)), torch_output[0].detach(), atol=0.01), "Decoder layer outputs do not match.")


    @unittest.skip("skip this test normally because its slow")
    def test_model_inference_comparison(self):
        model_name = "1bitLLM/bitnet_b1_58-large"
        torch_model = TorchBitnetModel.from_pretrained(model_name)
        mlx_model, _ = load_model(model_name)

        # Prepare input
        input_ids = torch.tensor([[101, 102, 103, 104]])
        attention_mask = torch.tensor([[1, 1, 1, 1]])

        # Torch model inference
        with torch.no_grad():
            torch_output = torch_model(input_ids, attention_mask=attention_mask)

        # MLX model inference
        mlx_input_ids = mx.array(input_ids.numpy())
        mlx_attention_mask = mx.array(attention_mask.numpy())
        mlx_output = mlx_model.forward(mlx_input_ids, attention_mask=mlx_attention_mask)

        print("torch_output", torch_output)
        print("mlx_output", mlx_output)

        # Check if the outputs are close enough
        self.assertTrue(torch.allclose(torch_output[0], torch.tensor(np.array(mlx_output[0])), atol=1e-4), "Model inferences do not match.")

    @unittest.skip("skip this test normally because its slow")
    def test_generate_words_from_torch_bitnet_model(self):
        model_name = "1bitLLM/bitnet_b1_58-large"
        torch_model = TorchBitnetForCausalLM.from_pretrained(model_name)

        print("[torch] lm head weight", torch_model.lm_head.weight)

        # Prepare input
        prompt = "Capital of India is"
        tokenizer = BitnetTokenizer.from_pretrained(model_name)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        print(inputs)

        # Generate tokens using torch_model.generate
        generated_token_ids = torch_model.generate(input_ids, attention_mask=attention_mask, max_length=20)

        # Convert generated token IDs to words
        generated_words_list = tokenizer.decode(generated_token_ids[0], skip_special_tokens=True)

        # Check if words were generated
        self.assertTrue(len(generated_words_list) > 0, "No new words were generated.")
        print("Generated words:", generated_words_list)

    @unittest.skip("skip this test normally because its slow")
    def test_single_inference_from_mlx_bitnet_model(self):
        model_name = "1bitLLM/bitnet_b1_58-large"
        torch_model = TorchBitnetForCausalLM.from_pretrained(model_name)
        mlx_model, _ = load_causal_model(model_name)
        mlx_model.lm_head.load_weights([
            ("weight", mx.array(torch_model.lm_head.weight.detach().numpy()))
        ])

        print("[mlx] lm head weight", mlx_model.lm_head.weight)

        # Prepare input
        prompt = "Capital of India is"
        tokenizer = BitnetTokenizer.from_pretrained(model_name)
        inputs = tokenizer(prompt, return_tensors="pt")
        print("inputs", inputs)
        input_ids = mx.array(inputs.input_ids.numpy())
        attention_mask = mx.array(inputs.attention_mask.numpy())

        # Generate a single token using mlx_model.generate
        generated = mlx_model.forward(input_ids, attention_mask=attention_mask)

        print("generated", generated)

        # Convert logits to probabilities
        probabilities = mx.softmax(generated.logits, axis=-1)

        # Get the index of the maximum probability to find the next token ID
        next_token_ids = mx.argmax(probabilities, axis=-1)

        print("Next predicted token ID:", next_token_ids)
        print("Next predicted token:", tokenizer.decode(np.array(next_token_ids).tolist()[0]))  # Decode using the numpy array

    @unittest.skip("skip this test normally because its slow")
    def test_single_inference_with_generate_from_mlx_bitnet_model(self):
        model_name = "1bitLLM/bitnet_b1_58-large"
        torch_model = TorchBitnetForCausalLM.from_pretrained(model_name)

        mlx_model, _ = load_causal_model(model_name)
        mlx_model.lm_head.load_weights([
            ("weight", mx.array(torch_model.lm_head.weight.detach().numpy()))
        ])
        tokenizer = BitnetTokenizer.from_pretrained(model_name)

        prompt = "The capital of Scotland is "
        max_tokens = 50
        temp = 1.0

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = mx.array(inputs.input_ids.numpy())
        attention_mask = mx.array(inputs.attention_mask.numpy())
        tokens = []
        for token in mlx_model.generate(input_ids, attention_mask, temp):
            tokens.append(token)

            if len(tokens) == 1:
                # Actually perform the computation to measure the prompt processing time
                mx.eval(token)

            if len(tokens) >= max_tokens:
                break

            # It is perfectly ok to eval things we have already eval-ed.
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            print(s)
            # print(s, end="", flush=True)

        mx.eval(tokens)
        s = tokenizer.decode([t.item() for t in tokens])
        print(s, flush=True)

class TestWeightQuant(unittest.TestCase):
    def test_weight_quant_values(self):
        mx.random.seed(1)
        for _ in range(100):
            random_weights = mx.random.uniform(-2, 2, shape=(10, 10))  # Generate random weights
            mlx_quantized_weights = mlx_weight_quant(random_weights, num_bits=1)  # Quantize weights
            torch_quantized_weights = torch_weight_quant(torch.tensor(np.array(random_weights)), num_bits=1)  # Quantize weights
            for i in range(len(random_weights)):
                for j in range(len(random_weights[i])):
                    self.assertAlmostEqual(mlx_quantized_weights[i][j].item(), torch_quantized_weights[i][j].item(), places=4)




if __name__ == '__main__':
    unittest.main()
