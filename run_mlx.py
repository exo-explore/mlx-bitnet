import argparse
import mlx.core as mx
from mlx_bitnet import load_causal_model

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="1bitLLM/bitnet_b1_58-xl", type=str)
parser.add_argument("--prompt", default="The meaning of life is", type=str)
parser.add_argument("--max_tokens", default=50, type=int)
parser.add_argument("--temp", default=0.0, type=float)
args = parser.parse_args()

print(f"Loading model {args.model} ...")
model, tokenizer = load_causal_model(args.model)

tokens = tokenizer.encode(args.prompt)
input_ids = mx.array([tokens])
attention_mask = mx.ones_like(input_ids)

generated = list(tokens)
decoded_so_far = args.prompt
print(decoded_so_far, end="", flush=True)

for token in model.generate(input_ids, attention_mask, temp=args.temp):
    generated.append(token.item())
    new_text = tokenizer.decode(generated)
    print(new_text[len(decoded_so_far):], end="", flush=True)
    decoded_so_far = new_text
    if len(generated) - len(tokens) >= args.max_tokens:
        break

print()
