import math
import argparse
import torch
import random
import time
from transformers import GenerationConfig

from minimal_bitnet import BitnetForCausalLM
# from modeling_bitnet import BitnetForCausalLM
from tokenization_bitnet import BitnetTokenizer 

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--hf_path', default='1bitLLM/bitnet_b1_58-3B', type=str)
parser.add_argument('--prompt', default='Elon musk is', type=str)
parser.add_argument('--maxlen', default=20, type=int)

def generate_text(model, tokenizer, prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.lm_head.weight.device)
    # Generate cos and sin values
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    # position_embeddings = model.embed_positions(position_ids)
    # cos = position_embeddings[:, :, 0::2].cos()
    # sin = position_embeddings[:, :, 1::2].sin()

    generation_config = GenerationConfig(
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
    )

    start_time = time.time()
    output_ids = model.generate(input_ids, generation_config=generation_config)
    # output_ids = model.generate(input_ids, generation_config=generation_config, cos=cos, sin=sin)
    end_time = time.time()

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    generation_time = end_time - start_time
    num_tokens = len(output_ids[0])
    tokens_per_second = num_tokens / generation_time

    print(f"Generated {num_tokens} tokens in {generation_time:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}")

    return generated_text


def main(args):
    # datasets = ['c4', 'wikitext2']

    model = BitnetForCausalLM.from_pretrained(
        args.hf_path,
        device_map='auto',
        low_cpu_mem_usage=True, 
        use_flash_attention_2=False,
        torch_dtype=torch.float16,
    ).half()
    tokenizer = BitnetTokenizer.from_pretrained(args.hf_path, use_fast=False)


    generated_text = generate_text(model, tokenizer, args.prompt, args.maxlen)
    print(generated_text)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
