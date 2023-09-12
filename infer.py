import json
import sys
import re
import argparse
import os

import transformers as tfs
import torch

class StopAtTok(tfs.StoppingCriteria):
    def __init__(self, stoptok):
        self.stoptok = stoptok["input_ids"]

    def __call__(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        return input_ids.flatten()[-1] == self.stoptok.flatten()

def infer(opt, cfg):
    print("Loading model for inference.")
    tokenizer = tfs.AutoTokenizer.from_pretrained(
        opt.modelname,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Need to manually enable use_cache. It must be disabled for training if
    # gradient_checkpointing is enabled, but if it's disabled, it slows generation
    config = tfs.AutoConfig.from_pretrained(opt.modelname)
    config.use_cache=True

    model = tfs.AutoModelForCausalLM.from_pretrained(
        opt.modelname,
        config=config
    ).to(cfg['device'])
    model.eval()

    prompt=""
    if os.path.exists(opt.prompt):
        print("Loading prompt from a file")
        with open(opt.prompt, 'r') as f:
            prompt = f.read().rstrip()
    else:
        print("Prompt does not look like a file, treating it as a string.")
        prompt = opt.prompt

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(cfg['device'])
    inputs = inputs.to(cfg['device'])
    prompt_len_tok = inputs['input_ids'].shape[-1]

    print("Prompt has size {}, leaving {} tokens for generation".format(
        prompt_len_tok,
        cfg['blocksize'] - prompt_len_tok
    ))

    generation_cfg = tfs.GenerationConfig(
        do_sample=True,
        eos_token_id=model.config.eos_token_id,
        bos_token_id=model.config.bos_token_id,
        pad_token_id=model.config.eos_token_id,
        use_cache=True,
        max_new_tokens=(cfg['blocksize'] - prompt_len_tok),
        temperature=opt.temperature,
        top_k=opt.top_k,
        top_p=opt.top_p,
        repetition_penalty=1.05,
        length_penalty=1.0,
        num_return_sequences=1
    )

    # Stop generation when a newline is hit
    stopper = StopAtTok(tokenizer("\n", return_tensors='pt').to(cfg["device"]))

    print("Loaded model.")
    output = []
    with torch.no_grad():
        logits = model.generate(
            **inputs,
            generation_config=generation_cfg,
            stopping_criteria=[stopper]
        )
        output = tokenizer.batch_decode(logits)

    print("\n".join(output))
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate text with a trained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        help="Temperature to use for generation.",
        default=0.5
    )
    parser.add_argument(
        "--top_k",
        type=int,
        help="Value to use for topk sampling during generation.",
        default=50,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        help="Value to use for topp sampling during generation.",
        default=1.0,
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to a configuration file. Pass the same thing that was given to training.",
        default="train_cfg.json"
    )
    parser.add_argument(
        '-n', '--modelname',
        type=str,
        help="Name of the model to load for inference. Can be a patch, or HF hub name.",
        default="ycr-chat"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default="<YCR>:",
        help="Prompt. Either a raw string, or the filename"
    )
    opt = parser.parse_args()

    with open(opt.config, 'r') as f:
        cfg = json.load(f)
        print("Using config", cfg)

    output = infer(opt, cfg)
    for ix, v in enumerate(output):
        fname = "out.txt"
        with open(fname, 'w') as f:
            f.write(v)
