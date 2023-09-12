import argparse
import time
import os

import transformers as tfs
import torch
from chat_downloader import ChatDownloader

DEBUG=False

# This is the trigger word that the model is trained for. When passing a different
# trigger, this replaces the user's custom trigger during inference, but does
# not appear in the final output.
MODEL_TRIGGER="<YCR>:"

def cid_to_url(channelid) -> str:
    return 'https://www.youtube.com/{}'.format(channelid)

class StopAtTok(tfs.StoppingCriteria):
    def __init__(self, stoptok):
        self.stoptok = stoptok["input_ids"]

    def __call__(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        return input_ids.flatten()[-1] == self.stoptok.flatten()

def infer(opt):
    print("Loading model for inference.")
    tokenizer = tfs.AutoTokenizer.from_pretrained(
        opt.modelname,
    )
    tokenizer.pad_token = tokenizer.eos_token

    config = tfs.AutoConfig.from_pretrained(opt.modelname)
    config.use_cache=True

    model = tfs.AutoModelForCausalLM.from_pretrained(
        opt.modelname,
        config=config
    ).to(opt.device)
    model.eval()

    generation_cfg = tfs.GenerationConfig(
        do_sample=True,
        eos_token_id=model.config.eos_token_id,
        bos_token_id=model.config.bos_token_id,
        pad_token_id=model.config.eos_token_id,
        use_cache=True,
        max_new_tokens=(opt.blocksize),
        temperature=opt.temperature,
        top_k=opt.top_k,
        top_p=opt.top_p,
        repetition_penalty=opt.repetition_penalty,
        length_penalty=1.0,
        num_return_sequences=1
    )

    stopper = StopAtTok(tokenizer("\n", return_tensors='pt').to(opt.device))

    if DEBUG:
        print("Loaded model.")

    dl = ChatDownloader()
    try:
        chat = dl.get_chat(
            url = cid_to_url(opt.channel)
            # output = outname,
        )
    except Exception as e:
        print("Failed to create chat downloader because of", e)
        exit(-1)

    output = []
    last_response = time.time()
    with torch.no_grad():
        for m in chat:
            on_cooldown = (time.time() - last_response) < opt.cooldown
            if DEBUG and on_cooldown:
                print(
                    "Skipping this message because it arrived during the cooldown period: {}"
                    .format(m['message'])
                )
            if m['message'].startswith(opt.trigger) and (not on_cooldown):
                if DEBUG:
                    print(
                        "==> Will respond to {} <=="
                        .format(m['message'])
                    )
                last_response = time.time()
                stripped = m['message'].strip(opt.trigger)
                prompt = MODEL_TRIGGER + stripped

                inputs = tokenizer(prompt, return_tensors='pt', truncation=False).to(opt.device)
                prompt_len_tok = inputs['input_ids'].shape[-1]
                logits = model.generate(
                    **inputs,
                    max_new_tokens=(opt.blocksize - prompt_len_tok),
                    generation_config=generation_cfg,
                    stopping_criteria=[stopper]
                )
                output = tokenizer.batch_decode(logits)[0]

                os.system('cls' if os.name == 'nt' else 'clear')
                print(output.replace(MODEL_TRIGGER, "\033[1mSnortana🐽💨\033[0m: "))

            else:
                if DEBUG:
                    print(
                        "Skipped \"{}\" because the trigger word wasn't found"
                        .format(m['message'])
                    )
                continue


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
        "-r", "--repetition_penalty",
        type=float,
        help="Repetition penalty to use for generation.",
        default=1.05
    )
    parser.add_argument(
        "--top_k",
        type=int,
        help="Value to use for topk sampling during generation.",
        default=50,
    )
    parser.add_argument(
        "-b", "--blocksize",
        type=int,
        help="Generate up to this many tokens.",
        default=256,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        help="Value to use for topp sampling during generation.",
        default=1.0,
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        help="Device to use for inference. Choose cpu or cuda.",
        default="cpu"
    )
    parser.add_argument(
        '-n', '--modelname',
        type=str,
        help="Name of the model to load for inference. Can be a patch, or HF hub name.",
        default="ycr-chat"
    )
    parser.add_argument(
        "--trigger",
        type=str,
        default="<YCR>:",
        help="Model responds when it sees this phrase"
    )
    parser.add_argument(
        "--channel",
        type=str,
        help="Live chat to watch."
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=30,
        help="Time in seconds to cooldown after generating a message."
             "If a message with the trigger arrives during cooldown, it is dropped."
    )
    opt = parser.parse_args()
    output = infer(opt)
    for ix, v in enumerate(output):
        fname = "out.txt"
        with open(fname, 'w') as f:
            f.write(v)
