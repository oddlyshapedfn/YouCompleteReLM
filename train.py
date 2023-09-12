import json
import argparse

import torch
import transformers as tfs
import datasets as ds


class JsonlGenerator:
    def __init__(self, cfg, tok, fname):
        with open(fname, 'r') as f:
            lines = f.readlines()
            self.text = lines

        self.cfg = cfg
        self.tokenizer = tok

    def _generate(self):
        for v in self.text:
            x = json.loads(v)
            yield x

    def tokenize(self, x):
        text = self.tokenizer(x['text'], truncation=False)

        return text

    def split_oversized_t(self, x):
        blocksize = self.cfg['blocksize']

        new_text = []
        new_mask = []

        for line, mask in zip(x['input_ids'], x['attention_mask']):
            if len(line) >= blocksize:
                total_length = (len(line) // blocksize) * blocksize
                for ix in range(0, total_length, blocksize):
                    new_text.append(line[ix: ix + blocksize])
                    new_mask.append(mask[ix: ix + blocksize])
            else:
                new_text.append(line)
                new_mask.append(mask)


        x['input_ids'] = new_text
        x['attention_mask'] = new_mask
        return x

    def get_dsets(self):
        raw_data = ds.Dataset.from_generator(self._generate)
        dset = raw_data.map(
            self.tokenize,
            remove_columns=['metadata', 'text'],
            batched=True,
        )
        dset = dset.map(
            self.split_oversized_t,
            batched=True
        ).shuffle()

        print("Processed dataset len", len(dset))
        self.dset_split = dset.train_test_split(test_size=0.10)
        self.tokenized_train = self.dset_split['train']
        self.tokenized_val = self.dset_split['test']
        print("Train: {}, Val: {}".format(len(self.tokenized_train), len(self.tokenized_val)))

        return self.tokenized_train, self.tokenized_val

class YouCompleteReModel:
    def __init__(self, cfg, tok):
        self.cfg = cfg
        self.tokenizer = tok

        config = tfs.AutoConfig.from_pretrained(
            cfg['model_name'],
        )
        config.use_cache = not cfg['gradient_checkpointing']
        config.attention_dropout = cfg['attention_dropout']

        self.model = tfs.AutoModelForCausalLM.from_pretrained(
            cfg['model_name'],
            config=config,
        ).to(self.cfg['device'])
        print("Model configuration:\n", self.model.config)

        self.collator = tfs.DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

    def train(self, dset_train, dset_val, resume=False):
        args = tfs.TrainingArguments(
            output_dir='./finetune',
            optim='adamw_torch',
            learning_rate=float(self.cfg['lr']),
            num_train_epochs=self.cfg['epochs'],
            per_device_train_batch_size=self.cfg['batchsize'],
            per_device_eval_batch_size=self.cfg['batchsize'],
            weight_decay=0.01,
            save_strategy='steps',
            save_steps=self.cfg['save_steps'],
            evaluation_strategy='steps',
            eval_steps=self.cfg['eval_steps'],
            push_to_hub=False,
            report_to='none',
            gradient_accumulation_steps=self.cfg['gradient_accumulation_steps'],
            gradient_checkpointing=self.cfg['gradient_checkpointing'],
            lr_scheduler_type=self.cfg['scheduler'],
            bf16=True,
            tf32=True,
        )
        # Thank you random person https://discuss.huggingface.co/t/training-loss-0-0-validation-loss-nan/27950
        self.trainer = tfs.Trainer(
            model=self.model,
            args=args,
            train_dataset=dset_train,
            eval_dataset=dset_val,
            tokenizer=self.tokenizer,
            data_collator=self.collator
        )
        self.trainer.train(resume_from_checkpoint=resume)

    def evaluate(self):
        import math
        eval_results = self.trainer.evaluate()
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True)
        inputs = inputs.to(self.cfg['device'])
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=1024,
                temperature=0.70,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
            )
        output = self.tokenizer.batch_decode(output)[0]
        print(output)

    def save(self, name):
        self.trainer.save_model(name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate text with a trained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="train_ycr_410m.json",
        help="Path to a configuration file. Pass the same thing that was given to training.",
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="ycr_dspcorpus.jsonl",
        help="Path to a .jsonl to train on.",
    )
    parser.add_argument(
        "-r", "--resume",
        action="store_true",
        help="Resume training.",
    )
    opt = parser.parse_args()

    with open(opt.config, 'r') as f:
        cfg = json.load(f)
        print("Using config", cfg)

    tokenizer = tfs.AutoTokenizer.from_pretrained(
        cfg['model_name'],
    )
    tokenizer.pad_token = tokenizer.eos_token

    t, v = JsonlGenerator(cfg, tokenizer, opt.dataset).get_dsets()
    mdl = YouCompleteReModel(cfg, tokenizer)
    mdl.train(t, v, opt.resume)
    mdl.save('ycr-chat')
    mdl.evaluate()
    mdl.predict("<YCR>:")
