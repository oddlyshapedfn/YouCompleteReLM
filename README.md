# YouCompleteReLM
Text completion model to generate DSP Corpus style messages with decoder style LMs

# Installation
```
pip install transformers datasets accelerate deepspeed
```
If training your own weights, the default configuration assumes you have an RTX 30-series
or newer. If you have a different GPU, you will need to tweak certain parameters in
train.py, such as toggling bf16 or tf32.

Training assumes you're using Linux. It may be possible to train in WSL, but
`deepspeed` will likely need some manual patches, and WSL will need to be configured
to allow significantly more memory in its VM.

# Training
```
accelerate launch --config_file accel_config.yaml train.py
```
`train.py` assumes certain paths that match this repository. Use `train.py --help`
if your paths deviate from its default configuration.

Inside of `train_cfg.json`, you can select various parameters for training. For instance,
you can use a different base model if your machine has a lower VRAM/DRAM capacity.
Decreasing blocksize, batchsize, or switching to a smaller model reduces RAM usage.
Setting `gradient_checkpoint` to `true` also significantly reduces RAM usage but slows down training.
See below about `gradient_accumulation_steps`.
For reasonable responses, I recommend using `EleutherAI/pythia-410m-deduped` or larger.

`accel_config.yaml` configures deepspeed, and will probably not need any changes.
However the value of `gradient_accumulation_steps` must match the value in `train_cfg.json`.
Deepspeed is practically required to train LLMs of a useful size on single GPU workstations
due to VRAM limitations of gaming GPUs. Deepspeed offloads certain tensors to main DRAM, so
your system must also have sufficient DRAM.

# Inference
Inference can run on just about any machine, even if a GPU is not available.
Once you have a trained model in `ycr-chat`, you can just run `python infer.py`
See `python infer.py --help` for sampling options too.
### Pretrained
If you do not wish to train the model yourself, you can use my pretrained weights like this:
```
python infer.py -p "<YCR>:My detractors are" -n oddlyshapedfn/YouCompleteRe
```
This downloads the weights from Huggingface Hub to your computer. They are quite large, so if you
want to delete them later, they'll be stored in `~/.cache/huggingface/hub` (On linux)

### Text Completion
To complete existing text, either create a file with your prompt or pass it as plaintext
on the commandline with `python infer.py -p <file or text>`.
This model responds best to the phrase `<YCR>: <your message>`, so start your text with
that for best results.
Example:
```
# Input
python infer.py -p "<YCR>:My detractors are"

# Response (Truncated)
<YCR>:My detractors are idiots, ive seen it myself. They literally just post up threads like this, derail the thread and then go on to make new ones of their own, completely changing the topic of the thread. Its sooo fucking annoying. Ive had to delete about a dozen posts in the last 24 hours because they were just too fucking annoying and derailing the thread. So if you hafta know, keep it on-topic and away from the personal attacks. Thanks.  Oh yeah, and by the way, theres a reason im not going to be at ECC: because im not going to waste my time with idiots
```
