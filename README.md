# YouCompleteReLM
Text completion model to generate DSP Corpus style messages with decoder style LMs

# Quickstart with Google Colab
You can easily generate text with this notebook without installing anything using [Google Colab](https://colab.research.google.com/github/oddlyshapedfn/YouCompleteReLM/blob/main/YCR.ipynb)!
You must be signed in to execute code on Colab, but you can always view it.

Although Colab notebooks can store data in your G Drive (after asking for permission), this notebook does not,
so you will need to rerun all the setup if you leave and come back. During setup, the model weights are
downloaded to the Colab instance storage, and are lost when you leave.

# Installation
### If using Windows, please see the [Windows instructions](WINDOWS_SETUP.md)
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
Setting `gradient_checkpointing` to `true` also significantly reduces VRAM usage but slows down training.
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
### Stream Integration
The `infer_with_chat.py` script uses `chat_downloader` to watch a stream chat for the program's trigger word, than it responds by using the most recent message as a prompt.
Use like:
```
python infer_with_chat.py --channel @ChannelName --trigger "" -d cuda -n ModelName --cooldown 10 --repetition_penalty=1.15 --top_p=0.5 --top_k=50 -t 0.90 --blocksize=128
```
The same generation options are available as on most text generation tools.
