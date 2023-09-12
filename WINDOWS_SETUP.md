# Setup on Windows
1. Install miniconda (https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)

2. Launch a terminal with the "Anaconda Prompt (miniconda3)" from the start menu

3. Create a new environment for YouCompleteRe. This only needs to be done the first time.
`conda create -n ycr python=3.10 pip transformers pytorch torchvision torchaudio cpuonly -c pytorch`
    * If you have an NVIDIA GPU and want to use it speed up text generation, do this instead: `conda create -n ycr python=3.10 pip transformers pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`

4. Activate the environment. This needs to be done every time.
`conda activate ycr`

5. Navigate to the location of infer.py. For example, if infer.py is in your Downloads folder...
`cd C:\Users\REPLACEWITHYOURUSERNAME\Downloads`

6. Run model inference on the CPU
`python infer.py -n oddlyshapedfn/YouCompleteRe`

# Notes
* On future runs, only do steps 2, 4, 5, 6
* To remove miniconda, run the uninstaller like any other windows program.
* To delete the model files, remove `C:\Users\REPLACEWITHYOURUSERNAME\.cache\huggingface\hub`
* At no point should this procedure require administrator privileges.
