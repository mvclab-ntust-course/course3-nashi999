# LoRa Training

## Set up environment

### CUDA

I'm using cuda 12.1, if don't have cuda toolkit, download it first

https://developer.nvidia.com/cuda-12-1-0-download-archive

### Conda virtual python environment

create a virtual environment with python version 3.10

```powershell
conda create -n <env_name> python=3.10
```

activate it

```powershell
conda activate <env_name>
```

### Clone and setup diffusers from github

```powershell
python -m pip install git+https://github.com/huggingface/diffusers
```

and setup

```powershell
cd diffusers
python -m pip install e .
```

and install requirements

```powershell
cd /example/text_to_image
python -m pip install -r requirements.txt
```

and might need to install xformers

```powershell
python -m pip install xformers
```

### Create you own dataset

create a directory to store data

```powershell
mkdir data
```

put 10~15 images which you want to generate with

and write a metadata.jsonl

```json
{"file_name": "0001.png", "text": "image content"}
{"file_name": "0002.png", "text": "image content"}
...
```

### Run training program

you can find various argument on the internet or use

```powershell
python train_text_to_image_lora.py -h
```

to see what argument you want to use

and here is how I train my model

```powershell
python train_text_to_image_lora.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --dataset_name="..\..\data\hamster" --dataloader_num_workers=0 --resolution=512 --center_crop --random_flip --train_batch_size=1 --gradient_accumulation_steps=4 --max_train_steps=240 --learning_rate=1e-04 --max_grad_norm=1 --lr_scheduler="cosine" --lr_warmup_steps=0 --output_dir="..\..\output" --report_to=wandb --use_8bit_adam --adam_beta1=0.9 --adam_weight_decay=1e-2 --validation_prompt="hamster" --seed=1337
```

### Bug solutions

#### torch

if torch encounter an error because of version difference

can install another version from https://pytorch.org/get-started/locally/

```powershell
python -m pip uninstall torch
python -m pip install pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### accrlerate

if accelerate encounter an error, you can downgrade it to version 0.18.0

```powershell
python -m pip uninstall accelerate
python -m pip install accelerate==0.18.0
```

#### triton

if no module named 'triton' but you can't install it from pip

you need to install .whl file from https://huggingface.co/r4ziel/xformers_pre_built/blob/main/triton-2.0.0-cp310-cp310-win_amd64.whl

and

```powershell
python -m pip install triton-2.0.0-cp310-cp310-win_amd64.whl
```

#### JITFunction.__init__() got an unexpected keyword argument 'debug'

if you encounter an error says " JITFunction.__init__() got an unexpected keyword argument 'debug'"

you might need to go into the file anaconda3\envs\ `<env_name>`\Lib\site-packages\triton\runtime\jit.py

```python
...
class JITFunction(KernelInterface):
...
  def __init__(self, fn, version=None, do_not_specialize=None, debug=False):
                                                               ^^^^^^^^^^^

```

add debug=False into "init" function
