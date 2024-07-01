environment
```bash
mkvirtualenv dddr
pip install -r requirement.txt
```

prepare data
```bash
cd data
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
python preprocess.py
cd ..
```

download pretrained diffusion model
```bash
mkdir -p models/ldm/text2img-large
wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

Please download bert-base-uncased from huggingface, and put it in models/bert

Finetune
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --method finetune --tasks 10 --beta 0.5 --seed 2024
```

EWC
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --method ewc --tasks 10 --beta 0.5 --seed 2024
```

Target
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --method target --tasks 5  --beta 0.5 --seed 2024 --w_kd 25
```

DDDR
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --method ours --tasks 5 --beta 0.5 --seed 2024
```

reproduction
```bash
bash rep_run.sh
```



