# ImageNet Training
Pytorch code for training imagenet with fp16

1. Install pytorch,torchvision
2. Install apex
```
conda install -c conda-forge nvidia-apex)
```
3. (optional) install speedups:
```
conda install -c thomasbrandon -c defaults -c conda-forge pillow-accel-avx2
conda install -c conda-forge libjpeg-turbo
```

usage example:
```
python3 train_imagenet.py -t /intel_nvme/imagenet_data/train/ -v  /intel_nvme/imagenet_data/val/ -a xception -b 256 -2=--fp16
```
