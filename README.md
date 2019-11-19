# imagenet_training
Pytorch code for training imagenet with fp16

1. Install pytorch,torchvision
2. Install apex (conda install -c conda-forge nvidia-apex)

usage example:

python3 train_imagenet.py -t /intel_nvme/imagenet_data/train/ -v  /intel_nvme/imagenet_data/val/ -a xception -b 256 -2=--fp16
