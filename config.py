from easydict import EasyDict as edict
from torchvision import transforms

config = edict()
config.dataset = "hagrid" # training dataset
config.embedding_size = 512 # embedding size of evaluation
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128 # batch size per GPU
config.lr = 0.1
config.global_step=0 # step to resume
config.s=64.0
config.m=0.50
config.beta=0.5

# type of network to train [ efficientnet | densenet ]
config.network = "efficientnet"

if config.dataset == "hagrid":
    config.rec = "/home/fbi1532/Databases/HAND/HaGRID/Images/train"
    config.num_classes = 31791 #200
    config.num_image = 503871 #69705
    config.num_epoch = 34   #  [22, 30, 35] [22, 30, 40]
    config.warmup_epoch = -1
    config.val_targets = '/home/fbi1532/Databases/HAND/HaGRID/Images/test'
    config.eval_step= 1510 #2055 #33350

    config.transform = transforms.Compose([
                    transforms.Resize((324, 324)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [20, 28, 32] if m - 1 <= epoch])
    config.lr_func = lr_step_func


