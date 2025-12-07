# Super Resolution

SRResNet
python train_resnet.py --iterations 100000

SRGAN
python train_srgan.py --iterations 100000 --warmup_model ./checkpoints/srresnet_66.pth.tar --warmup_model ./checkpoints/srresnet_66.pth.tar

SRRaNet
python train_srgan.py --iterations 100000 --exclude_activation --loss_strategy relativistic  --exclude_bn --psnr_mode

SRRaGAN
python train_srgan.py --iterations 200000 --exclude_activation --loss_strategy relativistic --exclude_bn --checkpoint ./checkpoints/srragan_psnr_nobn_66.pth.tar
