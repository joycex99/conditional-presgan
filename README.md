# CS 236 Entropy Regularized Conditional GANs

*Example usage*

Running conditional gan (train_cond.py) with regularization term 0.0002

`$ python main.py --dataset cifar10 --model presgan --lambda 0.0002`

Files: hmc.py -> HMC Sampling, train_cond.py -> Main conditional gan training with nets_cond.py architecture.

![Saturation](https://raw.githubusercontent.com/evazhang612/gan-results/master/cifar10_00005/presgan_cifar10_fake_epoch_077.png)
