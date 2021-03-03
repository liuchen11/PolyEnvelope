## Introduction
This is the code for Paper: "Training Provably Robust Models by Polyhedral Envelope Regularization"

## Requirement

```
python       = 3.7
scipy       >= 1.2.1
pytorch     >= 1.2.0
torchvision >= 0.4.1
```

## Main Modules

Folder `kw` is based on [the code](https://github.com/locuslab/convex_adversarial) KW provided by [Kolter & Wong, 2018] on Github.

Folder `util` provides necessary supporting functions needed to run the experiments.

Folder `run` contains scripts running the experiments, you can type `python run/XXX.py -h` to get help about the commands.

* train_mlp.py/train_cnn.py: training FC1/CNN models, including plain training, adversarial training and training with PER regularization.
* train_mlp_ibp.py/train_cnn_ibp.py: training FC1/CNN models using IBP or CROWN-IBP
* certify_mlp.py/certify_cnn.py: certifying FC1/CNN models, including Fast-Lin/KW/CROWN, PEC, IBP and PGD.
* search_eps_mlp.py/search_eps_cnn.py: Search for optimal value of epsilon on data points for a given model.

## Examples

There is one example to run the code, replace the content between dollars by what you want.

MNIST, CNN, $l_\infty$ norm attack:

```
# Training
python run/train_cnn.py --dataset mnist --batch_size 100 --epochs 100 --in_size 28 --in_channel 1 --conv_kernels 4,4 --conv_strides 2,2 --conv_channels 16,32 --conv_pads 1,1 --hidden_dims 100 --out_dim 10 --optim name=adam,lr=0.001 --alpha name=jump,start_v=0.0064,power=2.0,min_jump_pt=0.0,jump_freq=20.0,pt_num=1000 --eps name=jump,start_v=0.0064,power=2.0,min_jump_pt=0.0,jump_freq=20.0,pt_num=1000 --gamma name=constant,start_v=0.03,pt_num=1000 --norm -1 --T 4 --bound_calc_per_batch 20 --at_per 1 --bound_est bound_quad --pixel_range 0,1 --gpu $GPU id$ --out_folder $FOLDER$ --model_name $NAME$ --model2load $MODEL2LOAD$

# Evaluation
python run/certify_cnn.py --dataset mnist --subset test --batch_size 100 --in_size 28 --in_channel 1 --conv_kernels 4,4 --conv_strides 2,2 --conv_channels 16,32 --conv_pads 1,1 --hidden_dims 100 --out_dim 10 --model2load $FOLDER$/$NAME$.ckpt --out_file $OUT_FILE$ --eps 0.1 --norm -1 --bound_est bound_quad --pixel_range 0,1 --certify_mode $per/kw/crown/pgd/ibp$ --gpu $GPU id$

# Search for Optimal Epsilon
python run/search_eps_cnn.py --dataset mnist --subset test --in_size 28 --in_channel 1 --conv_kernels 4,4 --conv_strides 2,2 --conv_channels 16,32 --conv_pads 1,1 --hidden_dims 100 --out_dim 10 --model2load $FOLDER$/$NAME$.ckpt --out_file $OUT_FILE$ --eps_range min=0.,max=0.4 --precision 0.0001 --norm -1 --bound_est bound_quad --pixel_range 0,1 --ceritfy_mode $crown/per$ --gpu $GPU id$
```

