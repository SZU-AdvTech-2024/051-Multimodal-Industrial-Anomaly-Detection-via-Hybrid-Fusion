# 基于混合融合记忆库的多模态3D点云异常检测复现



## 1.环境配置

```bash
pip install -r requirement.txt
# install knn_cuda
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
# install pointnet2_ops_lib
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

我的环境参考：

```txt
Ubuntu 16.04 
Python 3.8.20
torch 1.9.0+cu111
torchvision 0.10.0+cu111
torchaudio 0.9.0
numpy 1.24.4
imageio 2.35.1
```

## 2.数据集下载

MVTec 3D-AD下载地址：https://www.mvtec.com/company/research/datasets/mvtec-3d-ad

下载好数据集后，放入dataset文件夹下

## 3. 数据预处理

运行数据预处理代码preprocessing.py

```shell
python utils/preprocessing.py datasets/mvtec3d/
```

## 4. 预训练模型下载

| Backbone          | Pretrain Method                                              |
| ----------------- | ------------------------------------------------------------ |
| Point Transformer | [Point-MAE](https://drive.google.com/file/d/1-wlRIz0GM8o6BuPTJz4kTt6c_z1Gh6LX/view?usp=sharing) |
| ViT-b/8           | [DINO](https://drive.google.com/file/d/17s6lwfxwG_nf1td6LXunL-LjRaX67iyK/view?usp=sharing) |
| ViT-b/8           | [Supervised ImageNet 1K](https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz) |
| ViT-b/8           | [Supervised ImageNet 21K](https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz) |
| ViT-s/8           | [DINO](https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth) |
| UFF               | [UFF Module](https://drive.google.com/file/d/1Z2AkfPqenJEv-IdWhVdRcvVQAsJC4DxW/view?usp=sharing) |

## 5. 模型测试与训练

* 训练测试DINO+Point_MAE版本（即是用两个记忆库：RGB特征记忆库、点云特征记忆库），save_feature:存储提取的特征，提取到的特征存储到datasets/patch_lib文件夹中，为后续的UFF模块的训练做准备

  ```bash
  mkdir -p datasets/patch_lib
  python3 main.py \
  --method_name DINO+Point_MAE \
  --memory_bank multiple \
  --save_feature
  ```

* 训练无监督特征融合UFF模块，训练好的融合模型保存到checkpoint文件夹中

  ```bash
  OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=1 fusion_pretrain.py    \
  --accum_iter 16 \
  --epoch 5 \
  --lr 0.003 \
  --batch_size 16 \
  --data_path datasets/patch_lib \
  --output_dir checkpoints
  ```

* 训练并测试整个M3DM框架，即使用三个记忆库，use_uff即表示使用融合的特征记忆库

  ```bash
  python3 main.py \
  --method_name DINO+Point_MAE+Fusion \
  --use_uff \
  --memory_bank multiple \
  --fusion_module_path checkpoints/{FUSION_CHECKPOINT}.pth
  ```

* 若仅使用单个记忆库，即RGB特征记忆库或者点云特征记忆库,则method_name为DINO 或 Point_MAE, memory_bank指定为single

