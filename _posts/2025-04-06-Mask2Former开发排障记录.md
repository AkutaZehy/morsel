---
title: Mask2Former 开发排障记录
date: 2025-04-06 12:00:00 +0800
categories: [笔记, 开发]
tags: [笔记, 编程, 深度学习, CUDA, 实例分割, detectron, DEBUG]
description: 记录 Detectron2/Mask2Former 项目中的 CUDA、GPU 选择与预训练骨干适配问题。
---

前情提要：

[第一次使用Linux远程开发]({% post_url 2025-03-12-第一次使用Linux远程开发 %})

本来打算这一篇文章全部写完的，想想会太长，单独开一篇记录detectron2/mask2former框架开发踩的坑，把前面一则中与这个框架相关一点的也一并移到这边。

> 请善用导航，本文内容特别长。
{: .prompt-info }

后来发现了一个极为棘手的问题，请移步：

[Mask2Former 多标签改造：第一周排障记录]({% post_url 2025-04-12-Mask2Former多标签改造：第一周排障记录 %})

## 关于detectron2的一点碎碎念

看到服务器上的环境时，我对这个经典的 [Detectron2 框架](https://github.com/facebookresearch/detectron2)有些谨慎。

这次使用的是 [Mask2Former](https://github.com/facebookresearch/Mask2Former)，Detectron2 为其提供了基础框架。

这里Former就是Transformer，QKV那一套，本科的时候也接触过，当时是做弱监督用，然而也是时间长效果差最后也没做下去了。

这次能否顺利完成还不确定，因此我先从环境和最小流程开始逐项确认。

## 棘手的 CUDA 环境

Detectron2这个框架对CUDA版本的要求相当严格，深度学习搭环境的老毛病了。

服务器上面有队友按开发文档（然而开发的库在今年年初就archived了），它是这么写的：

```bash
conda create --name mask2former python=3.8 -y
conda activate mask2former
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

cd ..
git clone git@github.com:facebookresearch/Mask2Former.git
cd Mask2Former
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

我尝试在本地部署，执行到 `sh make.sh` 时遇到了 C/CUDA 扩展构建错误；随后确认文档要求 Linux 或 macOS，而本地环境是 Windows，因此把编译和训练转移到服务器上。

服务器那边是Linux倒没问题，队友跟我说eval没问题，自己跑也确实没问题。

但是到了 train 和 inference 阶段又出现了 C/CUDA 相关报错。我结合错误信息和 AI 辅助逐步定位，最后回到下面这条 PyTorch 安装命令：

`conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia`

这里的 PyTorch 包绑定了 cudatoolkit 11.1，而 `nvidia-smi` 显示的是驱动支持的 CUDA 兼容能力（12.0），并不等同于系统安装的 toolkit 版本。为减少扩展编译和运行时的不确定性，我改用了已验证可用的 11.7 组合。

此外，系统的CUDA_HOME的值是空的，需要手动设一下值，也算是复习vi了。

### 设置CUDA_HOME

有读写权限但是没sudo权限，不敢乱搞。

1. `vim ~/.bashrc`定位到环境变量。
2. 添加了形如以下内容到环境变量的末尾：
```bash
# ADD CUDA HOME
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
```
3. 保存退出（`esc`，`:wq`）
4. `source ~/.bashrc`使环境变量生效。
5. `nvcc -V`检查是否生效。

### 升级CUDA

没有卸载整个环境，用依赖管理尝试调整torch和cuda toolkit版本，最后确认cuda toolkit 11.7可用。

使用了`conda install pytorch==1.13.1 torchvision==0.14.1 cudatoolkit=11.7 -c pytorch -c nvidia`。

这个包的下载和安装速度比较慢。

完成安装之后，重新走一遍流程，在`sh make.sh`前要手动`rm -rf build`清一下缓存。

## 单独指定一张卡来训练

~~是的这么一个简单问题居然要单独写一段。~~

官方文档的描述为：

```markdown
* To run __on cpu__, add `MODEL.DEVICE cpu` after `--opts`.
```

直接设置后仍未达到预期。

官方文档的示例没有覆盖我使用的分布式启动方式。尝试设置 `CUDA_VISIBLE_DEVICES=2,3 python3 xxx.py` 后仍未达到预期，观察进程后发现程序仍使用了 cpu0。

最后在[Stack Overflow](https://stackoverflow.com/questions/39649102/how-do-i-select-which-gpu-to-run-a-job-on)上面找到的结果。Github上也有对应的[Issue](https://github.com/facebookresearch/detectron2/issues/210)。

解决方案相当于是把窗口下的环境变量改了，如下：

```bash
export CUDA_VISIBLE_DEVICES=1
python xxx.py
```

后面阅读了`detectron2\engine\launch.py`的源码，下面有这样一段：

```python
def _distributed_worker(*args,**kargs):
# ...
    comm.create_local_process_group(num_gpus_per_machine)
    if has_gpu:
        torch.cuda.set_device(local_rank) # 按 local_rank 设置当前 GPU
```

在我使用的 launch/多进程配置下，训练进程会按 `local_rank` 重新选择设备；如果环境变量没有在启动子进程前生效，就可能出现设备选择与预期不一致的情况。

## （长）Modified Pretrained，爱来自Deeplab

> PyTorch的[官方源码](https://github.com/pytorch/vision/)
{: .prompt-info }

做到了把backbone替换成了自己构建的MobileNetV3，下一步的需求是引入一个预训练模型，以对标有`model_final_r50.pkl`的ResNet50。

改法是提features->拼最后一层，非常符合直觉，很舒服。

```python
import torchvision.models as models

@BACKBONE_REGISTRY.register()
class PretrainedMobileNetV3(Backbone):
    def __init__(self, cfg, input_shape):
        # ...

        mobilenet = models.mobilenet_v3_large(pretrained=True)
        self.features = mobilenet.features # 提features
        original_channels = mobilenet.last_channel if hasattr(mobilenet, 'last_channel') else 960

        # 拼最后一层
        out_channels = int(112 * self.width_mult)
        self.conv_res5 = nn.Conv2d(original_channels, out_channels, kernel_size=1, bias=False)
        self.bn_res5 = nn.BatchNorm2d(out_channels)
        self.act_res5 = nn.Hardswish(inplace=True)

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": int(16 * self.width_mult),
            "res3": int(24 * self.width_mult),
            "res4": int(40 * self.width_mult),
            "res5": out_channels,
        }

    # ...
```

然后一边跑的时候一边写文档，发现一个问题：我的目标是实例分割，这个weights不会是分类的吧？

一看[说明文档](https://pytorch.org/vision/main/models.html)，果不其然，这个weights拿IMAGENET1K_V1训的。

与目标任务更接近的是语义分割权重 `DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1`。不过 DeepLabV3 是语义分割模型；在本项目中，我只借用其中的 backbone 权重进行特征初始化，并不能把它等同于实例分割权重。

它的全类名是`torchvision.models.segmentation.deeplabv3_mobilenet_v3_large`，很显然，和`torchvision.models.mobilenet_v3_large`对不上层级，盲猜直接改要出事。

果不其然，DeepLabV3版本的MobileNetV3被动过了，首先它报了找不到features，果断去看了源码：

```python
def deeplabv3_mobilenet_v3_large(
    *,
    weights: Optional[DeepLabV3_MobileNet_V3_Large_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[MobileNet_V3_Large_Weights] = MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    **kwargs: Any,
) -> DeepLabV3:
    weights = DeepLabV3_MobileNet_V3_Large_Weights.verify(weights)
    weights_backbone = MobileNet_V3_Large_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
        aux_loss = _ovewrite_value_param("aux_loss", aux_loss, True)
    elif num_classes is None:
        num_classes = 21

    backbone = mobilenet_v3_large(weights=weights_backbone, dilated=True) # 注意这一行
    model = _deeplabv3_mobilenetv3(backbone, num_classes, aux_loss)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model
```

小问题改个名字的事。

但是改成了backbone之后，又报了KeyError，排了两个多小时没找到原因，后面发现是TorchVision的语义分割模型（如FCN、DeepLabV3）普遍采用字典结构，这一点查了很久才查到——同一行，在`_deeplabv3_mobilenetv3`中，调用了一个关键类`IntermediateLayerGetter`，位于`models._utils`：

```python
class IntermediateLayerGetter(nn.ModuleDict):
    # ...

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out
```

这个手术类返回的是OrderedDict，而非nn.Sequential，因此不兼容，问题解决。

解决方案是将字典结构的层再转换回连续的nn.Sequential模块，来兼容Detectron2的Backbone接口。

> 这是为兼容 Detectron2 Backbone 接口而增加的适配层，属于工程性折中；后续仍应补充接口测试。
{: .prompt-warning }


```python
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_MobileNet_V3_Large_Weights

@BACKBONE_REGISTRY.register()
class PretrainedMobileNetV3(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self._out_features = cfg.MODEL.MOBILENET.OUT_FEATURES
        self.width_mult = cfg.MODEL.MOBILENET.WIDTH_MULT

        deeplab_model = deeplabv3_mobilenet_v3_large(
            weights=DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1,
            weights_backbone=None
        )

        self.stage2 = nn.Sequential(
            deeplab_model.backbone['0'],  # ConvBNHardswish
            deeplab_model.backbone['1']   # InvertedResidual
        )
        self.stage3 = nn.Sequential(
            deeplab_model.backbone['2'],
            deeplab_model.backbone['3']
        )
        self.stage4 = nn.Sequential(
            deeplab_model.backbone['4'],
            deeplab_model.backbone['5'],
            deeplab_model.backbone['6']
        )
        self.stage5 = nn.Sequential(*[
            deeplab_model.backbone[str(i)] for i in range(7, 17)
        ])

    # ...
```

目前无论使用预训练还是不使用预训练，效果都未达到预期，仍需从数据、损失、初始化和评估流程继续排查。

（未完待续）
