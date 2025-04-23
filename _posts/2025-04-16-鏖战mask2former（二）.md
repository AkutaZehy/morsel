---
title: 鏖战mask2former（二）
date: 2025-04-16 12:00:00 +0800
categories: [笔记, 编程]
tags: [笔记, 编程, 深度学习, CUDA, 实例分割, detectron, DEBUG]
description: 事情变得越发麻烦起来...
---

> 感觉算是经验帖，以后可能可以作为自己DEBUG的范式了。
{: .prompt-info }

> 因为和实际工程挂钩，所以还暂时不能完整公开干了什么，就尽可能以替代的方式描述。后续如果论文发了中了再补个论文的链接。
{: .prompt-warning }

## 前言

前情提要：

[鏖战mask2former（一）]({% post_url 2025-04-12-鏖战mask2former（一） %})

按照开发周期（一周）来进行记录，（二）记录的是第二周的内容。

下一篇见[鏖战mask2former（三）]({% post_url 2025-04-23-鏖战mask2former（三） %})

> 请善用导航，本文内容特别长。
{: .prompt-info }

## BEFORE EVERYTHING GET STARTED

### 需求描述

也算是敏捷开发的一个迭代周期了。

这一部分主要开发时间：2025-04-13至2025-04-19。

> 用户故事：作为一个计算机视觉开发工程师（？我不是我没有），我希望能检出并修复训练中的问题，以让新模型的训练效果与正常水平齐平。

这是一条难度非常大的用户故事，如果是我肯定会分配更多的故事点，但是时间不允许了。

### 更具体些

> 喜报：AP=0！
{: .prompt-danger }

> 这个结果意味着，除了流程确实跑通了以外，整个代码的实现中存在着**较多极为严重的**问题，这使得模型几乎没有学习到任何内容。

考虑到在上一周中，改动的代码包括了Mapper、Meta-Arch、Decoder、Criterion四块组件，因此着重要检查这四块组件的问题。

> 事实上，上一周基本上只是粗略的看了一遍代码，除了检查过一部分的Meta-Arch，其余部分的继承实现几乎全是让AI写的。
>
> 这是一个极度危险的行为，也是导致了这一周工作量巨大的罪魁祸首。这并不是好的AI应用实践。
{: .prompt-danger }

具体的事项包括了：

- [x] 检查原始模型的实现
  - [x] 检查代码中对于预处理部分的实现
  - [x] 检查backbone阶段输出层与decoder部分的链接实现
  - [ ] 检查多损失实现
  - [x] 检查pretrained
- [ ] 检查、修改继承方法
- [ ] 测试

> 你可能注意到了，此处有些事项没有打钩，这意味着它们在这个开发周期没有被完成。
{: .prompt-info }

## LET'S DIVE INTO CODE

### 预处理阶段

预处理阶段的内容被配置在了`mask2former/data/dataset_mappers`，以COCO为例：

```python
def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation
```

代码的内容可以描述为，仅在Train阶段执行数据增强，包括了RandomFlip和Resize-Crop两步骤，使得最终的尺寸为image_size=[1024, 1024]。

而在AI写的继承版本（继承自DatasetMapper），则忽视了数据的预处理，相当于直接把原始尺寸扔进模型了，存在较多隐患，在这一部分上需要做出修改。

### backbone与decoder的集成

在原始的模型文件`maskformer_model.py`中，有如下内容：

```python
@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    # ...

    def forward(self, batched_inputs):
        # ...
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        # ...
```

其中，sem_seg_head追溯到`mask_former_head.py`：

```python
@SEM_SEG_HEADS_REGISTRY.register()
class MaskFormerHead(nn.Module):
  # ...

    def forward(self, features, mask=None):
          return self.layers(features, mask)

    def layers(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(multi_scale_features, mask_features, mask)
        # ...

        return predictions
```

再进一步追溯到multiscale pixel-decoder的实现：
```python
@SEM_SEG_HEADS_REGISTRY.register()
class MSDeformAttnPixelDecoder(nn.Module):
  # ...

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        conv_dim: int,  # 注意这个参数
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        # deformable transformer encoder args
        transformer_in_features: List[str],
        common_stride: int,
    ):
        # ...

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)

            lateral_conv = Conv2d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    # ...

    @autocast(enabled=False)
    def forward_features(self, features):
        # ..

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            lateral_conv = self.lateral_convs[idx] # 注意这里
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = output_conv(y)
            out.append(y)

        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        return self.mask_features(out[-1]), out[0], multi_scale_features
```

而通过观察原始backbone的两个实现，来自detectron2框架内部的resnet实现和swin实现，发现out_features是并不匹配的。

resnet的out_features=[256, 512, 1024, 2048]，而swin的out_features=[96, 192, 384, 768]。通过对上述代码的分析，其又经过了lateral_conv和output_conv的步骤，使得最终的每一层结果被固定转换到conv_dim=256，再进入后续的transformer decoder流程。

那么其对于输入，理论上只要能匹配上res2~res5的多级结构就应该可以完成后续的训练，不需要任额外的转换；而我自己的代码却做了强制转换到features=112才能运行，而且效果相对较差，需要排查其他原因的问题。

> 如果你比我对数字敏感的话，那么这里你应该可以看出来问题出在哪里。
>
> 往下看，看看你猜对了没：
{: .prompt-info }

报错点在以下位置：

```python
@SEM_SEG_HEADS_REGISTRY.register()
class MSDeformAttnPixelDecoder(nn.Module):
    # ...

    @autocast(enabled=False)
    def forward_features(self, features):
        srcs = []
        pos = []

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()  # deformable detr does not support half precision
            srcs.append(self.input_proj[idx](x)) # 报错在这一行
            pos.append(self.pe_layer(x))
        
        #...
```

检查这里的features，得到：

```python
torch.Size([1, 24, 336, 200])
torch.Size([1, 40, 168, 100])
torch.Size([1, 80, 84, 50])
torch.Size([1, 112, 84, 50])
```

作为对照，resnet的输出为：
```python
torch.Size([1, 256, 160, 160])
torch.Size([1, 512, 80, 80])
torch.Size([1, 1024, 40, 40])
torch.Size([1, 2048, 20, 20])
```

这里出现一个怪异的问题，我明明设置的out_features=[24, 40, 80, 160]，为什么得到的out_features=[24, 40, 80, 112]？

经过逐层的检查，最后还是回到了mobilenetv3，打印了Torch中mobilenetv3-large的各层次。

它共包含有20层。以[channel, scale]来记，层2-3为[16, 1/2]，层4-5为[24, 1/4]，层6-8为[40, 1/8]，层9-12为[80, 1/16]，层13-14为[112, 1/16]，层15-17为[160, 1/32]，层18-20为最后分类的层。

> 所以，还是太相信AI导致的了。
{: .prompt-warning }

mobilenetv3没有像resnet那样的res1-res5，而是这样的bottleneck块。交给AI时错误地把112层（13-14）作为了层5。而根据通道翻倍尺寸长宽折半的原则，为了提取到160，要往下额外提取一级。最终枚举出res2-res5应该是层5、8、12、17，而在先前的代码中错误地选为了5、8、12、14。

最终形成的完整可训练架构如下：

```python
@BACKBONE_REGISTRY.register()
class D2PretrainedMobileNetV3(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self._out_features = cfg.MODEL.MOBILENET.OUT_FEATURES

        # 加载预训练的 MobileNetV3-Large 模型
        self.mobilenet = models.mobilenet_v3_large(pretrained=True)

        # 截断模型，只保留到倒数第二层（不包括最后的分类层）
        self.features = nn.Sequential(*list(self.mobilenet.features.children()))

        # 定义特征层的通道数和步长
        self._out_feature_channels = {
            "res2": 24,
            "res3": 40,
            "res4": 80,
            "res5": 160,
        }
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N, C, H, W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"MobileNetV3 takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        for i, layer in enumerate(self.features):
            x = layer(x) # 下面的数要+2才会对应上原始的层次
            if i == 3:  # res2
                outputs["res2"] = x
            elif i == 6:  # res3
                outputs["res3"] = x
            elif i == 10:  # res4
                outputs["res4"] = x
            elif i == 15:  # res5
                outputs["res5"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32
```

这个问题顺利解决。

> 那你肯定要问了，为什么这么简单一个问题早没看出来。
>
> 事实上就是，报错隔得确实特别远，即便在backbone写错，根本不会报在backbone的问题。
> 
> 加上对数字确实太不敏感了。
{: .prompt-info }

> However，结果并没有出现好转，因此这个部分被暂时搁置了：即下一个开发周期将暂停使用Mobilenetv3 backbone。
{: .prompt-danger}

### Criterion？

> 既然整个loss是一个加权相加的结构，那么就意味着，如果设置SUBCAT_WEIGHT=0，从逻辑上说，它应该表现出正常的预测性能，包括main category和mask，只是不能预测出subcategories。
{: .prompt-warning }

然而，无论如何训练，目前无法得出正常的训练结果，除了能保证工作流外完全无法正常进行训练。除开上述两个阶段的问题，在进行训练，会有一定概率诱发“因为Criterion错误而异常中断”的状况，且在SUBCAT_WEIGHT取较低值（如与原始模型中no_cat相似取0.01，或取0）时相对高发。因此需要进一步检查这一部分的内容。

> However，这块没时间看了，太过于低估前面几个部分的工作量。
>
> 这一个问题被留到下一个开发周期。
{: .prompt-danger}

## 突发状况 & 收尾

2025-04-18发生了突发状况，要求用自己的数据集套用原始结构去做可视化，迭代周期只有1天，非常紧张；好在有可视的结果；不过结果能看的同时指标还是非常低（AP约为作为对照数据集的1/5），只能留给下一个周期了。

这一周的有效进展其实没有很多，但做起来就是它特别累，尤其是末尾的突发状况更是雪上加霜，总体来说还是比较糟心的。

后面等待着我的是...

[鏖战mask2former（三）]({% post_url 2025-04-23-鏖战mask2former（三） %})