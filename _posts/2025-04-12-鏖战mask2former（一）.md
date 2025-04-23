---
title: 鏖战mask2former（一）
date: 2025-04-12 12:00:00 +0800
categories: [笔记, 编程]
tags: [笔记, 编程, 深度学习, CUDA, 实例分割, detectron, DEBUG]
description: 折磨了我整整一周的时间，进入编程领域至今处理的最棘手的问题...
---

> 感觉算是经验帖，以后可能可以作为自己DEBUG的范式了。
{: .prompt-info }

> 因为和实际工程挂钩，所以还暂时不能完整公开干了什么，就尽可能以替代的方式描述。后续如果论文发了中了再补个论文的链接。
{: .prompt-warning }

## 前言

前情提要：

[第一次使用Linux远程开发]({% post_url 2025-03-12-第一次使用Linux远程开发 %})

[迎击mask2former]({% post_url 2025-04-06-迎击mask2former %})

这个description可能是有点标题党了，对于很多人来说可能只是一个微不足道的小问题，但它确实也是我编程以来处理过的最棘手的问题，于是又单开了一篇文章来记录。

按照开发周期（一周）来进行记录，（一）记录的是第一周的内容，主要包括了整个新模型构建到运行的阶段。

下一篇见[鏖战mask2former（二）]({% post_url 2025-04-16-鏖战mask2former（二） %})

> 请善用导航，本文内容特别长。
{: .prompt-info }

### 一点题外话

细究自己的编程史的话，大概是小学开始学turtle，初中还是高中接触了一点shell和VBasic，高中信赛学C结果初试被刷，一气之下自己把Java和JavaScript学了点基础。

大一想学C没报上课只能学了Python，当年暑假把JavaSE完整看了一遍，JavaEE看到SSM三件套；大二大三到大学毕业都是接着Python这条路走机器学习和深度学习（学校开的专业课），碰到导师也是搞这个方向，直到毕设都是搞的这个。

大学学的土木，毕业读研转了计，大学毕业到研究生开学这会，看的主要是网页这块，像Vue、Nuxt这块。研一上跟组里面干点杂货，组员带我写了React（ANT Design）。

然后现在就是目前这个项目，和毕设那会的东西差不多还是深度学习。

虽然遇到的麻烦问题不少，但是现在这个真给我恶心坏了。

## BEFORE EVERYTHING GET STARTED

### 需求描述

也算是敏捷开发的一个迭代周期了。

这一部分主要开发时间：2025-04-06至2025-04-12。

> 用户故事：作为一个计算机视觉开发工程师（？我不是我没有），我希望能在数据类型中引入新的特征，以便更好地训练。

是的，就是这样一条平平无奇的需求，非常直白。

### 更具体些

在正常情况下，一条语义分割的结果通常会包括labels（包含有完整label列表的概率分布）和masks（逐像素的遮罩概率分布）；在某些情况下，它还会带有bbox（可以由遮罩计算或者单独保存的，用于表征位置的图像框）；其中labels取最高概率，masks采用阈值或其他连通算法等，导出为最终的分割结果。

对于标注数据，由于全部的信息已经由人工确定，因此label和mask被唯一确定。一个常见的标注数据类型是[COCO](https://cocodataset.org/)。

COCO数据集现在有3种标注类型：object instances（目标实例，用于目标检测）, object keypoints（目标上的关键点，用于姿态估计）, and image captions（看图说话），其中目标检测即为目前项目的目标，其数据集标注的关键结构长下面这样：

```json
{
    "info": "info",
    "licenses": ["license"],
    "images": ["image"],
    "annotations": {
        "id": "int",
        "image_id": "int",
        "category_id": "int",
        "_comment_1": "与下面的categories对应",
        "segmentation": "RLE or [polygon]",
        "_comment_2": "RLE是一种压缩的方法",
        "area": "float",
        "bbox": "[x, y, width, height]",
        "iscrowd": "0",
        "_comment_3": "在Panoptic Segmentation（全景分割）下取1，在Instance Segmentation（实例分割）下取0",
    },
    "categories": ["category"]
}
```

这里的category_id就是label的确定值。

现在的需求是扩展这里的label。例如：原本的label可能是常见的房子、车等，现在我希望它是**多个独立**的子类，像是：

```json
{
  "color": {
    "red", "blue", "green"
  },
  "size": {
    "small", "medium", "large"
  }
}
```

并且，这个扩展类需要投入训练-预测的完整过程。

## 开始开发

### PLAN FIRST

确定以以下的任务步骤完成本阶段的开发：

- [x] 分解为可执行任务（本步）
- [x] 开发环节
  + [x] 明确数据格式
  + [x] 分析原始构造路径
  + [x] 开发
    * [x] 数据导入
    * [x] 新工作流模型构建
    * [x] 测试
- [x] 收尾

### 数据部分

很显然，由于数据规模严重不足，把几个属性合并为一个属性（形如“A_B_C_D”）是不切实际的，除非使用few-shot，但是训练时间根本来不及。

考虑两种数据结构：

1. 先合并，再分块；
2. 直接分开。

先合并再分块的策略是把全部的类别并入一个大类，然后对于每个大类的每个分区，取每个区最高的类别作为结果，例如：

```python
pred_logits = [0.1, 0.2, 0.3, 0,4, 0.5, 0.6]
// 假设类别的划分是[4, 2]，那么结果会是{4, 6}，相较于单logits结果的{6}。
```

它好改，但是从语义上似乎不那么符合需求，而且标注和后续用于训练可能会比较麻烦，因此选了另一种策略，也就是直接分开：

考虑使用一个属性作为训练的主属性，其余属性作为训练的次要属性，在IO流上尽可能保持和原始的一致性，同时在训练时让每个属性的权重处于平级关系。

此时的输出就应该是：

```python
pred_logits = [0.1, 0.2, 0.3, 0,4] # 主类别
pred_subcat = [0.5, 0.6]           # 次要类别
```

其数据标注为下述结构，以下简称特别COCO：

```json
{
    "info": "info",
    "licenses": ["license"],
    "images": ["image"],
    "annotations": {
        "id": "int",
        "image_id": "int",
        "category_id": "int",
        "subcat_id": "int",
        "↑1": "额外的部分",
        "segmentation": "RLE or [polygon]",
        "area": "float",
        "bbox": "[x, y, width, height]",
        "iscrowd": "0",
    },
    "categories": ["category"],
    "subcats": ["subcats"],
    "↑2": "额外的部分"
}
```

### 构造路径

原始的结构见[Mask2Former](https://bowenc0221.github.io/mask2former/)的[论文](https://arxiv.org/abs/2112.01527)和[相应代码](https://github.com/facebookresearch/Mask2Former)。

它以[MaskFormer](https://github.com/facebookresearch/MaskFormer)作为蓝本，同时使用了FAIR的[detectron2](https://github.com/facebookresearch/detectron2)为其基础框架。

> 尽管在框架的描述中，它是“组装性”的，块状的，但实际开发过程中却因为框架自身的原因出现了不少麻烦。它的文档也写的相对糟糕，必须要读框架的源码才能正常进行复杂任务的开发。
{: .prompt-warning }

阅读了框架的结构，差不多看了两三天源码，构造问题转化为以下几条。

- 现状：只有以[labelme](https://github.com/jameslahm/labelme)格式标注的数据。
- 目标：实现特别COCO数据的导入、训练结构。

➡️ 任务的子步骤：

1. 将labelme格式的数据批量处理为特别COCO；
2. 经过研究，需要调整的被组装件有Mapper（映射器）、Meta-Arch（元架构）、Decoder（解码结构）、Criterion（损失函数）；
   1. Mapper需要转为能理解多元label的Mapper；
   2. Decoder需要转为能输出多头label的Decoder；
   3. Criterion需要转为能对多label进行评估的Criterion；
   4. Meta-Arch需要兼容上述结构。
3. 需要制作一个独立的可视化工具，可以利用输出的pth权重文件进行multi-label的预测并可视化结果。

> “应该还蛮好搞的吧。”
{: .prompt-danger }

~~说出上面一句话的自己在开发过程中被狠狠打脸了。~~

没有不存在的，其实之前跟detectron2打过交道，知道这个框架用起来坑会比较多，现在的需求其实是比较麻烦的，所幸算是模块化程度比较好，没有出现完全无法开发（底层写死根本没法动）的情况发生。

## DEEP INTO CODE

### 数据处理与可视化工具

数据处理之前要洗数据，虽然是队友帮忙标的数据不放心检查了一下，果然还是有漏掉逗号的。随便写了下检出了问题数据自己手动进JSON补标了一下（是的因为已经不是labelme的数据了，所以不能用labelme）了。然后一套JSON处理流程很顺利的就解决了到特别COCO的过程。

此外，为了读数据还需要按[detectron规范](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html)写一个register，原生的像load_coco_json不能用，继承了也没法读额外的标，只能自己写一个custom_loader然后手动输list丢进去给它注册。

可视化工具这块同样的问题，detectron2自带的[Visualizer类](https://detectron2.readthedocs.io/en/latest/_modules/detectron2/utils/visualizer.html)和里面的方法是一条龙，丢进去数据直接就生成segmentation mask出图，拿不到query没法做多出的label与mask的匹配，只好重写整个方法逐个query画mask+label。

### 不会测试的大麻烦

问题最大的是第二部分。

因为数据是**包含有**标准COCO的部分的，因此光读取和训练是可以的（相当于只关注主标签）。

第二部分的组件几乎都是以继承实现，仿照已经写好的mask2former组件做一样的io处理。不过这种东西没法做单独的测试（因为你也不知道你写的能跑通的玩意输入输出是不是和它的一样，看着一样的实际上又会冒各种问题），只能整个接一起测，一次性把四个组件都加上多标签，然后一遍遍跑完整程序，蹲报错，顺着报错改，然后又跑一遍重复。

每次都是完整启动，要把数据载到内存里面的启动过程相当花时间，再加上有些组件和内部组件有联动，错了要从逻辑去分析原因还是挺麻烦的。

> 其实肯定会有办法能做独测的，只是我暂时没想到w
>
> 还是基础太糟糕。
{: .prompt-tip }

印象比较深的有下面两个，全是基础问题。

#### 不合法的loss

> 太依赖AI，不仔细看源代码导致的。
{: .prompt-danger }

首先是报loss不合法，不是一维tensor，追查到`detectron2/engine/train_loop.py`：

```python
if self.zero_grad_before_forward:
            self.optimizer.zero_grad()
        with autocast(dtype=self.precision):
            loss_dict = self.model(data)               # 这里
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())       # 报错在这但显然不是它的锅
```

这里根据config里面写的整合规则，逻辑上可以得出`self.model=MultiTaskMaskFormer(MaskFormer)`，追溯到自己实现的Meta-Arch件。

怀疑是不是loss的实现有问题，又去看了自己重写的Criterion件，加了测试方法输出，但是loss里面输出全是一维tensor的情况下，依然会出现sum有不是一维tensor而报错。

最后忍无可忍动底层把`train_loop.py`改了测试，发现会莫名其妙多出来非一维tensor，就是把额外标的label的矩阵结果也当loss返回了。

百思不得其解，又去研究model，注意到了之前没认真研究的一个事情，也算是语法糖吧。

> [PyTorch中model(image)会自动调用forward函数。](https://zhuanlan.zhihu.com/p/366461413)
{: .prompt-tip }

在PyTorch继承的Module类里面，会自动调用forward函数，根据知乎博主[Pinging](https://www.zhihu.com/people/pinging-92)观察到的，结果如下：

```python
def _call_impl(self, *input, **kwargs):
    # forward_pre_hook:记录网络前向传播前的特征图
    # 将_global_forward_pre_hooks与_forward_pre_hooks中的hook拿来用，这两个参数的内容是在上面的注册函数中不断补充进来的，如果前端不调用注册，则这边没有内容；
    for hook in itertools.chain(
            _global_forward_pre_hooks.values(),
            self._forward_pre_hooks.values()):
        result = hook(self, input)
        if result is not None:
            if not isinstance(result, tuple):
                result = (result,)
            input = result
    # forward
    if torch._C._get_tracing_state():
        result = self._slow_forward(*input, **kwargs)
    else:
        result = self.forward(*input, **kwargs)
    # forward_hook:记录前向传播后的特征图
    for hook in itertools.chain(
            _global_forward_hooks.values(),
            self._forward_hooks.values()):
        hook_result = hook(self, input, result)
        if hook_result is not None:
            result = hook_result
    # backward_hook:记录反向传播后的梯度数据
    if (len(self._backward_hooks) > 0) or (len(_global_backward_hooks) > 0):
        var = result
        while not isinstance(var, torch.Tensor):
            if isinstance(var, dict):
                var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
            else:
                var = var[0]
        grad_fn = var.grad_fn
        if grad_fn is not None:
            for hook in itertools.chain(
                    _global_backward_hooks.values(),
                    self._backward_hooks.values()):
                wrapper = functools.partial(hook, self)
                functools.update_wrapper(wrapper, hook)
                grad_fn.register_hook(wrapper)
    return result

__call__ : Callable[..., Any] = _call_impl
```

因此，既然Criterion组件也没问题，那么可以断言问题应该还是出在其后的阶段，很有可能是Meta-Arch哪里调错了。

最后对比源文件`maskformer_model.py`，观察到forward方法中有这样一个内容被我漏掉了：

```python
    def forward(self, batched_inputs):
        # ...

        if self.training:
            # ...
                    
            return losses
        else:
            # ...

            return processed_results
```

而我在我的maskformer重写里面没有考虑训练态，一律返的一般结果，这就导致了预测结果的错误嵌入和使用。

其实是一个很明显的问题，但是因为时间太紧基础不行，用AI辅助完全就没注意到这个问题，一开始光看到最后的输出一样就没管了，很长了一点教训。

#### How to Inheritance in Python?

> 我个人不是一般地讨厌Python，尤其是它的面向对象让这门语言的恶心程度上升到了一个新的境界。
{: .prompt-warning }

这次是被Python的面向对象上了一课。

在detectron2的原始文件`mask2former_transformer_decoder.py`（Decoder件）中，有这样两段声明：

```python
@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder(nn.Module):

    # ...

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            ...
        """
        super().__init__()

        # ...
    
    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        return ret
```

其中*是表示其后的参数必须传值，是3.8开始的新特性，参见[Python官方文档](https://docs.python.org/zh-cn/3.8/tutorial/controlflow.html#special-parameters)对应内容的叙述，也是长知识了。

那么问题来了，这个类怎么继承？

在先前做简单修改，还没有引入多标签的时候，有一则成功实践：

```python
@TRANSFORMER_DECODER_REGISTRY.register()
class CustomMultiScaleMaskedTransformerDecoder(MultiScaleMaskedTransformerDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ...
```

但是如果要再加参数呢？

我试图采用`__init__(self, subcat_1, subcat_2, *args, **kwargs)`来继承，长下面这样才勉强看上去合法：

```python
@TRANSFORMER_DECODER_REGISTRY.register()
class CustomMultiScaleMaskedTransformerDecoder(MultiScaleMaskedTransformerDecoder):
    @configurable
    def __init__(self, subcat_1, subcat_2, *args, **kwargs):
        self.subcat_1 = subcat_1
        self.subcat_2 = subcat_2

        # ...

        super().__init__(
            in_channels=in_channels,
            mask_classification=mask_classification,
            num_classes=kwargs.pop('num_classes'),
            hidden_dim=kwargs.pop('hidden_dim'),
            num_queries=kwargs.pop('num_queries'),
            nheads=kwargs.pop('nheads'),
            dim_feedforward=kwargs.pop('dim_feedforward'),
            dec_layers=kwargs.pop('dec_layers'),
            pre_norm=kwargs.pop('pre_norm'),
            mask_dim=kwargs.pop('mask_dim'),
            enforce_input_project=kwargs.pop('enforce_input_project'),
            **kwargs
        ) # 父类明确指定了，所以继承的时候必须这么做
```

你会发现这玩意还是很长。

而且由于这玩意和Meta-Arch中的有一小块有联携，其父类参数没有明确指定（用的`kwargs.pop()`），还有一些来自于各种各样的装饰器的成员变量导致参数的遮盖等问题，实际上上面这个还是没有跑通。

**最后完整把父类抄了一遍**，真被Python这个气笑了。

## 收尾

在反复的来回看文档、测试和重写代码之后，整个流程也算是跑通了。

不得不感慨还是要把基础打牢，DEBUG要好好学一学了。

~~能不写Python了吗写JS或者Java也行啊我要死了~~

后面等待着我的是...

[鏖战mask2former（二）]({% post_url 2025-04-16-鏖战mask2former（二） %})