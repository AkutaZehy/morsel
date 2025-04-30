---
title: 鏖战mask2former（三）
date: 2025-04-23 12:00:00 +0800
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

[鏖战mask2former（二）]({% post_url 2025-04-16-鏖战mask2former（二） %})

（三）介绍的是当前项目最后一个**开发阶段**的工作，差不多两周左右。

> 请善用导航，本文内容特别长。
{: .prompt-info }

## BEFORE EVERYTHING GET STARTED

### 需求描述

在紧急处理完核心开发周期二的任务之后，对时间上的要求缓和了不少。

这一部分主要开发时间：2025-04-21至2025-04-30。

> 用户故事：作为一个计算机视觉开发工程师（？我不是我没有），我希望能检出并修复训练中的问题，让新模型结果能够连接上下游管道，对于上游给定的输入，能够产出可用于下游管道的、可接受的数据。

考虑到先前的开发并不顺利，如果是我肯定会分配更多的故事点，但是时间不允许了。

### 更具体些

在先前的开发周期中，紧急处理任务部分套用了原始结构，仅仅采用了数据集适配，得到*勉强可以被接受的*可视化结果，AP=4；在自己继承的多任务结构中依旧得到AP=0，且可视化结果完全不对。

> 显然，AP=4不能解释“可视化结果还可以被接受”的现象（相较于数据集ADE20K的AP=26）；
> 
> 同时，AP=0和错乱的可视化结果也是不可被接受的。

由于开发周期长一些，本文按该周期的开发阶段展开具体的实施。

## 第一阶段

目标：让新引入的数据集，在原始的评估下获得正常的训练结果

按照自己的观察和理解，目前存在两个问题：

1. Query数量不对
2. 训练参数好像设错了

### Query是怎么回事

对于Query，训练采用了和原始模型一样的`Q=100`。在自己的可视化代码片做了测试，表明最后输出了非常多的空类。

有篇医学刊 *[Taming Detection Transformers for Medical Object Detection](https://arxiv.org/abs/2306.15472)* 在不同数据集上用了不同的query；同时，阅读到detectron2框架的val验证部分，有如下代码：

```python
coco_dt = coco_gt.loadRes(coco_results)  # 加载预测结果
coco_eval = COCOeval(coco_gt, coco_dt, "segm")  # 初始化segm评估器

# 关键评估参数设置
coco_eval.params.maxDets = [1, 10, 100]  # 默认使用100个检测结果
coco_eval.evaluate()  # 执行评估
coco_eval.accumulate()  # 累积指标
coco_eval.summarize()  # 输出结果
```

其中这个`maxDets = 100`的部分引起了我的注意。这也就意味着，如果我有大量的低质量预测，全部会被塞进去：如果减少一下query的数量，是不是就可以抹去多余的低质量预测了？

当然这是一个针对评估器的HACK做法，不过值得一试——FAIR写死的评估器有问题的话，那修改验证策略就行，结果是没问题的。

不过与此同时，`maxDets = [1, 10]`的部分好像值也不太高，也许这部分没毛病？

### 训练参数的常识性问题

回过头来看训练参数犯了个大蠢。

在原始的训练条件下，配置这么写的：

```yaml
SOLVER:
  IMS_PER_BATCH: 16
  BASE_LR: 0.0001
```

但是batch_size=16这个，卡的内存实在不够带不动，训练的启动参数我写成了：

```bash
python train_net.py \
  --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml \
  SOLVER.IMS_PER_BATCH 2
```

而根据缩放原则，16->2的batch_size变化意味着，我需要修改lr以匹配这一变化，但是我没改，此时的lr应该在1e-6到1e-5这个水平，1e-4显然就太大了。

不过对于原始数据集，这个“忘了改lr”的问题没有对结果产生不利影响。

### Ablation Result

这几天就我在忙，组里面刚好没啥人在用卡，考虑到上述两种情况，干脆每种都跑了一个ablation。

> 实验结果表明，核心问题在第二个，第一个确实影响不大。
{: .prompt-info }

这部分结束后，按照GPT的指示，我回看了一篇FAIR在2020的核心，DETR，相当于mask2former前置maskformer的前置论文（是的这三篇都是FAIR的）。

在文章 *[End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)* 中，有这么一段：

> **Increasing the number of instances**
>
> <mark>By design, detr cannot predict more objects than it has query slots, i.e. 100 in our experiments. </mark> In this section, we analyze the behavior of detr when approaching this limit. We select a canonical square image of a given class, repeat it on a $10\times 10$ grid, and compute the percentage of instances that are missed by the model. To test the model with less than 100 instances, we randomly mask some of the cells.
> 
> <mark>This ensures that the absolute size of the objects is the same no matter how many are visible.</mark> To account for the randomness in the masking, we repeat the experiment 100 times with different masks. The results are shown in Fig.\ref{fig:instances}. The behavior is similar across classes, and while the model detects all instances when up to 50 are visible, it then starts saturating and misses more and more instances. Notably, when the image contains all 100 instances, the model only detects 30 on average, which is less than if the image contains only 50 instances that are all detected. The counter-intuitive behavior of the model is likely because the images and the detections are far from the training distribution. 
>
> Note that this test is a test of generalization out-of-distribution by design, since there are very few example images with a lot of instances of a single class. It is difficult to disentangle, from the experiment, two types of out-of-domain generalization: the image itself vs the number of object per class. But since few to no COCO images contain only a lot of objects of the same class, this type of experiment represents our best effort to understand whether query objects overfit the label and position distribution of the dataset. <mark> Overall, the experiments suggests that the model does not overfit on these distributions since it yields near-perfect detections up to 50 objects. </mark>

所以结论就是，`Q=100`的多余检测问题不大，是我过虑了。

> 还是不得不感叹，搞深度学习的资源多就是好啊。
{: .prompt-info }

## 第二阶段

在前一步的基础上，解决了新数据集+原始pipeline的AP低的问题后，把更新的lr用到新的pipeline上，主类数值是好了不少，但是现在出现的需要解决的问题有：

- [x] 预测出的图像有一个明显的位移
- [x] 在`subcat=0`的状态下Criterion件依旧会报错中断训练

这一阶段的目标是，在不替换backbone的状态下（即只替换分类头+对应优化pipeline的条件下，完成完整训练并得到可接受结果）。

### Criterion的问题

位移问题通过仿照了原始COCO，补充aug过程得到解决。

在上一个开发周期中所提出的Criterion件异常被复现出来，为：

```bash
/opt/conda/conda-bld/pytorch_1670525541702/work/aten/src/ATen/native/cuda/NLLLoss2d.cu:104: nll_loss2d_forward_kernel: block: [0,0,0], thread: [0,0,0] Assertion `t >= 0 && t < n_classes` failed.
/opt/conda/conda-bld/pytorch_1670525541702/work/aten/src/ATen/native/cuda/NLLLoss2d.cu:104: nll_loss2d_forward_kernel: block: [0,0,0], thread: [1,0,0] Assertion `t >= 0 && t < n_classes` failed.
/opt/conda/conda-bld/pytorch_1670525541702/work/aten/src/ATen/native/cuda/NLLLoss2d.cu:104: nll_loss2d_forward_kernel: block: [0,0,0], thread: [3,0,0] Assertion `t >= 0 && t < n_classes` failed.
/opt/conda/conda-bld/pytorch_1670525541702/work/aten/src/ATen/native/cuda/NLLLoss2d.cu:104: nll_loss2d_forward_kernel: block: [0,0,0], thread: [4,0,0] Assertion `t >= 0 && t < n_classes` failed.
/opt/conda/conda-bld/pytorch_1670525541702/work/aten/src/ATen/native/cuda/NLLLoss2d.cu:104: nll_loss2d_forward_kernel: block: [0,0,0], thread: [5,0,0] Assertion `t >= 0 && t < n_classes` failed.
/opt/conda/conda-bld/pytorch_1670525541702/work/aten/src/ATen/native/cuda/NLLLoss2d.cu:104: nll_loss2d_forward_kernel: block: [0,0,0], thread: [6,0,0] Assertion `t >= 0 && t < n_classes` failed.
/opt/conda/conda-bld/pytorch_1670525541702/work/aten/src/ATen/native/cuda/NLLLoss2d.cu:104: nll_loss2d_forward_kernel: block: [0,0,0], thread: [7,0,0] Assertion `t >= 0 && t < n_classes` failed.
/opt/conda/conda-bld/pytorch_1670525541702/work/aten/src/ATen/native/cuda/NLLLoss2d.cu:104: nll_loss2d_forward_kernel: block: [0,0,0], thread: [8,0,0] Assertion `t >= 0 && t < n_classes` failed.
/opt/conda/conda-bld/pytorch_1670525541702/work/aten/src/ATen/native/cuda/NLLLoss2d.cu:104: nll_loss2d_forward_kernel: block: [0,0,0], thread: [9,0,0] Assertion `t >= 0 && t < n_classes` failed.
/opt/conda/conda-bld/pytorch_1670525541702/work/aten/src/ATen/native/cuda/NLLLoss2d.cu:104: nll_loss2d_forward_kernel: block: [0,0,0], thread: [10,0,0] Assertion `t >= 0 && t < n_classes` failed.
/opt/conda/conda-bld/pytorch_1670525541702/work/aten/src/ATen/native/cuda/NLLLoss2d.cu:104: nll_loss2d_forward_kernel: block: [0,0,0], thread: [11,0,0] Assertion `t >= 0 && t < n_classes` failed.
/opt/conda/conda-bld/pytorch_1670525541702/work/aten/src/ATen/native/cuda/NLLLoss2d.cu:104: nll_loss2d_forward_kernel: block: [0,0,0], thread: [12,0,0] Assertion `t >= 0 && t < n_classes` failed.
/opt/conda/conda-bld/pytorch_1670525541702/work/aten/src/ATen/native/cuda/NLLLoss2d.cu:104: nll_loss2d_forward_kernel: block: [0,0,0], thread: [13,0,0] Assertion `t >= 0 && t < n_classes` failed.
/opt/conda/conda-bld/pytorch_1670525541702/work/aten/src/ATen/native/cuda/NLLLoss2d.cu:104: nll_loss2d_forward_kernel: block: [0,0,0], thread: [14,0,0] Assertion `t >= 0 && t < n_classes` failed.

# ...

RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
```

如果仔细观察，不仅发生断言失败，且`thread: [2,0,0]`并没有出现在上述异常outputs中，表明这个是正确的。

从报错的n_classes来看，问题出现在Criterion件中，但这个量应该出现在原版Criterion中（继承没重写），这不太合理。

怀疑是数据变化引起的问题，检查了Criterion件中可能用到交叉熵的部分，发现以下部分：

```python
class SetCriterion(nn.Module):
    # ...
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                  num_points, oversample_ratio, importance_sample_ratio):
        # ...
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1) # 注意这里
    
    # ...

    def loss_labels(self, outputs, targets, indices, num_masks):
          """Classification loss (NLL)
          targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
          """
          assert "pred_logits" in outputs
          src_logits = outputs["pred_logits"].float()

          idx = self._get_src_permutation_idx(indices)
          target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
          target_classes = torch.full(
              src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device # 注意这里
          )
          target_classes[idx] = target_classes_o

          loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
          losses = {"loss_ce": loss_ce}
          return losses

    # ...
```

由于在我自己的标注中没有背景类指定，对于一个给定的num_classes值，任意一个无目标的query（背景类）的target class均为num_classes，刚好不满足t < num_classes，这是很显然的。然而，在原版本中也是这么写的，没有出现任何问题，这意味着不是这里的问题。

使用标准的正交测试方法，对class和另外的subcat进行测试，测试结果说明确实不是loss_labels的问题，而是loss_subcat的问题，即在重写的Criterion件中。

此外，在上一个阶段，如果让代码运行起来的话，subcat的对应weight没有做NUM+1，这引起了我的警觉：表明subcat并不能像原始的classes那样支持背景类。

顺着检查到了Decoder件，这个问题就顺利解决了：

```python
@TRANSFORMER_DECODER_REGISTRY.register()
class MultiTaskTransformerDecoder(MultiScaleMaskedTransformerDecoder):
    @configurable
    def __init__(
        self,
        cfg,
        in_channels: int,
        mask_classification: bool,
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
        num_subcat1: int,
        num_subcat2: int,
        **kwargs
    ):
        # 使用父类的完整属性
        super().__init__(
            in_channels=in_channels,
            mask_classification=mask_classification,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            mask_dim=mask_dim,
            enforce_input_project=enforce_input_project,
            **kwargs
        )
        # 多任务预测头
        self.subcat_1_embed = nn.Linear(hidden_dim, num_subcat1) # 这里
        self.subcat_2_embed = nn.Linear(hidden_dim, num_subcat2) # 还有这里
        # 正确的写法应该形如 self.subcat_embed = nn.Linear(hidden_dim, num_subcat + 1)
```

把对应的位置+1加上背景类之后，对应地修改criterion的数量做上匹配，没有再出现Criterion异常。

### 怎么全是背景类

完成上述的修改，推进¼完整iter的train，val的结果显示主类别训练正常稳定（即基本不需要再调整），但subcat全是背景类。

推测出现这种情况的原因有：

1. 优化器出现问题
2. 类别合理性出现异常
3. 梯度更新缓慢
4. 背景类设置异常

做了排查，上述的Criterion实际上解决了原因4，其次对代码进行了排查，解决了可能是原因2、3的可能性。

进一步对原因1进行排查，尝试了不设主类权重，将权重全部偏置给subcat，此时主类别如预期一样无法训练，mask可以正常训练，但subcat依旧无法训练。表明模型内部的label-mask的匹配机制在工作，但似乎没有带上多出的subcat。

通过检查到原始模型中的matcher类，即HungarianMatcher（匈牙利匹配）的部分：

```python
class HungarianMatcher(nn.Module):
    # ...

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            # ...

            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        # ...
```

这表明确实没加上，于是对应继承了匈牙利匹配，并仿照logits加上了对应的subcat。

然而，加上之后也仅仅只是有好转，而且有一个很怪异的问题，就是随着训练的进行，iteration较小的时候还能够预测出一些错误的类别，但当iteration变高时，又都回到了背景类。但很显然并不是没有训练的状态，对应的loss确实是随着训练进行而不断变小的，而且DEBUG了loss的对应位置的输出，是符合常理的。

> 那就奇了怪了。
{: .prompt-warning }

我决定看一下到底输出了什么，不仅仅是输出的subcat，还要看一下每个值到底是什么。

挺离谱的，但是这一检查很快我就发现了问题：所有的subcat的输出logits好像没动？反复检查了几遍，证实了我的猜想，所有的输出logits都是一样的。

所有输出的logits一样，但是训练的时候的loss是会改变的，这到底是什么情况呢？

使用二分法检错，紧接着又检查了criterion前面的一步（模型进行可视化预测的时候，必定需要加载模型），得到的output确实会随着输入不同而改变，而且output的输出和输出的logits不一样，这是并不合理的，说明问题很有可能出在output和logits之间，这部分内容在Meta-Arch件。

最后，在重写的Meta-Arch件中检出了问题：

```python
@META_ARCH_REGISTRY.register()
class MultiTaskMaskFormer(MaskFormer):
    # ...

    def forward(self, batched_inputs):
        # ...

        if self.training:
                # ...
        else:
            # Non-training (inference) phase
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            # Upsample masks to the original image size
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                # ...

                # Add pred_subcat_1 and pred_subcat_2 during inference (non-training)
                # Calculate pred_subcat_1 and pred_subcat_2 from the decoder output
                decoder_output = self.sem_seg_head.predictor.decoder_norm(
                    self.sem_seg_head.predictor.query_feat.weight
                ) # 错在这里
                processed_results[-1]["subcat_1"] = self.sem_seg_head.predictor.subcat_1_embed(decoder_output) # 这里是输出的logits
                processed_results[-1]["subcat_2"] = self.sem_seg_head.predictor.subcat_2_embed(decoder_output) # 还有这里

            return processed_results
```

仔细看了实在没绷住。AI自己注释都写的是`from decoder output`，输入却是`query_feat.weight`。也就是说，它把模型自己的权重信息（weight）直接接到了subcat_embed，而完全没有考虑图片的特征，这结果能变才怪了。

> 本来是偷懒让AI照着logits部分抄的（因为结构差不多），看样子是抄也抄不会了。
>
> 没辙自己改吧，以后核心的代码还是得自己写了。
{: .prompt-warning }

正确的写法是由`outputs`引导到输出，看懂了就很简单了，把这个问题修了之后就解决了，修复后的代码如下：

```python
@META_ARCH_REGISTRY.register()
class MultiTaskMaskFormer(MaskFormer):
    # ...

    def forward(self, batched_inputs):
        # ...

        if self.training:
                # ...
        else:
            # Non-training (inference) phase
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            mask_subcat_1 = outputs["subcat_1"]
            mask_subcat_2 = outputs["subcat_2"]

            # Upsample masks to the original image size
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                # ...

                # Add pred_subcat_1 and pred_subcat_2 during inference (non-training)
                # Calculate pred_subcat_1 and pred_subcat_2 from the decoder output
                processed_results[-1]["subcat_1"] = mask_subcat_1[-1]
                processed_results[-1]["subcat_2"] = mask_subcat_2[-1]

            return processed_results
```

> 回过头来看感觉还挺微妙的。
>
> 也就是说，整个代码的逻辑其实已经崩掉了，但是由于train和val两个阶段是分开的，被框架救了一命，AI犯下的错误不至于让我整个训练都报废了。
{: .prompt-info }

## 第三阶段

### 现状

整合mobilenetv3的想法直接down掉了没时间做了，multitask的结构基本可以正常运行，基本就可以告捷了。

还有一个小问题就是结果的可视化和结果值的评估，其实对于Transformer结构给出Queries的状态我其实没啥头绪。

### 结果评估算法？

结果评估的代码在detectron2框架的coco_evaluation.py中，有如下实现：

```python
def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results
```

这个results包括了score（主类置信度）和segmentation（即mask）。紧接着，results被送入[COCOeval](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)进行匹配并计算结果，源码如下：

```python
class COCOeval:
    # ...

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }
```

总的来说，是基于logits的贪心匹配，优先处理高置信度的预测，每个GT只能被匹配一次（最先匹配的高分预测），未被匹配的高分预测会被视为FP（假阳性），未被匹配的GT会被视为FN（假阴性）。

按照同样的逻辑，考虑目前category和subcat的组合，由于目前几者的weight存在竞争关系，从逻辑上说主类和子类的scores输出不一定都高的情况下，单独使用一个类进行排序是不理智的，因此考虑使用三者的和作为排序logits进行匹配，那么问题就解决了。

> However
{: .prompt-warning }

我没时间了。

试了一下，尽管三个类的输出不是都高，但是相较于背景类来说也还是高多了，logits顺序不一样但是和GT的iou组合还是匹配的，最后对结果的影响不大。

所以最后直接套了原版的，直接拿一个主类的结果来替代了最终的结果。

### 可视化

原版的ADE20K的可视化中，DT都是直接输出的，从逻辑上没有问题。

然而到了我自己的数据集，主要有两个问题：

- 我没有背景类（尽管为了训练加上了）
- 由于算法的问题，出现了一些主类+有效subcat和主类+无效subcat（值取背景）同时预测到了一个mask的情况
  - NMS（非极大值抑制）处理不当
  - Query设计缺乏相似性约束（摒除对同一个mask的多个预测）—— 尤其常见于DETR

导致最后输出的图片里面标签乱七八糟，但好消息是正确的（主类+有效subcat）的分布比较正常。

考虑到算法实在没法改了，想了一个取巧的办法，就是用经典的筛法把无效的结果筛掉，使用了以下的筛：

- 因为subcat理论上没有背景类，因此subcat=背景类的一定是无效结果
- 考虑了主类logits筛，由于存在竞争，实际上0.1就已经足够筛出有效的主类
- 考虑了subcat的logits筛，尽管也存在竞争，但subcat的输出没有完整归一化，取值2~3左右比较合适

## 收尾

想法很多但是时间就只够做这些了，尤其是中间Criterion那一部分实在耗时太多了。不过总算是有个能看的结果了还是蛮好的。

最后差不多是2025-04-29才结束这部分，paper也同步在写。

再写一篇总结篇总结一下吧：

[总结篇！]({% post_url 2025-04-30-2504项目总结 %})