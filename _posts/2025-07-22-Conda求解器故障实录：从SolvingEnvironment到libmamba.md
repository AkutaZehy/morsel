---
title: Conda 求解器故障实录：从 Solving Environment 到 libmamba
date: 2025-07-22 12:00:00 +0800
categories: [笔记, 开发]
tags: [笔记, 编程, 深度学习, python, conda]
description: 记录在 Mega-NeRF 环境部署中遇到的 Conda 求解、版本与 libmamba 兼容问题，以及最终采用的绕行方案。
---

## 前言

涉及到Conda开发的往期文章：

[Labelme 安装说明（Windows）]({% post_url 2025-03-17-Labelme安装说明（Windows） %})

[Mask2Former 开发排障记录]({% post_url 2025-04-06-Mask2Former开发排障记录 %})

[Conda 开发两则：Conda 与 Pip 的协作边界]({% post_url 2025-07-16-Conda开发两则：Conda与Pip的协作边界 %})

Conda 是常见的跨平台包管理和环境管理工具。这次为 Mega-NeRF 配置环境时，求解过程长时间无响应，随后又遇到版本与 solver 兼容问题，因此记录下这次排查经历。

## 万恶之源：Solving Environment

这一次需要配的Conda环境来自[Mega-NeRF](https://github.com/cmusatyalab/mega-nerf)，事情的导火索来自其初始化步骤的：

`conda env create -f environment.yml`

而这份yml文件的配置如下：

```yaml
name: mega-nerf
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - _libgcc_mutex=0.1=main
  - _openmp_mutex=4.5=1_gnu
  - absl-py=1.0.0=pyhd8ed1ab_0
  - aiohttp=3.7.4.post0=py39h3811e60_0
  - async-timeout=3.0.1=py_1000
  - attrs=21.2.0=pyhd8ed1ab_0
  - blas=1.0=mkl
  - blinker=1.4=py_1
  - brotlipy=0.7.0=py39h3811e60_1001
  - bzip2=1.0.8=h7b6447c_0
  - c-ares=1.17.1=h27cfd23_0
  # 这玩意太长就不接着列了
```

这份配置给出了较完整的版本号，部分依赖甚至精确到了 hash。实际执行时仍卡在 `Solving Environment`，我先按依赖冲突的方向继续排查。

## 从mamba到libmamba

然后这次，想要试试yml安装/更新的方法，把yml简化到了基本不能再简化的地步。

```yaml
name: mega-nerf
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - cudatoolkit=11.3
  - pip
```

这里仍保留了 CUDA，因为在这个项目中由 Conda 管理 CUDA 依赖更方便，但求解过程依旧卡住。由于 Mega-NeRF 的环境文件没有提供太多相关复现信息，我只能暂时把问题范围收敛到当前 Conda 版本、渠道配置和 solver 的组合。

用经典的指令检查了一下目前装在服务器的Anaconda版本，得到的结果是22.9.0，就是说是22年的老东西。

在前面的文章中一笔带过了mamba，它包含了一个相对于Conda而言更有效的solver，有些conda反解不出来或者解起来比较费事的库，用mamba来解就会好很多。抱着这个念头，我想着试试要不要去装个mamba。

给出的安装程序是这样的：

`conda install -c conda-forge mamba`

这个安装过程同样卡在 `Solving Environment`。这说明在当前 Conda 版本和渠道配置下，安装 mamba 的路径仍然无法顺利完成；网络上也有[相似的报告](https://www.cnblogs.com/xjkj/articles/17680330.html)，其中提到可以尝试本地安装。

我测试的 pip 安装路径也没有按预期工作，甚至把 `environment.yml` 识别成了一个“库”。这条路径的兼容性问题更多，因此没有继续采用。

经过检索，有着另外一个替代方案，就是仅使用 Conda 的 `libmamba` solver，同样能够提高速度。安装的方法为：

```bash
conda install conda-libmamba-solver
conda config --set solver libmamba
```

但第二条是不可用的，提示是：

`CondaValueError: Key 'solver' is not a known primitive parameter.`

libmamba 的 [FAQ](https://conda.github.io/conda-libmamba-solver/user-guide/faq/) 对这一问题有说明：参数名称取决于 Conda/libmamba 版本，较新的版本使用 `solver`，旧版本可能需要 `experimental_solver`。

于是又把参数换成了experimental_solver，然而还是不可用，在base（基环境）中尝试使用libmamba时，指出：

`CondaEnvironmentError: LibMambaSolver is not allowed on the base environment during the experimental release phase. Try using it on a non-base environment!  `

## 版本自锁

于是，现在我有两条路走，要么试着用上更新的libmamba，那么需要更新Conda；要么就在需要的环境里面单独配个libmamba，而后者显然是我不期待的（这意味着如果每次都解析出错的话，那么我需要在每个环境中都配一个，无疑是重复的多余工作）。

However，当我尝试用最经典的指令更新Conda时，发生了很诡异的事情：

```bash
(base) 【实际路径】$ conda update -n base -c defaults conda
Collecting package metadata (current_repodata.json): done
Solving environment: done

==> WARNING: A newer version of conda exists. <==
  current version: 22.9.0
  latest version: 25.5.2

Please update conda by running

    $ conda update -n base -c defaults conda


# All requested packages already installed.

Retrieving notices: ...working... done
```

也就是说，标准更新命令完成后仍提示存在新版本，至少在这次执行中没有得到预期的升级结果。

我怀疑问题与 `repodata.json` 有关，但更换了几个源（使用 `--repodata-fn=repodata.json` 参数指定）后仍能复现。就这次环境而言，我推测可能是 strict priority、已有依赖和旧版 Conda 组合造成的版本自锁：在解析 Conda 自身时，当前环境中的库限制了可用版本。由于没有权限进行 fresh install，我没有继续尝试强制降级或指定版本更新。

顺着这条线索去找，在Github上面找到了类似的[issue](https://github.com/conda/conda/issues/12519)，但给出的解决方案是fresh install，我是没有权限的，所以只能作罢。

在指向它的几个 issue 中，我看到了一段颇有代表性的讨论：

![Cover](http://img.akutazehy.xyz/morsel/posts/conda-comment.png)
_没有一个脏字，却充分体现了用户排查这类问题时的无奈_

关于这个问题，还有一个由开发团队于 2024 年 2 月创建、目前仍处于 Planning 状态的 [issue](https://github.com/conda/conda/issues/13570)。其中列出了更多待处理场景，虽然一部分已经解决，但截至本文写作时仍保持 open。

## 无奈的妥协

最后的解决方案就是，走了另外一条路，先用标准的`conda create`创建了环境，引入libmamba，然后把环境顺顺利利地装上去了。

故事的结局是：尝试更新、安装新求解器和升级 Conda 本身后，我回到了更稳妥的办法——先用标准的 `conda create` 创建一个非 base 环境，再在其中引入 libmamba。原本期待在 base 环境中完成的修复，最终需要在新环境里单独验证。

这次经历暴露出旧版本、渠道优先级和 solver 之间的耦合成本：用于改善求解的工具本身，也可能受制于当前环境的求解状态。它更像是一个需要拆开验证的依赖闭环，而不是一个可以仅凭单条命令解决的问题。

这次经历让我发现，环境部署的主要成本并不只来自 Mega-NeRF 本身，还来自等待 `Solving Environment`、检索 issue 以及反复验证版本组合。对这台服务器和这份配置来说，排查时间明显挤压了算法开发时间；如果再次遇到类似情况，我会先确认 Conda 版本、渠道优先级和 solver 配置，再决定新建环境或 fresh install。

这次的 solver 没有去世，我先换了路线：在受限环境中，先建立最小可用环境，再逐步引入求解器和项目依赖，反而更容易定位问题。
