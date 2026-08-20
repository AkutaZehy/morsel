---
title: Labelme 安装说明（Windows）
date: 2025-03-17 12:00:00 +0800
categories: [笔记, 开发]
tags: [笔记, 编程, 深度学习]
description: 在 Windows 上使用 Conda 创建独立环境并安装 Labelme 5.1.1 的步骤与注意事项。
---

因为深度学习要标注数据的缘故，最近又要把尘封已久的labelme拿出来用，但是打不开估计是转移仓库的时候出现问题，只好重装。

However，众所周知，Python的依赖管理是比较糟糕的，尤其臭名昭著的是`numpy`这个库，带这个库的很多库安装都会出问题。一个比较常见的问题是numpy>=2，需要降级到numpy<2。

安装最新版本的 labelme 时，安装程序可能引入较新的依赖，随后出现版本不兼容；更麻烦的是，部分已安装的依赖无法自动回滚，最终表现为 labelme 未能正常安装、冲突依赖却被保留下来。

后面找了个稳定点的版本，基本ok，作为教程给组员用也没出问题。

## 步骤

整个的安装基于conda环境。

1. 配Python环境。
2. 装labelme。

```bash
python -m pip install --upgrade pip
conda create -n labelme python=3.8
conda activate labelme
pip install labelme==5.1.1
```
