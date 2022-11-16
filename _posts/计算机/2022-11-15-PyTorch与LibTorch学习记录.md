---
layout: post
title: PyTorch与LibTorch学习记录
category: 计算机
tags: Linux
keywords: linux
description: 工作与学习中用到的指令
---


## Autograd 简介
 `PyTorch` 大量操作基本可以对齐 `numpy`, 相对好理解。之所以 `PyTorch` 能够在深度学习框架中脱颖而出，很大程度上得益于它简单的自动求导特性（当然还有动态图）。这里看一下它的基础用法：

 ```
 from __future__ import print_function
 import torch
 x = torch.ones((3, 3), requires_grad=True)
 print(x.grad_fn)
 # None  # x 还没有计算历史，它的导数当然是空
 y = x * 2
 print(y.grad)
 # None  # 这是为什么呢？
 y.backward(x)
 x.grad
 # [[2, 2, 2]....]
 ```

上述计算有一些坑，在于 `y` 必须关于 `x` 有一个雅克比矩阵 [矩阵对矩阵的求导](https://soptq.me/2020/06/19/matrix-derivation )。而且，**`grad` 在反向传播过程中是累加的！** 每次运行反向传播，该tensor的梯度都会累加，坑如下：

```
 x = torch.ones((3, 3), requires_grad=True)
 y = x*2
 y2 = x**2
 y.backward(x)
 y2.backward(x)
 x.grad()
 # [[4, 4, 4]....]
```
这里累计到了4！却不是应该有的[[2,2...]...]。 

## CPP版本实现
上述是 `PyTorch` 的简单实现，这里我们看看 `LibTorch` 的大致操作。

```c
#include <iostream>
#include <torch/torch.h>

int main(){
	torch::Tensor x = torch::randint(0, 9, (2, 2), torch::requires_grad());
	std::cout<<"x is:  "<<x[0]<<"  "<<std::endl;
	std::cout<<"   "<<x[1]<<std::endl;
	// [6], [3]
	torch::Tensor y = x * x + 100;
	std::cout<<"y is:  "<<y[0]<<"  "<<y[1]<<std::endl;
	// [136], [109]
	y.backward(x);
	std::cout<<"x grad:  "<<x.grad()[0]<<"    "<<x.grad()[1]<<"    "<<std::endl;
	// [72], [18]
}
```
可以看到结果和 `PyTorch` 的保持一致！