---
layout: post
title: 手撕SVM代码之numpy低效实现SMO算法
category: 科研
tags: 机器学习
keywords: 机器学习,SVM
description: 面试遇到过的问题
---

(本文使用https://www.codecogs.com/进行latex渲染，若latex公式有问题，请开启全局梯子)

一直以来SVM都停留在一看就大概知道，一上手推导就傻眼的阶段。实际上还是没有真正地理解了SVM的精髓，对于 **关键定理** 这里总是囫囵吞枣。看一看网上大部分的教程，也都是以推导为主，所以想手写实现一下，加深理解。

> talk is cheap, show me the code.

假设我们已经艰难地推导到了最后，得到了SVM的对偶问题上：

<p><center><a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;argmax\sum_{i=1}^n{\alpha_{i}}-\frac{1}{2}{\sum_{i=1}^{n}}{\sum_{j=1}^{n}}y_{i}y_{j}\alpha_{i}\alpha_{j}\left&space;\langle&space;x_{i},x_{j}&space;\right&space;\rangle\\&space;s.t.&space;\alpha_{i}\geq0,&space;\sum_{i=1}^{N}\alpha_{i}y_{i}=0" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;argmax\sum_{i=1}^n{\alpha_{i}}-\frac{1}{2}{\sum_{i=1}^{n}}{\sum_{j=1}^{n}}y_{i}y_{j}\alpha_{i}\alpha_{j}\left&space;\langle&space;x_{i},x_{j}&space;\right&space;\rangle\\&space;s.t.&space;\alpha_{i}\geq0,&space;\sum_{i=1}^{N}\alpha_{i}y_{i}=0" title="\large argmax\sum_{i=1}^n{\alpha_{i}}-\frac{1}{2}{\sum_{i=1}^{n}}{\sum_{j=1}^{n}}y_{i}y_{j}\alpha_{i}\alpha_{j}\left \langle x_{i},x_{j} \right \rangle\\ s.t. \alpha_{i}\geq0, \sum_{i=1}^{N}\alpha_{i}y_{i}=0" /></a></center></p>

这时候，我们需要求解一系列的<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\alpha_{1},\alpha_{2},\alpha_{3}..." target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;\alpha_{1},\alpha_{2},\alpha_{3}..." title="\large \alpha_{1},\alpha_{2},\alpha_{3}..." /></a>值了，只要得到了 <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\alpha_{1}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;\alpha_{1}" title="\large \alpha_{1}" /></a>的值，那么W,b 就好求了。[《西瓜书》](https://book.douban.com/subject/26708119/)上也就到了这里，告诉我们这是一个二次规划问题（这我也不会啊😭）。顺理成章地引入了 SMO 算法来求解<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\alpha_{1}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;\alpha_{1}" title="\large \alpha_{1}" /></a>，得到了<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\alpha_{1}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;\alpha_{1}" title="\large \alpha_{1}" /></a>后，对W,b进行更新迭代：

![](https://github.com/anxingle/anxingle.github.io/blob/master/public/img/ML/svm_latex.png?raw=true)


即可求出超分类面，也就是分类函数:

<p><center><a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;f(x)=W^Tx&plus;b&space;=&space;\sum_{i=1}^m\alpha_{i}y_{i}x_{i}^T&plus;b" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;f(x)=W^Tx&plus;b&space;=&space;\sum_{i=1}^m\alpha_{i}y_{i}x_{i}^T&plus;b" title="\large f(x)=W^Tx+b = \sum_{i=1}^m\alpha_{i}y_{i}x_{i}^T+b" /></a></center></p>
这是基本的SVM推导，如果这一步还不知道怎么来的，那就需要拿[《西瓜书》](https://book.douban.com/subject/26708119/)好好推导一下了。

我们回过头来继续看SMO算法如何来求解的<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\alpha_{1,2,3...}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;\alpha_{1,2,3...}" title="\large \alpha_{1,2,3...}" /></a> ，实际上这里我也不能明白SMO的精髓，只能按着 [wikipedia SMO序列最小优化算法](https://zh.wikipedia.org/wiki/序列最小优化算法中)  介绍的的流程来进行计算了😏。

![](https://github.com/anxingle/anxingle.github.io/blob/master/public/img/ML/smo.jpeg?raw=true)

其中，$L$和$H$分别是$\alpha_{2}^{new}$ 的下界和上界。特别地，有：
![](https://github.com/anxingle/anxingle.github.io/blob/master/public/img/ML/smo2.jpeg?raw=true)

推导过程实在繁琐，我们这里直接拿作者的解析解：

> 记  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\eta=K_{1}^1&space;&plus;&space;K_{2}^2&space;-&space;2K_{1,2}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;\eta=K_{1}^1&space;&plus;&space;K_{2}^2&space;-&space;2K_{1,2}" title="\large \eta=K_{1}^1 + K_{2}^2 - 2K_{1,2}" /></a>, 这里<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;K(x1,&space;x2)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;K(x1,&space;x2)" title="\large K(x1, x2)" /></a>为核函数，用于将低维空间的数据映射到高维空间，我们以后再谈；

> 记<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;E_{i}=f(x_{i})-y_{i}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;E_{i}=f(x_{i})-y_{i}" title="\large E_{i}=f(x_{i})-y_{i}" /></a>,也就是更新后的SVM预测值与真实值的误差。

得到：
<p><center><a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\alpha_{2}^{new}&space;=&space;\alpha_{2}^{old}&plus;{y_{2}}\frac{&space;(E_{1}&space;-&space;E_{2})&space;}{\eta}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;\alpha_{2}^{new}&space;=&space;\alpha_{2}^{old}&plus;{y_{2}}\frac{&space;(E_{1}&space;-&space;E_{2})&space;}{\eta}" title="\large \alpha_{2}^{new} = \alpha_{2}^{old}+{y_{2}}\frac{ (E_{1} - E_{2}) }{\eta}" /></a></center></p>

此时未考虑约束条件下的解，如果考虑约束：
<p><center><a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\alpha_{1}y_{1}&plus;\alpha_{2}y_{2}=-\sum_{i=1}^n{\alpha_{i}{y_{i}}}=\zeta;&space;0\leq\alpha_{i}\leq{C}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;\alpha_{1}y_{1}&plus;\alpha_{2}y_{2}=-\sum_{i=1}^n{\alpha_{i}{y_{i}}}=\zeta;&space;0\leq\alpha_{i}\leq{C}" title="\large \alpha_{1}y_{1}+\alpha_{2}y_{2}=-\sum_{i=1}^n{\alpha_{i}{y_{i}}}=\zeta; 0\leq\alpha_{i}\leq{C}" /></a></center></p>

即可得上下界：
<p><center><a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\text{if:}&space;y1\neq&space;y2\\\\&space;\left\{\begin{aligned}&space;L=max(0,&space;\alpha_{2}^{old}-\alpha_{1}^{old}),\\&space;H=min(C,&space;C&plus;\alpha_{2}^{old}-\alpha_{1}^{old})&space;\end{aligned}&space;\right." target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;\text{if:}&space;y1\neq&space;y2\\\\&space;\left\{\begin{aligned}&space;L=max(0,&space;\alpha_{2}^{old}-\alpha_{1}^{old}),\\&space;H=min(C,&space;C&plus;\alpha_{2}^{old}-\alpha_{1}^{old})&space;\end{aligned}&space;\right." title="\large \text{if:} y1\neq y2\\\\ \left\{\begin{aligned} L=max(0, \alpha_{2}^{old}-\alpha_{1}^{old}),\\ H=min(C, C+\alpha_{2}^{old}-\alpha_{1}^{old}) \end{aligned} \right." /></a></center></p>

以及：
<p><center><a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\text{if:}&space;{y1}=&space;{y2}\\\\&space;\left\{\begin{aligned}&space;L=max(0,&space;\alpha_{2}^{old}&plus;\alpha_{1}^{old}-C),\\&space;H=min(C,&space;C&plus;\alpha_{2}^{old}&plus;\alpha_{1}^{old})&space;\end{aligned}&space;\right." target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;\text{if:}&space;{y1}=&space;{y2}\\\\&space;\left\{\begin{aligned}&space;L=max(0,&space;\alpha_{2}^{old}&plus;\alpha_{1}^{old}-C),\\&space;H=min(C,&space;C&plus;\alpha_{2}^{old}&plus;\alpha_{1}^{old})&space;\end{aligned}&space;\right." title="\large \text{if:} {y1}= {y2}\\\\ \left\{\begin{aligned} L=max(0, \alpha_{2}^{old}+\alpha_{1}^{old}-C),\\ H=min(C, C+\alpha_{2}^{old}+\alpha_{1}^{old}) \end{aligned} \right." /></a></center></p>

则最终的<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\alpha_{2}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;\alpha_{2}" title="\large \alpha_{2}" /></a>更新公式为:
<p><center><a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\alpha_{2}^{new}=&space;\left\{\begin{array}{l}&space;H,&space;\qquad\qquad\alpha_{2}^{new,&space;unclipped}\geq{H}\\&space;\alpha_{2}^{new,&space;unclipped},&space;L\leq\alpha_{2}^{new,&space;unclipped}\leq{H}\\&space;L,&space;\qquad\qquad\alpha_{2}^{new,&space;unclipped}\leq{L}\\&space;\end{array}&space;\right." target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;\alpha_{2}^{new}=&space;\left\{\begin{array}{l}&space;H,&space;\qquad\qquad\alpha_{2}^{new,&space;unclipped}\geq{H}\\&space;\alpha_{2}^{new,&space;unclipped},&space;L\leq\alpha_{2}^{new,&space;unclipped}\leq{H}\\&space;L,&space;\qquad\qquad\alpha_{2}^{new,&space;unclipped}\leq{L}\\&space;\end{array}&space;\right." title="\large \alpha_{2}^{new}= \left\{\begin{array}{l} H, \qquad\qquad\alpha_{2}^{new, unclipped}\geq{H}\\ \alpha_{2}^{new, unclipped}, L\leq\alpha_{2}^{new, unclipped}\leq{H}\\ L, \qquad\qquad\alpha_{2}^{new, unclipped}\leq{L}\\ \end{array} \right." /></a></center></p>

得到  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\alpha_{2}^{new}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;\alpha_{2}^{new}" title="\large \alpha_{2}^{new}" /></a> 后，<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\alpha_{1}^{new}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;\alpha_{1}^{new}" title="\large \alpha_{1}^{new}" /></a>也可以同样求出来：

<p><center><a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\alpha_{1}^{new}=\alpha_{1}^{old}&space;&plus;&space;y_{1}y_{2}(\alpha_{2}^{old}-\alpha_{2}^{new})" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;\alpha_{1}^{new}=\alpha_{1}^{old}&space;&plus;&space;y_{1}y_{2}(\alpha_{2}^{old}-\alpha_{2}^{new})" title="\large \alpha_{1}^{new}=\alpha_{1}^{old} + y_{1}y_{2}(\alpha_{2}^{old}-\alpha_{2}^{new})" /></a></center></p>然后循环对所有的<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\alpha_{i}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;\alpha_{i}" title="\large \alpha_{i}" /></a>进行选取，便可以进行优化更新了。

看看Platt的原文提供的伪代码：

![](https://github.com/anxingle/anxingle.github.io/blob/master/public/img/ML/smo_0.jpeg?raw=true)

![](https://github.com/anxingle/anxingle.github.io/blob/master/public/img/ML/smo_1.jpeg?raw=true)

![](https://github.com/anxingle/anxingle.github.io/blob/master/public/img/ML/smo_2.jpeg?raw=true)

我们

$$
W=\sum_{i=1}^{n}{\alpha_{i}}{y_{i}}{x_{i}},\\
\sum_{i=1}^{n}{{\alpha_{i}}{y_{i}}}=0
$$