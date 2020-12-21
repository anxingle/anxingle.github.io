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

<div>![](https://github.com/anxingle/anxingle.github.io/blob/master/public/img/ML/svm_latex.png?raw=true)


即可求出超分类面，也就是分类函数:

<p><center><a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;f(x)=W^Tx&plus;b&space;=&space;\sum_{i=1}^m\alpha_{i}y_{i}x_{i}^T&plus;b" target="_blank"><img src="https://latex.codecogs.com/png.latex?\large&space;f(x)=W^Tx&plus;b&space;=&space;\sum_{i=1}^m\alpha_{i}y_{i}x_{i}^T&plus;b" title="\large f(x)=W^Tx+b = \sum_{i=1}^m\alpha_{i}y_{i}x_{i}^T+b" /></a></center></p>
这是基本的SVM推导，如果这一步还不知道怎么来的，那就需要拿[《西瓜书》](https://book.douban.com/subject/26708119/)好好推导一下了。
<center><embed src="https://raw.githubusercontent.com/anxingle/Exam/dac71b6b54ac42b41cc76cb8996c030d18f58c26/pic/SMO.pdf" width="900" height="400"></center>

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

根据Platt的原文提供的伪代码，我们有如下实现：

```
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.datasets import make_blobs,make_circles,make_moons
from sklearn.preprocessing import StandardScaler

class SMOStruct:
	""" 按照John Platt的论文构造SMO的数据结构"""
	def __init__(self, X, y, C, kernel, alphas, b, errors, tol, eps, user_linear_optim, loop_threshold=2000):
		self.X = X              # 训练样本
		self.y = y              # 类别 label
		self.C = C              # 正则化常量，用于调整（过）拟合的程度
		self.kernel = kernel    # 核函数，实现了两个核函数，线性和高斯（RBF）
		self.alphas = alphas    # 拉格朗日乘子，与样本一一相对
		self.b = b              # 标量，偏移量
		self.errors = errors    # 用于存储alpha值实际与预测值得差值，与样本数量一一相对
		self.tol = tol          # error tolerance
		self.eps = eps          # alpha tolerance

		self.m, self.n = np.shape(self.X)  # 训练样本的个数m 和 每个样本的features数量n

		self.user_linear_optim = user_linear_optim    # 判断模型是否使用线性核函数
		self.w = np.zeros(self.n)     # 初始化权重w的值，主要用于线性核函数
		#self.b = 0
		self.loop_threshold = loop_threshold
		# 差值矩阵
		initial_error = SMOStruct.decision_function(self.alphas, self.y, self.kernel, self.X, self.X, self.b) - self.y
		self.errors = initial_error

	# 判别函数一: f(x)=WX-b, 用于单一样本
	@classmethod
	def decision_function_output(cls, model, i):
		if model.user_linear_optim:
			# Equation (J1)
			# 返回在平面中的位置(f(x)=WX-b)
			return float(model.w.T @ model.X[i]) - model.b
		else:
			# Equation (J10)
			return np.sum([model.alphas[j] * model.y[j] * model.kernel(model.X[j], model.X[i]) for j in range(model.m)]) - model.b

	# 判别函数二，用于多个样本
	@staticmethod
	def decision_function(alphas, target, kernel, X_train, x_test, b):
		""" Applies the SVM decision function to the input feature vectors in 'x_test'. """
		result = (alphas * target) @ kernel(X_train, x_test) - b   # *，@ 两个Operators的区别?
		return result


	# 选择了alpha2, alpha1后开始第一步优化，然后迭代
	# 主要的优化步骤在这里发生
	def take_step(self, i1, i2):
		# 如果两个 alphas 相同，跳过
		if i1 == i2:
			return 0
		# alpha1, alpha2, y1, y2, E1, E2, s 都是论文中出现的变量，含义与论文一致
		alpha1 = self.alphas[i1]
		alpha2 = self.alphas[i2]

		y1 = self.y[i1]
		y2 = self.y[i2]

		E1 = get_error(self, i1)
		E2 = get_error(self, i2)
		s = y1 * y2

		# 计算alpha的边界，L, H
		if(y1 != y2):
			# y1,y2 异号，使用 Equation (J13)
			L = max(0, alpha2 - alpha1)
			H = min(self.C, self.C + alpha2 - alpha1)
		elif (y1 == y2):
			# y1,y2 同号，使用 Equation (J14)
			L = max(0, alpha1 + alpha2 - self.C)
			H = min(self.C, alpha1 + alpha2)
		if L == H:
			return 0

		# 分别计算样本1, 2对应的核函数组合，目的在于计算eta  也就是求一阶导数后的值，目的在于计算a2_new
		k11 = self.kernel(self.X[i1], self.X[i1])
		k12 = self.kernel(self.X[i1], self.X[i2])
		k22 = self.kernel(self.X[i2], self.X[i2])
		# 计算 eta，equation (J15)
		eta = k11 + k22 - 2*k12

		# 如论文中所述，分两种情况根据 eta 为正还是为负或0来计算计算 a2_new

		if(eta>0): 
			# equation (J16) 计算alpha2
			a2 = alpha2 + y2 * (E1 - E2)/eta
			# 把a2夹到限定区间 equation （J17）
			if L < a2 < H:
				a2 = a2
			elif (a2 <= L):
				a2 = L
			elif (a2 >= H):
				a2 = H
		# TODO: 这里还没有搞懂
		# if eta is non-positive, move new a2 to bound with greater objective function value
		else:
			# Equation （J19）
			# 在特殊情况下，eta可能不为正
			f1 = y1*(E1 + self.b) - alpha1*k11 - s*alpha2*k12
			f2 = y2*(E2 + self.b) - s*alpha1*k12 - alpha2*k22

			L1 = alpha1 + s*(alpha2 - L)
			H1 = alpha1 + s*(alpha2 - H)

			Lobj = L1 * f1 + L * f2 + 0.5 * (L1 ** 2) * k11 + 0.5 * (L**2) * k22 + s * L * L1 * k12

			Hobj = H1 * f1 + H * f2 + 0.5 * (H1**2) * k11 + 0.5 * (H**2) * k22 + s * H * H1 * k12

			if Lobj < Hobj - self.eps:
				a2 = L
			elif Lobj > Hobj + self.eps:
				a2 = H
			else:
				a2 = alpha2

		# 当 new_a2 接近C或0时候，就让它等于C或0
		if a2 <1e-8:
			a2 = 0.0
		elif a2 > (self.C - 1e-8):
			a2 = self.C

		#If examples can't be optimized within epsilon(eps), skip this pair
		if (np.abs(a2 - alpha2) < self.eps * (a2 + alpha2 + self.eps)):
			return 0

		#根据新 a2 计算新 a1 Equation(J18)
		a1 = alpha1 + s * (alpha2 - a2)

		#更新 bias b的值 Equation (J20)
		b1 = E1 + y1*(a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12 + self.b
		#equation (J21)
		b2 = E2 + y1*(a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 + self.b

		# Set new threshoold based on if a1 or a2 is bound by L and/or H
		if 0 < a1 and a1 < C:
			b_new = b1
		elif 0 < a2 and a2 < C:
			b_new = b2
		# Average thresholds if both are bound
		else:
			b_new = (b1 + b2) * 0.5

		# 更新模型的 b
		self.b = b_new

		# 当所训练模型为线性核函数时
		# Equation (J22) 计算w的值
		if self.user_linear_optim:
			 self.w = self.w + y1 * (a1 - alpha1)*self.X[i1] + y2 * (a2 - alpha2) * self.X[i2]
		# 在 alphas 矩阵中分别更新a1, a2的值
		self.alphas[i1] = a1
		self.alphas[i2] = a2

		# 优化完了，更新差值矩阵的对应值;  同时更新差值矩阵其它值
		self.errors[i1] = 0
		self.errors[i2] = 0

		# 更新差值 Equation (12)
		for i in range(self.m):
			if 0 < self.alphas[i] < self.C:
				self.errors[i] += y1*(a1 - alpha1) * self.kernel(self.X[i1], self.X[i]) + \
								y2*(a2 - alpha2) * self.kernel(self.X[i2], self.X[i]) + self.b - b_new

		return 1

	def examine_example(self, i2):
		"""
		启发式寻找 alpha1
		"""
		y2 = self.y[i2]
		alpha2 = self.alphas[i2]
		E2 = get_error(self, i2)
		r2 = E2 * y2

		#XXX: 这一段的重点在于确定 alpha1, 也就是 old_a1，并送到 take_step 去分析优化
		# 下面条件之一满足，进入 if 开始找第二个 alpha，送到 take_step 进行优化
		if ((r2 < -self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0)):
			if len(self.alphas[(self.alphas != 0) & (self.alphas != self.C)]) > 1:
				# 选择Ei矩阵中差值最大的先进行优化
				# 要想|E1-E2|最大，只需要在E2为正时，选择最小的Ei作为E1
				# 在E2为负时选择最大的Ei作为E1
				if self.errors[i2] > 0:
					i1 = np.argmin(self.errors)
				elif self.errors[i2] <= 0:
					i1 = np.argmax(self.errors)

				step_result = self.take_step(i1, i2)
				if step_result:
					return 1

			# 循环所有非0 非C alphas值进行优化，随机选择起始点
			for i1 in np.roll(np.where((self.alphas != 0) & (self.alphas != self.C))[0],
							  np.random.choice(np.arange(self.m))):
				step_result = self.take_step(i1, i2)
				if step_result:
					return 1

			#a2确定的情况下，如何选择a1? 循环所有(m-1) alphas, 随机选择起始点
			for i1 in np.roll(np.arange(self.m), np.random.choice(np.arange(self.m))):
				#print("what is the first i1",i1)
				step_result = self.take_step(i1, i2)

				if step_result:
					return 1
		#先看最上面的if语句，如果if条件不满足，说明KKT条件已满足，找其它样本进行优化，则执行下面这句，退出
		return 0

	def fit(self):
		"""
		确定第一个alpha: alpha2
		"""
		numChanged = 0 # alpha 更新量
		examineAll = True # 需要遍历所有样本

		# 计数器，记录优化时的循环次数
		loopnum = 0
		loopnum1 = 0
		loopnum2 = 0

		"""
		当numChanged = 0(alpha没有更新) and examineAll(不再需要遍历样本)时 循环退出
		实际是顺序地执行完所有的样本 (第一个if中的循环)
		else中的循环没有可优化的 alpha, 目标函数收敛了; 且在容差之内, 并且满足KKT条件; 则循环退出，如果执行3000次循环仍未收敛, 也退出
		"""

		# XXX: 确定 alpha2，也就是old_a2, 或者说alpha2的下标，old_a2 和 old_a1都是启发式选择
		while(numChanged > 0) or (examineAll): 
			numChanged = 0
			if loopnum == self.loop_threshold: # 循环量大于最大阈值
				break
			loopnum = loopnum + 1

			if examineAll:
				loopnum1 = loopnum1 + 1 # 记录顺序一个一个选择 alpha2 时的循环次数
				# 从0,1,2,3,...,m顺序选择 a2 的，送给 examine_example 选择 alpha1，总共 m*(m-1)种选法
				for i in range(self.alphas.shape[0]): 
					examine_result = self.examine_example(i) # 是否发生改变
					numChanged += examine_result
			else:  # 上面if里m(m-1)执行完的后执行 
				loopnum2 = loopnum2 + 1
				# 遍历样本中 alphas 未更新的值
				for i in np.where((self.alphas != 0) & (self.alphas != self.C))[0]:
					examine_result = self.examine_example(i)
					numChanged += examine_result

			if examineAll:
				examineAll = False
			elif numChanged == 0:
				examineAll = True
		print(" loopnum012: %s, : %s,  : %s" % (loopnum, loopnum1, loopnum2))


def linear_kernel(x, y, b=1):
	#线性核函数
	""" returns the linear combination of arrays 'x' and 'y' with
	the optional bias term 'b' (set to 1 by default). """
	result = x @ y.T + b
	return result # Note the @ operator for matrix multiplications


def gaussian_kernel(x, y, sigma=1):
	#高斯核函数
	"""Returns the gaussian similarity of arrays 'x' and 'y' with
	kernel width paramenter 'sigma' (set to 1 by default)"""

	if np.ndim(x) == 1 and np.ndim(y) == 1:
		result = np.exp(-(np.linalg.norm(x-y,2))**2/(2*sigma**2))
	elif(np.ndim(x)>1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y)>1):
		result = np.exp(-(np.linalg.norm(x-y, 2, axis=1)**2)/(2*sigma**2))
	elif np.ndim(x) > 1 and np.ndim(y) > 1 :
		result = np.exp(-(np.linalg.norm(x[:, np.newaxis]- y[np.newaxis, :], 2, axis = 2) ** 2)/(2*sigma**2))
	return result






def get_error(model, i1):
	if 0< model.alphas[i1] <model.C:
		return model.errors[i1]
	else:
		return SMOStruct.decision_function_output(model, i1) - model.y[i1]

def plot_decision_boundary(model, ax, resolution=100, colors=('b','k','r'), levels = (-1, 0, 1)):
	"""
	绘出分割平面及支持平面，
	用的是等高线的方法，怀疑绘制的分割平面和支持平面的准确性
	"""
	# Generate coordinate grid of shape [resolution x resolution]
	# and evalute the model over the entire space
	x_range = np.linspace(model.X[:, 0].min(), model.X[:, 0].max(), resolution)
	yrange = np.linspace(model.X[:, 1].min(), model.X[:, 1].max(), resolution)
	grid = [[SMOStruct.decision_function(model.alphas, model.y, model.kernel, model.X,
							   np.array([xr,yr]), model.b) for xr in x_range] for yr in yrange]

	grid = np.array(grid).reshape(len(x_range), len(yrange))

	# Plot decision contours using grid and make a scatter plot of training data
	ax.contour(x_range, yrange, grid, levels=levels, linewidths = (1,1,1),
				linestyles=('--', '-', '--'), colors=colors)
	ax.scatter(model.X[:, 0], model.X[:, 1],
				c=model.y, cmap=plt.cm.viridis, lw=0, alpha =0.25)

	# Plot support vectors (non-zero alphas) as circled points (linewidth >0)
	mask = np.round(model.alphas, decimals = 2) !=0.0
	ax.scatter(model.X[mask, 0], model.X[mask, 1],
				c=model.y[mask], cmap=plt.cm.viridis, lw=1, edgecolors='k')

	return grid, ax


if __name__ == '__main__':
	# 生成测试数据，训练样本
	X_train, y = make_blobs(n_samples = 1000, centers =2, n_features=2, random_state = 2)
	# StandardScaler()以及fit_transfrom函数的作用需要解释一下
	scaler = StandardScaler()   #数据预处理，使得经过处理的数据符合正态分布，即均值为0，标准差为1
	X_train_scaled = scaler.fit_transform(X_train, y)
	y[y == 0] = -1

	# 将训练数据绘制出来
	fig, ax = plt.subplots()
	ax.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
					c=y, cmap=plt.cm.viridis, lw=0, alpha =0.25)
	plt.show()
	plt.savefig('./train_data.jpg')


	# SVM超参数
	C = 20.0 # 软间隔-惩罚因子
	m = len(X_train_scaled) # 样本数
	initial_alphas = np.zeros(m) # 初始化 alphas
	initial_b = 0.0 # 初始化 b
	tol = 0.01 # error 容忍度
	eps = 0.01 # alpha 容忍度

	model = SMOStruct(X_train_scaled, y, C, linear_kernel, initial_alphas, initial_b, np.zeros(m), tol, eps, user_linear_optim=True)

	print("starting to fit...")
	start = time.time()
	# 开始训练
	model.fit()
	print("fit cost time: ", time.time()-start)
	# 绘制训练完，找到分割平面的图
	fig, ax = plt.subplots()
	grid, ax = plot_decision_boundary(model, ax)
	plt.show()
	plt.savefig('./svm.jpg')
```

