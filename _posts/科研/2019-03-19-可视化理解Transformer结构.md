---
layout: post
title:  Attention is all you need---可视化理解Transformer结构
category: 科研
tags: 计算机
keywords: 
description: 
---

# Attention is all you need---可视化理解Transformer结构

在[上一篇文章](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)中我们介绍了注意力机制—目前在深度学习中被广泛应用。注意力机制能够显著提高神经机器翻译任务的性能。本文将会看一看**Transformer**---加速训练注意力模型的方法。Transformers在很多特定任务上已经优于Google神经机器翻译模型了。不过其最大的优点在于它的并行化训练。Google云强烈建议使用[TPU云](https://cloud.google.com/tpu/)提供的Transformer模型。我们赶紧撸起袖子拆开模型看一看内部究竟如何吧。

文章[Attention is All You Need](https://arxiv.org/abs/1706.03762) 首次提出该模型。[Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) 库包含了模型的一份Tensorflow实现。哈佛NLP组也提供了一份详细注释的[PyTorch实现](http://nlp.seas.harvard.edu/2018/04/03/attention.html)。我们在这里尽可能的简化模型，一个一个地介绍关键概念，希望能够让没有较深相关知识的同学也能够轻松理解。

## 俯瞰全局

首先我们将模型视为一个单独的黑盒。在机器翻译任务中，它将接收一种语言的一个句子，输出另一种语言的对应翻译。

![黑盒](https://jalammar.github.io/images/t/the_transformer_3.png)

打开核心部分，你会看到编码组件，解码组件，以及他们之间的连接。

![](https://jalammar.github.io/images/t/The_transformer_encoders_decoders.png)

编码组件是一系列编码器的堆叠（文章中是6个编码器的堆叠——没什么特别的，你也可以试试其他的数字）。解码部分也是同样的堆叠数。

![](https://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png)

编码器在结构上都是一样的（但是它们不共享权重）。每个都可以分解成两个子模块：

![](https://jalammar.github.io/images/t/Transformer_encoder.png)

编码器的输入首先流经self-attention层，该层有助于编码器对特定单词编码时查看输入序列的其他单词。本文后面将会详细介绍self-attention。

Self-attention层的输出被送入前馈神经网络。完全相同的前馈神经网络独立应用在每个位置。

解码器也具有这两层，但是这两层中间还插入了attention层，能帮助解码器注意输入句子的相关部分（和[seq2seq模型](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)的attention相同）。

![](https://jalammar.github.io/images/t/Transformer_decoder.png)

## 将张量带入图片

现在我们看到了模型的主要部件，我们现在开始研究各种向量/张量以及他们如何在这些组件中流动来将训练好的模型的输入转换为输出。

和传统NLP任务一样，我们首先使用[词嵌入](https://medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca)来将每个输入单词转换为向量。

<center>     <img style="border-radius: 0.3125em;     box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"      src="https://jalammar.github.io/images/t/embeddings.png">     <br>     <div style="color:orange; border-bottom: 1px solid #d9d9d9;     display: inline-block;     color: #999;     padding: 2px;">每个词被映射为512大小的向量。我们将这些向量表示成如上所示的盒子</div> </center>

词嵌入仅发生在最底部的编码器中。所有编码器的共同之处是他们接收元素大小为512的向量列表——在最底部的编码器中这恰好是词嵌入后的大小，而在其他的编码器中这恰好是其下面编码器输出的大小。这个列表大小是我们设置的超参数----基本上它就是训练集中最长句子的大小。

在输入序列中进行词嵌入后，每一个输入都将会流过编码器的两个层。

![](https://jalammar.github.io/images/t/encoder_with_tensors.png)

这里我们看到Transformer一个重要特性，每个位置的单词在经过编码器时流经自己的路径。self-attention层中这些路径之间有依赖关系。然而前馈层并不具有这些依赖关系，所以各种路径在流经前馈层时可以并行执行。

下面我们将例子中句子换为更短的句子来看一下每个编码器中的子层发生了什么。

## 开始编码

上面提到过，编码器接受向量列表作为输入。编码器将向量列表传入self-attention层，之后进入前馈神经网络，然后再输出到下一个编码器。

<center>     <img style="border-radius: 0.3125em;     box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"      src="https://jalammar.github.io/images/t/encoder_with_tensors_2.png"><br> <div style="color:orange; border-bottom: 1px solid #d9d9d9;     display: inline-block;     color: #999;     padding: 2px;">每个位置的单词通过self-attention层，之后它们每个都流经前馈神经网络———每个向量分别单独流经完全相同的网络</div> </center>

## 更高的视角看self-attention

别被“self-attention”这么高大上的词给唬住了，乍一听好像每个人都应该熟悉一样。我在读文章[Attention is All You Need](https://arxiv.org/abs/1706.03762) 之前就没有听过这个词。先看看它如何工作吧。

假设下面这句话是我们想翻译的：

```
"The animal didn't cross the street because it was too tired"
```

句子中“it”指的是什么？指street还是说animal呢？对人来说很简单的问题，对机器却很复杂。

当模型处理单词“it”时，self-attention 就可以使它指代“animal”。

当模型处理每个单词时（输入序列中每个位置），self-attention使得它可以查看输入序列的其他位置以便于更好的编码该单词。

如果你熟悉RNN，考虑一下如何维护隐藏层状态来更好的结合已经处理的先前的单词/向量与目前正在处理的单词/向量。Transformer使用self-attention来将其他相关单词的“理解”融入到目前正在处理的单词。
<center><img style="border-radius: 0.3125em;     box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"      src="https://jalammar.github.io/images/t/transformer_self-attention_visualization.png"><br> <div style="color:orange; border-bottom: 1px solid #d9d9d9;     display: inline-block;     color: #999;     padding: 2px;">在编码器处理单词“it”时，注意力机制的一部分集中注意力在“The Animal”上，将这部分的含义融入了“it”中。</div> </center>

务必查看加载Transformer模型后的 [Tensor2Tensor Notebook](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb) ，并以交互式可视化方式进行检查。

## Self-Attention 细节

首先我们看看如何使用向量计算self-Attention，之后再研究它如何实现的———使用矩阵实现呗。

计算self-attention的 **第一步** 需要从每个编码器的输入向量（这个例子中是每个词的词嵌入表示）创建三个向量。因此，对于每个单词，我们创建一个Query向量，一个Key向量和一个Value向量。这些向量是通过将**词嵌入**(embedding)乘以在训练过程中训练的三个矩阵来创建的。

**注意**，这些新创建的向量的维度小于**词嵌入向量**(embedding vector)。它们（新创建的向量）的维度是64，而词嵌入和编码器的输入输出向量的维度是512。它们不必更小，这是一种架构选择，可以使多头注意力(multiheaded attention)计算不变。

<center><img style="border-radius: 0.3125em;     box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"      src="https://jalammar.github.io/images/t/transformer_self_attention_vectors.png"><br> <div style="color:orange; border-bottom: 1px solid #d9d9d9;     display: inline-block;     color: #999;     padding: 2px;">将X1乘以矩阵WQ将产生q1。也就是与该单词有关的“查询”向量。最终在输入句子中创建了每个单词的“query”，“key”，“value”向量。</div> </center>

那么，究竟什么是“query”，“key”，“value”向量呢？

这些抽象概念有助于思考和计算注意力。一旦你继续读下面如何计算注意力，你将会清楚的知道每个向量的作用。

计算self-attention的 **第二步** 是计算得分(score 权重)。假设我们正在计算例子中第一个单词"Thinking"的self-attention。我们需要根据这个词对输入句子的每个词进行评分。当我们在某个位置编码单词时，分数决定了对输入句子的其他部分放置多少的焦点(注意力)。

这里的分数是通过将<span color="#B36AE2">"query"向量</span>与我们正在评分的单词的<span color="#F39019">“key”向量</span>做点积来得到。所以如果我们计算位置<span color="#70BF41">#1</span>处的单词的self-attention，第一个得分就是就是<span color="#B36AE2">q1</span>和<span color="#F39019">k1</span>的点积。第二个得分是<span color="#B36AE2">q1</span>和<span color="#F39019">k2</span>的点积。

![](https://jalammar.github.io/images/t/transformer_self_attention_score.png)

**第三第四** 是将分数除以8（论文中使用“Key”向量维数的平方根---64。这可以有更稳定的梯度。实际上还可以有其他可能的值，这里使用默认值），然后经过一个softmax操作后输出结果。Softmax可以将分数归一化，这样使得结果都是正数并且加起来等于1。

![](https://jalammar.github.io/images/t/self-attention_softmax.png)

softmax后的分数决定了每个单词在这个位置被表达了多少。很明显该位置的这个词具有最高的softmax分数，但是有时候关注与当前单词相关的其它词更有用。

**第五步** 将每个值向量乘以softmax得分（准备将他们相加）。直觉上讲需要保持我们关注的单词的值不变，忽略掉不相关的单词（比如可以将它们乘以0.001这样的小数字）。

**第六步** 对加权值向量求和。这样就产生了在这个位置的self-attention的输出（对于第一个单词）。

![](https://jalammar.github.io/images/t/self-attention-output.png)

这就是self-attention计算。得到的向量可以送往前馈神经网络。然而在真正的实现中，计算过程通过矩阵计算来进行，以便加快计算。现在我们已经清楚了单词级别的计算过程。

## Self-Attention的矩阵计算

**第一步** 是计算Query, Key, Value矩阵。通过将词嵌入整合到矩阵<span color="#70BF41">X</span>中，并将其乘以我们训练过的权重矩阵(<span color="#B36AE2">WQ</span>，<span color="#F39019">WK</span>，<span color="#5CBCE9">WV</span>)来实现。

<center><img style="border-radius: 0.3125em;     box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"      src="https://jalammar.github.io/images/t/self-attention-matrix-calculation.png"><br> <div style="color:orange; border-bottom: 1px solid #d9d9d9;     display: inline-block;     color: #999;     padding: 2px;">矩阵X中每行对应于输入序列中某个单词。我们又看到了词嵌入向量的大小差异（图中是4个方框，现实往往是512）,也看到了 q/v/k 向量（图中是3个方框，现实往往是64）</div></center>

**最后** ，由于我们在处理矩阵，我们可以将步骤2到步骤6合并为一个公式来计算self-attention层的输出。

<center><img style="border-radius: 0.3125em;     box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"      src="https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png"><br> <div style="color:orange; border-bottom: 1px solid #d9d9d9;     display: inline-block;     color: #999;     padding: 2px;">公式表达self-attention层的计算过程</div></center>

## 多头注意力机制

这篇文章通过增加一种称为“多头”注意力的机制完善了self-attention层。这通过两种方式改善了注意力层的性能：

1. 它扩展了模型关注不同位置的能力。在上面的例子中，Z1包含了每个其他编码的一点，但它可能由实际的单词本身支配。翻译句子：“The animal didn’t cross the street because it was too tired”，我们很想知道这里的“it”指代什么？ 这时候会很有帮助。

2. 它给予attention层多个“表达子空间”。接下来会看到多头注意力有多组Query/ Key /Value权重矩阵（Transformer使用了8组注意力头，所以这里我们为每个编码器/解码器设置了8组），而不是简单的一组。每组集合都是随机初始化。之后在训练中每组用于将词嵌入（或来自较低层编码器/解码器的输出）映射到不同的表达子空间。

<center><img style="border-radius: 0.3125em;     box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"      src="https://jalammar.github.io/images/t/transformer_attention_heads_qkv.png"><br> <div style="color:orange; border-bottom: 1px solid #d9d9d9;     display: inline-block;     color: #999;     padding: 2px;">通过多头注意机制，我们为每个头维护单独的Q/K/V权重矩阵，从而得到不同的Q/K/V矩阵。正如上面提到的那样，我们将X乘以 WQ/WK/WV矩阵 来得到 Q/K/V 矩阵</div></center>

如果我们进行上面概述提到的相同的self-attention计算，在8个不同的时间使用8个不同的权重矩阵，最终将会得到8个不同的Z矩阵。

![](https://jalammar.github.io/images/t/transformer_attention_heads_z.png)

这就有点麻烦了。因为前馈神经网络层并不是期望8个矩阵，而是需要一个矩阵（每个单词一个向量）。所以我们需要将这8个矩阵整合成一个矩阵。

怎么办？我们将8个矩阵连接起来然后乘以一个单独的矩阵WO。

![](https://jalammar.github.io/images/t/transformer_attention_heads_weight_matrix_o.png)

这就是多头注意力的全部内容。你有没有发现这仅是一小部分的矩阵。我们把这些矩阵都放到一个图解中，更容易总揽全局：

![](https://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)

现在我们已经粗略了解了注意力头了，我们重新审视之前的例子，看看在例子中编码单词“it”的时候，不同的注意力头关注在哪里？

<center><img style="border-radius: 0.3125em;     box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"      src="https://jalammar.github.io/images/t/transformer_self-attention_visualization_2.png"><br> <div style="color:orange; border-bottom: 1px solid #d9d9d9;     display: inline-block;     color: #999;     padding: 2px;">当编码单词“it”时，一个注意力头关注在“the animal”上，而另一个关注在“tired”上。从某种意义上来说，模型对单词“it”的表达在“animal”和“tired”中均有所表现。</div></center>

然而当我们把所有注意力头都在图上画出来时，可能就有点难以理解了：

![](https://jalammar.github.io/images/t/transformer_self-attention_visualization_3.png)

## 使用位置编码表示序列顺序

到目前为止我们还未考虑输入序列中单词顺序的问题。

为解决这个问题，Transformer为每个输入的词嵌入增加了一个向量。这些向量遵循模型学习到的特定模式，这有助于确定每个单词的位置，或者学习到不同单词之间的距离。直觉告诉我们，将这些值添加到词嵌入之中可以在计算点积注意和将词嵌入映射到 Q/K/V 向量时提供有意义的距离信息。

<center><img style="border-radius: 0.3125em;     box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"      src="https://jalammar.github.io/images/t/transformer_positional_encoding_vectors.png"><br> <div style="color:orange; border-bottom: 1px solid #d9d9d9;     display: inline-block;     color: #999;     padding: 2px;">为了给模型增加单词顺序信息，我们增加了位置编码向量---其值遵循特定模式。</div></center>



如果假定词嵌入维度为4，那真实的位置编码如下：

<center><img style="border-radius: 0.3125em;     box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"      src="https://jalammar.github.io/images/t/transformer_positional_encoding_example.png"><br> <div style="color:orange; border-bottom: 1px solid #d9d9d9;     display: inline-block;     color: #999;     padding: 2px;">示例说明词嵌入维度为4时的位置编码</div></center>

这种模式究竟看起来如何呢？

下图中，每一行对应一个向量的位置编码。因此第一行就是输入序列中第一个单词的词嵌入向量。每行包含512个值—每个值介于-1到1之间。这里我们进行了涂色，使模式可见。

<center><img style="border-radius: 0.3125em;     box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"      src="https://jalammar.github.io/images/t/transformer_positional_encoding_large_example.png"><br> <div style="color:orange; border-bottom: 1px solid #d9d9d9;     display: inline-block;     color: #999;     padding: 2px;">该例子中共20个词（行），词嵌入向量维度为512维(列)。你可以看到中心区域分成两半。这是因为左边的值是由一个函数(正弦)产生的，右边的值是由另一个函数(余弦)产生的。然后将它们连接起来形成每个位置编码向量</div></center>

位置编码的公式在文章(3.5节)有描述。你可以在 [`get_timing_signal_1d()`](https://github.com/tensorflow/tensor2tensor/blob/23bd23b9830059fbc349381b70d9429b5c40a139/tensor2tensor/layers/common_attention.py) 函数中看到用于生成位置编码的代码。这并不是生成位置编码的唯一方式。然而，它的优点在于可以扩展到看不见的序列长度（eg. 如果要翻译的句子的长度远长于训练集中最长的句子）。

## 残差连接

需要注意一下：编码器架构中每个编码器中每个子层（self-attention, ffnn）都在其周围有残差连接，之后就是层标准化([layer-normalization](https://arxiv.org/abs/1607.06450))步骤。

![](https://jalammar.github.io/images/t/transformer_resideual_layer_norm.png)



如果将向量和self-attention层的标准化操作可视化，它会如下所示：

![](https://jalammar.github.io/images/t/transformer_resideual_layer_norm_2.png)

这也适用于解码器的子层。如果我们想看到堆叠两个编码器和解码器的Transformer，它将如下所示：

![](https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png)

## 解码器

我们已经介绍了编码器的的大部分概念，相信大家都知道解码器如何工作的。现在我们看一下它们是如何协同工作的。

编码器开始处理输入序列。然后将顶部编码器的输出变换为一组注意力向量K和V。这些将在每个解码器的“encoder-decoder attention” 层使用，这有助于解码器集中注意力在输入序列的合适位置：

<center><img style="border-radius: 0.3125em;     box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"      src="https://jalammar.github.io/images/t/transformer_decoding_1.gif"><br> <div style="color:orange; border-bottom: 1px solid #d9d9d9;     display: inline-block;     color: #999;     padding: 2px;">编码阶段结束后，开始解码阶段了。解码中每个时间步都输出一个来自输出序列的元素。(例子中是翻译后的英文句子)</div></center>

接下来的步骤会一直重复此过程，直到遇到结束符。下一时间步骤中，每个步骤的输出被发送到其底部解码器中，解码器就像编码器那样弹出他们的解码结果。就像对编码器输入所做的那样，我们对解码器输入中嵌入位置进行编码来指示每个词的位置。

![](https://jalammar.github.io/images/t/transformer_decoding_2.gif)

解码器中的self-attention层与编码器中的操作方式略有不同：

在解码器中，仅允许 self-attention层 关注输出序列中较早的位置。这是通过在计算self-attention中softmax步骤前屏蔽未来位置（将它们设置为-inf）实现的。

“Encoder-Decoder Attention” 层就像多头注意力(multiheader self-attention)一样工作———而“Encoder-Decoder Attention” 层从其下面的层创建其Queries矩阵，并从编码器堆栈的输出中获取Keys和Values矩阵。

## 最后的线性层和softmax层

解码器堆叠(decoder stack)输出浮点数向量。如何将其转换为一个单词？这就是最后Softmax层后线性层的工作了。

线性层是一个简单的全连接神经网络，它将解码器堆叠(decoder stack)产生的向量映射到一个更大更大的向量中去，这个向量称为logits向量。

假设模型有10000个单独的英文单词（模型的“输出词汇表”），这是从训练集中学到的。这使得logits向量有10,000个单元的宽度 ———每个单元对应一个唯一单词的得分。这样就解释了线性层后面的模型输出。

softmax层将这些分数转化为概率(全部为正数，加起来为1.0)。选择具有最高概率的单元，并将与其相关的单词作为本时间步的输出。

<center><img style="border-radius: 0.3125em;     box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"      src="https://jalammar.github.io/images/t/transformer_decoder_output_softmax.png"><br> <div style="color:orange; border-bottom: 1px solid #d9d9d9;     display: inline-block;     color: #999;     padding: 2px;">底部是解码器堆叠产生的向量。然后会变成输出的单词。</div></center>

## 温习一下训练过程

现在我们了解了训练好的Transformer模型的整个前向过程，能对整个训练过程有个直觉上的感知很有用。

训练期间，未经训练的模型将会遵循完全相同的前进方式行进。但是当我们在标记的训练集上训练的时候，我们可以将它的输出与真实的标签进行对比。

可视化理解一下，当我们假设输出词汇仅包含6个单词（"a", "am", "i", "thanks", "student", "<eos>"）。

<center><img style="border-radius: 0.3125em;     box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"      src="https://jalammar.github.io/images/t/vocabulary.png"><br> <div style="color:orange; border-bottom: 1px solid #d9d9d9;     display: inline-block;     color: #999;     padding: 2px;">训练之前，模型的输出词汇表已经在预处理阶段生成了。</div></center>

一旦定义了输出词汇表，就可以使用相同宽度(大小)的向量来表示词汇表中的每个单词了。这就是one-hot编码。例如，可以使用如下向量来表示单词“am”：

<center><img style="border-radius: 0.3125em;     box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"      src="https://jalammar.github.io/images/t/one-hot-vocabulary-example.png"><br> <div style="color:orange; border-bottom: 1px solid #d9d9d9;     display: inline-block;     color: #999;     padding: 2px;">输出词汇表的one-hot编码示例。</div></center>

介绍了训练过程，我们接着讨论模型的损失函数 ——— 这是模型训练中优化的指标，来告诉我们训练的模型有多么精确。

## 损失函数

假设我们正在训练模型，这是训练一个简单例子的第一步，比如将“merci”翻译为“thanks”。

我们如何理解这个翻译任务？这意味着我们希望输出一个指向“thanks”的概率分布。但是模型还未训练好，它输出极有可能是这个样子：

<center><img style="border-radius: 0.3125em;  box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"   src="https://jalammar.github.io/images/t/transformer_logits_output_and_label.png"><br> <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;"> 因为模型的参数(权重)都是随机初始化的，这个(未训练的)模型产生的对每个词的概率分布是随机值。这里我们可以比较一下和真实分布的差异。可以使用反向传播来更新模型的权重，使输出更接近真实输出。</div></center>

怎么来比较两个概率分布呢？可以简单地一个减去另一个。更多详细信息，就需要看一下 [交叉熵](https://colah.github.io/posts/2015-09-Visual-Information/) 和 [KL散度](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)。

注意，这是一个过度简化的例子。更现实一点的是，我们将使用更长一些的句子而不是单个单词。比如：输入：“je suis étudiant” ，期望输出：“I am a student”。这意味着我们希望模型能够输出一个如下的连续概率分布：

* 每个概率分布都被表达成宽度为 **vocab_size** 的向量。（在我们这个玩具模型中是6。现实一点的数字往往是3,000或10,000）
* 第一个概率分布在 与单词”I“相关联的单元处 有最高概率。
* 第二个概率分布在 与单词”am“相关联的单元处 有最高概率。
* 以此类推，直到第五个输出分布表示 ‘`<end of sentence>`’ ，这个符号也与10,000个元素的单词表中某个单元相关联。

<center><img style="border-radius: 0.3125em;  box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"   src="https://jalammar.github.io/images/t/output_target_probability_distributions.png"><br> <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">我们将在样本例子中训练产生这样目标概率分布的模型  </div></center>

在足够大的数据集中训练模型足够长的时间后，我们希望生成的概率分布如下所示：

<center><img style="border-radius: 0.3125em;  box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"   src="https://jalammar.github.io/images/t/output_trained_model_probability_distributions.png"><br> <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;"> 希望通过训练，模型会输出我们期望的正确翻译。不过这也并不能说明什么———如果这个短语是训练集的一部分的话（参考：[交叉验证](https://www.youtube.com/watch?v=TIgfjmp-4BA)）。注意，每个位置即便不是该时间步的输出，它也会获得一点的概率值——这就是softmax有用的地方。 </div></center>

现在因为模型一次产生一个输出，我们可以假设模型从该概率分布中选择具有最高概率的单词并丢弃其他可能的单词。这种方法称为贪婪解码。另一种方法是保持住该词的前两个候选（比如是“I”和“a”），在下一步解码中运行模型两次：一旦假设第一个输出位置是单词“I”，另一次假设输出位置是“me”，考虑#1和#2位置，保留错误较少的那个候选版本.... 这种方法称为“集束搜索(beam search)”，在这个例子中，beam_size是2（因为我们比较了两个位置#1，#2后给出的结果），top_beams也是2（因为我们保留了2个词）。这都是试验中可以尝试的超参数。

## 更深一步

我希望本文能够帮助你很好的理解Transformer。如果想更深理解的话，我建议：

* 阅读文章 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) ，和Transformer的官方博文： ([Transformer: A Novel Neural Network Architecture for Language Understanding](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)), 和 [Tensor2Tensor announcement](https://ai.googleblog.com/2017/06/accelerating-deep-learning-research.html)。

* 看 [Łukasz Kaiser’s 的讲解视频](https://www.youtube.com/watch?v=rBCqOTEfxvg)  深入模型细节。

* 打开 [Tensor2Tensor的Jupyter notebook](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb) 来详细了解。

* 探索 [Tensor2Tensor 代码仓库](https://github.com/tensorflow/tensor2tensor)。

以及相关工作：

+ [Depthwise Separable Convolutions for Neural Machine Translation](https://arxiv.org/abs/1706.03059)

+ [One Model To Learn Them All](https://arxiv.org/abs/1706.05137)

+ [Discrete Autoencoders for Sequence Models](https://arxiv.org/abs/1801.09797)

+ [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/abs/1801.10198)

+ [Image Transformer](https://arxiv.org/abs/1802.05751)

+ [Training Tips for the Transformer Model](https://arxiv.org/abs/1804.00247)

+ [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)

+ [Fast Decoding in Sequence Models using Discrete Latent Variables](https://arxiv.org/abs/1803.03382)

+ [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)

