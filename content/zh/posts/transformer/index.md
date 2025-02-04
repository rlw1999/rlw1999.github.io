---
date: '2025-02-05T13:43:41+08:00'
draft: false
title: 'Transformer'
math: true
ShowReadingTime: false
ShowWordCount: false
---

[Transformer](https://arxiv.org/pdf/1706.03762) 的重要性已经无需多言，简洁性和高效性让其成为现代大模型的基石。然而阅读 Transformer 原文并不是一件简单的事。原文受限于篇幅对很多重要的细节并未过多解释，并且行文顺序也不太适合缺乏背景知识的读者。本文是我学习 Transformer 论文的笔记，综合了多篇文章、教程对原文进行补充。

## 注意力机制 Attention {#attention}

Transformer 论文标题 “Attention Is All You Need” 中的注意力机制（Attention）并不是 Transformer 首创的，[d2l](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html) 对其中的来龙去脉有比较详细的解释。Transformer 的核心贡献是向大家展示了注意力机制本身的重要性：仅仅使用注意力模块本身就能构建强大的神经网络。从理解顺序上来看，我们可以先来总结一下 [d2l](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html) 中的内容。

注意力机制可以总结为下面的公式：

$$ \text{Attention}(\mathbf{q}, \mathcal{D}) = \sum_{i=1}^{m} \underline{\frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{\sum_j \alpha(\mathbf{q}, \mathbf{k}_j)}} \mathbf{v}_i $$

其中 $\mathcal{D}$ 是数据集 $\\{(\mathbf{k}_1, \mathbf{v}_1),\cdots, (\mathbf{k}_m, \mathbf{v}_m)\\}$，对应键（Key，K）和值（Value，V），$\mathbf{q}$ 是问题（Query，Q）。每个 Q，K，V 都是一个向量。$\alpha(\mathbf{q}, \mathbf{k}_i)$ 是一个标量，表示 $\mathbf{q}$ 对 $\mathbf{k}_i$ 的注意（匹配）程度。公式中划线的部分是对 $\alpha(\mathbf{q}, \mathbf{k}_i)$ 进行归一化。这个公式描述的过程，就是我们使用问题 $\mathbf{q}$ 和数据集中的每个键 $\mathbf{k}_i$ 计算匹配度 $\alpha$，经过归一化得到权重之后对值 $\mathbf{v}_i$ 进行线性插值。[d2l](https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-pooling.html) 中进一步给出了一个数据拟合的任务来帮助注意力机制的理解：我们希望拟合一堆散点 $(x_i, y_i)$ 得到一条曲线，对于任意一个 $x$，越近的 $x_i$ 对应的 $\alpha(x,x_i)$ 就越大，表示 $x$ 越“注意” $x_i$，归一化之后再对 $y_i$ 进行插值，就得到了曲线上对应的 $y$ 值。


注意力机制乍一看跟自然语言处理没什么关系，我们可以参照 [3Blue1Brown](https://www.youtube.com/watch?v=eMlx5fFNoYc) 的视频来理解。在处理自然语言时，数据集就是当前处理的上下文（context），里面保存了网络对现有文本的理解。注意力机制的目的是让每个单词相互影响，不断让网络的理解更精准。比如“国王”是一个比较宽泛的概念，“英国的国王”则更为具体。那么网络在处理“国王”这个单词时，就可以生成一个问题 $\mathbf{q}$，对应的概念为“这个单词的前面有地区的修饰吗？”，然后将 $\mathbf{q}$ 与上下文中的 $\mathbf{k}_i$ 进行匹配。比如“英国”这一词的 Key 中可能就包含了“我是一个地名”的概念，这样通过权重计算，“国王”这个词就可以“注意”到“英国”这个词，并使用“英国”对应的 Value 更新“国王”这个词的涵义。之后当其他词再需要理解“国王”这个词时，其中就已经包含了“英国的国王”这一概念了。
{{< sidenote >}}
这个过程只是我们对注意力机制假想的理解，真实的网络可能根本不是这样工作的。
{{< /sidenote >}}

回到具体的公式上来，可以发现注意力机制的核心是$\alpha(\mathbf{q}, \mathbf{k}_i)$的具体形式。Transformer 文章的选择非常简单：

$$ \alpha(\mathbf{q}, \mathbf{k}_i) = \exp(\mathbf{q}^\top \mathbf{k}_i) $$

也就是直接计算内积，$\exp$ 对应着使用 softmax 进行归一化。假设 $\mathbf{q}$ 和 $\mathbf{k}_i$ 中每个元素都独立同分布，均值为 $0$， 方差为 $1$，那么 $\mathbf{q}^\top\mathbf{k}_i$ 的方差为 $\text{Dim}(q)=\text{Dim}(\mathbf{k}_i)=d$。为了归一化内积的方差，最终选择的形式为：

$$ \alpha(\mathbf{q}, \mathbf{k}_i) = \exp(\mathbf{q}^\top \mathbf{k}_i/\sqrt{d}) $$

## 自回归 Autoregression

有了对注意力机制的基本了解之后，我们就可以深入了解 Transformer 的细节。原始的 Transformer 不是现在的通用的大模型，而是专门解决序列到序列的问题，比如翻译任务：给定一个 A 语言的源序列，网络输出翻译到 B 语言的序列结果。这里就引出了 Transformer 网络最关键的框架图：

{{< figure
  src="transformer.png"
  width=60%
  align=center
  caption="Transformer 的框架图，来自[原论文](https://arxiv.org/pdf/1706.03762)"
>}}

左边的部分是编码器（Encoder），右边的部分是解码器（Decoder）。忽略网络结构的细节，具体生成的过程由下面的伪代码表示：

```python
# 代码由 deepseek-R1 生成
def generate(input_sequence, max_length):
  # 编码器处理输入
  encoder_output = encoder(input_sequence)
  # 初始化解码器输入（起始符）
  decoder_input = ["<s>"]
  for t in range(max_length):
    # 解码器处理当前输入（掩码自注意力 + 交叉注意力）
    decoder_output = decoder(decoder_input, encoder_output)
    # 预测下一个词（贪心选择）
    next_token = argmax(decoder_output[-1])
    # 终止检查
    if next_token == "</s>":
      break
    # 更新解码器输入
    decoder_input.append(next_token)
  return decoder_input[1:]  # 去除起始符
```

编码器只运行一次，输入源序列之后保存输出，称为记忆（Memory）。解码过程是从开始符 `<s>` 逐语素（Token）生成，直到输出结束符 `</s>` 或者达到最大输出长度。Token 可理解为文本的基本单元，可能是单词或子词。在解码器要输出第 $t$ 个 Token 时，解码器会将之前的生成的 $t-1$ 个 Token 和编码器输出的记忆作为输入，然后输出第 $t$ 个 Token 的概率分布。也就是说，假如一共有 $N$ 个 Token，解码器的输出就是一个 $N$ 维的归一化的向量，然后我们从其中选择概率最大的 Token 作为输出。这一过程被称为自回归（autoregression），因为我们不断用旧的输出作为新的输入。

## 词嵌入 Word Embedding

下面我们就逐模块分析 Transformer 的具体结构。输入的句子被离散为 Token 的序列。每个 Token 就是一个整数，标记其在词表（Vocabulary）中的序号，因此输入的网络的是一个整数类型的大小为 `(n_batch, len_seq)` 的 Tensor。所有的句子会对齐到长度 `len_seq`，不够的会在末尾填充词表中预留的`<pad>` Token，太长的会切片或者丢弃。

每个 Token 可以理解为一个 one-hot 向量，维度为数据集中的所有 Token 个数 $n_{\text{vocab}}$（比如 Transformer 原文中使用的英语-德语数据集大概有 37000 个公共 Token）。在输入注意力模块之前，每个 Token 都会通过词嵌入 （Word Embedding）从 one-hot 向量转换成一个 $d_{\text{model}}$ 维的向量（原文中 $d_{\text{model}}=512$）。我们可以定义一个大小为 `(d_model, n_vocab)` 的矩阵 $\mathbf{W}_E$，矩阵的每一列对应一个 Token 的向量。词嵌入的过程就是将 $\mathbf{W}_E$ 乘以 Token 的 one-hot 向量得到对应的嵌入向量。

> `<pad>` 不应该对前面文本的理解造成影响，因此规定对应的嵌入向量为零向量，并在后面的网络中始终将其输出 mask 掉。

将离散的 Token 转化为连续向量的思想源于早期的工作，比如[Word2Vec](https://arxiv.org/pdf/1301.3781)。这其中的核心观察是 Token 所表示的语义可以被近似映射到高维空间中的向量，是可以被计算的。比如“巴黎”和“法国”之间的关系与“罗马”和“意大利”之间的关系是类似的，那么在语义空间中，我们就应该有“巴黎”-“法国”+“意大利”=“罗马”。我们期待的是，通过网络学习不断更新 $\mathbf{W}_E$，嵌入层可以自发学到这样的语义关系，形成一个语义空间。

在整个网络中，我们保持每个语义向量的维度 $d_{\text{model}}$，不会升高也不会降低，相当于只会在语义空间中移动这些向量。而在解码器的最后我们需要输出概率，就需要将向量重新变回 Token。Transformer 中的做法是使用嵌入矩阵的转置 $\mathbf{W}_E^\top$ 乘以向量（称为 “Unembedding”），然后经过 Softmax 层归一化得到概率。这样的做法称为 [Weight Tying](https://arxiv.org/pdf/1608.05859)，也就是输入的 Embedding 层和输出的 Unembedding 层使用的是相同的权重。这样不仅可以降低参数量，还有助于模型更好地对齐输入和输出的语义表示。然而 Weight Tying 也不是所有时候都是一个好的选择，比如输入和输出词汇空间差别比较大的情况。

在 Transformer 的框架图中，经过 Embedding 之后还会加上位置编码（Positional Encoding）向量。这是因为网络接受的是所有 Token 的并行输入，并不包含 Token 的远近关系，因此需要一个机制告诉网络 Token 的位置关系。Transformer 中使用的是基于 $\sin$ 和 $\cos$ 的位置编码，对于位置为 $pos$ 的 Token，其位置编码 $PE \in \mathbb{R}^{d_{\text{model}}}$为：

$$ PE_{2i} = \sin(\frac{pos}{10000^{2i/d_{\text{model}}}}) $$
$$ PE_{2i+1} = \cos(\frac{pos}{10000^{2i/d_{\text{model}}}}) $$

位置编码也是一个长度为 $d_{\text{model}}$ 的向量，因此可以直接加到嵌入向量上。经过 Embedding 操作之后的 Tensor 形状为 `(n_batch, len_seq, d_model)`。

## 多头注意力模块 Multi-Head Attention

{{< figure
  src="attention.png"
  width=90%
  align=center
  caption="Attention 框架图，来自[原论文](https://arxiv.org/pdf/1706.03762)"
>}}

在[第一节]({{< ref "#attention" >}})中我们大概了解了注意力机制是如何工作的，现在可以来看模型中具体是如何实现的。为了简单考虑，我们从单头注意力（Single-Head Attention）开始。Attention 模块的输入维度为 `(n_batch, len_seq, d_model)`，我们首先要将其转化为 Q，K，V 三个 Tensor，这通过三个矩阵乘法实现。Q 和 K 的维度为 `(n_batch, len_seq, d_k)`，对应的矩阵 $\mathbf{W}_Q$ 和 $\mathbf{W}_K$ 维度为 `(d_k, d_model)`； V 的维度为 `(n_batch, len_seq, d_v)`，对应的矩阵 $\mathbf{W}_V$ 维度为 `(d_v, d_model)`。

> 原文中 $d_k=d_v=d_{\text{model}}=512$。

然后这三个 Tensor 会被代入下面的公式中：

$$\text{Attention}(Q, K, V)=\text{softmax}(\frac{QK^T}{\sqrt{d_k} })V$$

这个公式可以对应到下面的代码：

```python
"""
输入
X: (n_batch, len_seq, d_model)
mask: (n_batch, len_seq, len_seq)

输出
V: (n_batch, len_seq, d_v)
"""
Q = X @ W_Q.T
# (n_batch, len_seq, d_k)
K = X @ W_K.T
# (n_batch, len_seq, d_k)
V = X @ W_V.T
# (n_batch, len_seq, d_v)
# 实际实现中 W_Q，W_K，W_V 是网络线性层

score = (Q @ K.transpose(-2, -1)) / torch.sqrt(d_k) 
# (n_batch, len_seq, len_seq)
# score[b] 是一个 (len_seq, len_seq) 大小的矩阵，表示 batch 中第 b 个句子内部的点积相似度
# score[b] 的每行对应一个 Query，每列对应一个 Key
score = score.masked_fill(mask, -1e9)
# 将被 mask 的 score 置为 -1e9，这样经过 softmax 之后权重就近似为 0
score = torch.softmax(score, dim=-1)
# (n_batch, len_seq, len_seq)
# 将相似度矩阵 score[b] 每行归一化为权重
V = score @ V
# (n_batch, len_seq, d_v)
# 使用权重对 Value 进行加权，结果 V[b] 的每一行对应一个 Query 得到的答案
```

代码中还多了一步掩码（Mask）操作，将被 Mask 的权重置为 $0$。我们前面提到每个序列的长度是不一定的，后面需要填充 `<pad>`，那么其对应位置的权重就应该被 Mask 掉。在[后面]({{< ref "#causality" >}})我们还会介绍 Mask 的其他作用。

Transformer 在单头注意力机制上进一步提出了多头注意力机制（Multi-Head Attention），这个思想类似于在 CNN 中进行多通道的卷积操作。我们不是将每个输入向量映射为一个 $Q$，$K$，$V$，而是映射到多个 $Q_i$，$K_i$，$V_i$。具体来说，我们定义一共 $h$ 个头，头内正常计算 Attention，输出 Value 维度为 `(n_batch, len_seq, d_v)`；头之间并行计算，最后将输出拼到一起，得到 Value Tensor 维度为 `(n_batch, len_seq, h * d_v)`。最后，我们再使用一个输出矩阵 $\mathbf{W}_O$ 将拼接的 Value 重新映射为 `(n_batch, len_seq, d_model)` 大小的 Tensor。对应的代码为：

```python
"""
输入
X: (n_batch, len_seq, d_model)
mask: (n_batch, h, len_seq, len_seq)

输出
V: (n_batch, len_seq, d_model)
"""

Q = (X @ W_Q.T).view(n_batch, len_seq, h, d_k).transpose(1, 2)
# W_Q: (h * d_k, d_model)
# X @ W_Q.T: (n_batch, len_seq, h * d_k)
# Q: (n_batch, h, len_seq, d_k)
K = (X @ W_K.T).view(n_batch, len_seq, h, d_k).transpose(1, 2)
# W_K: (h * d_k, d_model)
# X @ W_K.T: (n_batch, len_seq, h * d_k)
# K: (n_batch, h, len_seq, d_k)
V = (X @ W_V.T).view(n_batch, len_seq, h, d_k).transpose(1, 2)
# W_V: (h * d_v, d_model)
# X @ W_V.T: (n_batch, len_seq, h * d_v)
# V: (n_batch, h, len_seq, d_v)
# 实际实现中 W_Q，W_K，W_V 是网络线性层

score = (Q @ K.transpose(-2, -1)) / torch.sqrt(d_k) 
# (n_batch, h, len_seq, len_seq)
score = score.masked_fill(mask, -1e9)
# (n_batch, h, len_seq, len_seq)
score = torch.softmax(score, dim=-1) 
# (n_batch, h, len_seq, len_seq)
V = score @ V
# (n_batch, h, len_seq, d_v)

V = V.transpose(1, 2).contiguous().view(n_batch, len_seq, h * d_v)
# V.transpose(1, 2): (n_batch, len_seq, h, d_v)
# contiguous() 重排 Tensor 使其在内存中连续
# V: (n_batch, len_seq, h * d_v)
V = V @ W_O.T
# W_O: (d_model, h * d_v)
# V: (n_batch, len_seq, d_model)
# 实际实现中 W_O 是网络线性层
```

原文使用了 $h=8$ 个头，并且 $d_k=d_v=d_{\text{model}}/h=64$，我们可以在这个设置下对比单头注意力和多头注意力的储存和计算开销。在参数量上，多头注意力模块比单头多需要一个线性层 $W_O$，大小为 `(d_model, h * d_v)`，其他线性层与单头是一样的。在计算开销上，我们可以衡量二者的浮点数计算次数（floating point operations, FLOPs）。参考[知乎文章](https://zhuanlan.zhihu.com/p/648988727)，[Mambaout Tutorial](https://github.com/yuweihao/misc/blob/master/MambaOut/mambaout_eq6_tutorial.pdf)，我们可以得到多头注意力机制的 FLOPs 估计为 `4 * n_batch * d_model * len_seq * (2 * d_model + len_seq)`，可以发现这个结果与头的数量 $h$ 没有关系。但是更多的头隐含了更大的并行性，因此我们可以推断理想情况下多头注意力机制在计算开销上与单头注意力是接近的。实验表明，在差不多参数和计算开销的情况下，多头注意力的结果要优于单头注意力，因此后面大家都默认用多头注意力模块了。

## 残差与归一化 Residual & Normalization

注意力模块输出的 `(n_batch, len_seq, d_model)` 维的 Value 张量会通过残差连接（Residual Connection）和层归一化（Layer Normalization）之后再输入给后面的网络：$x \gets \text{LayerNorm}(x + \text{Attention}(x))$。残差连接的思想源于 [ResNet](https://arxiv.org/abs/1512.03385)，可以避免梯度消失/爆炸，简化优化，是深层网络的标准做法。在 Attention 层选择 $h \cdot d_v = d_{\text{model}}$ 也是出于要使用残差连接的考量。

Transformer 使用的归一化方法是层归一化，具体来说是在 `(n_batch, len_seq, d_model)` 的最后一维做归一化：
```python
"""
X: (n_batch, len_seq, d_model)
"""
mean = X.mean(-1, keepdim=True)
var = X.var(-1, unbiased=False, keepdim=True)
out = (x - mean) / torch.sqrt(var + 1e-6)
# 最后一维的均值为 0，方差为 1
out = gamma * out + beta
# gamma 和 beta 都是 (d_model) 维的可学习的向量 
```
使用层归一化的想法是比较自然的，我们相当于对每个 Token 对应的语义向量做了归一化，跟序列长度和 batch 大小都没有关系。并且这里有一个涉及到高维高斯分布的观察可以帮助我们理解。

如果向量 $\mathbf{x}\in \mathbb{R}^d$ 的每个分量都满足 $\mathcal{N}(0, 1)$ 的高斯分布，那么其平方模长 $\lVert \mathbf{x} \rVert^2$ 满足[卡方分布](https://en.wikipedia.org/wiki/Chi-squared_distribution) $\chi_d^2$，概率密度公式为：

$$f(x)=\frac{1}{2^{\frac{d}{2}}\Gamma(\frac{d}{2})} x^{\frac{d}{2}-1}e^{-\frac{x}{2}}$$

根据概率密度函数的变量替换关系，我们可以推出向量长度 $r=\lVert \mathbf{x} \rVert$ 对应的概率密度函数为 $2f(r^2)r$，函数图像如下所示：

{{< figure
  src="chi.jpg"
  width=80%
  align=center
  caption="向量长度 $r=\lVert \mathbf{x} \rVert$ 在不同维度 $d$ 下的概率分布: $2f(r^2)r$，高维时形状接近 $\mathcal{N}(\sqrt{d}, \frac{1}{2})$"
>}}

从图中可以发现，在高维情况下，长度 $r$ 的分布会趋向于均值为 $\sqrt{d}$，方差为 $\frac{1}{2}$ 的正态分布 $\mathcal{N}(\sqrt{d}, \frac{1}{2})$。这个结论的严格证明可以使用中心极限定理，这里就不展开了。由于方差相对于均值可以基本忽略不计，因此我们可以说归一化之后的向量 $\mathbf{x}$ 主要集中在半径为 $\sqrt{d}$ 的薄球壳内。
{{<sidenote>}}
这一反常的结论是[维数灾难](http://en.wikipedia.org/wiki/Curse_of_dimensionality)（Curse of Dimensionality）的一种体现。
{{</sidenote>}}
对于 Transformer 而言，不断进行这样的归一化意味着我们更希望向量通过方向编码语义信息，而非其模长。更重要的是，Attention 层的点积注意力机制对于长度相近的向量更为合理，这样我们就是在直接测量二者之间的夹角。

另外一种常用的归一化方法是批归一化（Batch Normalization），也就是对 `(n_batch, len_seq, d_model)` 的第一维进行初始化。批归一化的思想是对每一个特征在批内进行归一化，每个特征之间保持同等地位。对于序列而言，由于序列长度不固定，无效的 padding 可能干扰归一化的进行；并且批归一化导致了一批不同序列的独立 Token 相互影响，对 Attention 层造成干扰。因此 Transformer 并没有使用批归一化。也有[文章](https://arxiv.org/abs/2210.05153)对批归一化与 Transformer 做了更详细的分析。

## 前馈网络 Feed Forward Networks

尽管文章标题叫 “Attention Is All You Need”，但实际上除了 Attention 层之外，Transformer 网络还包含了前馈网络（Feed Forward Networks，FFN）。前馈层是两层线性层中间加上 ReLU 的激活函数。与一般的两层 MLP 不同，FFN 称为“基于位置”（Position-wise），核心在于 FFN 对每个 Token 独立进行，Token 之间相互不影响，都进行相同的变化，也就是只在 `(n_batch, len_seq, d_model)` 的最后一维上做矩阵乘法。Transformer 原文并没有过多解释 FFN 为什么要这么设计，不过一个显然的好处是 FFN 的作用和 Attention 的作用被明显分开，不会造成参数冗余：FFN 对让每个 Token 独立进行非线性变化，增强网络的表达能力；Attention 层让不同 Token 相互影响。不过在其他领域（如计算机视觉）中，根据数据特性调整FFN结构可能带来性能提升。另外，FFN 输出的 Tensor 也要进行上面提到的残差连接和层归一化。

将 Attention 和 FFN 交替堆叠 $N$ 次就实现了编码器，输出的向量维度为 `(n_batch, len_seq, d_model)`。 解码器的结构与编码器基本相同，除了两个细节上面没有提到，一是因果（Causality），二是交叉注意力（Cross Attention）。

## 因果 Causality {#causality}

我们上面解释了 Transformer 生成结果的方式是自回归，每次根据输入序列和输出序列的前几个 Token 生成下一个 Token，解码器最多需要运行 `len_seq` 次才能拿到完整的输出，并且这 `len_seq` 必须串行进行。在训练时，如果我们运行 `len_seq` 次才能计算一次 loss 并反传，反传的频率太低了，这相当于把网络加深了 `len_seq` 倍。因此训练时我们实际是将数据集里的目标序列作为输入直接喂给解码器，让解码器运行一次计算序列每个 Token 的概率。显然，我们不能让网络根据答案预测答案，网络预测的 Token 必须只能基于这个位置之前的信息，不能依赖当前位置以及之后输入的 Token，也就是必须保证因果（Causality）。Transformer 提到了两个机制来保证因果：输入右移和掩码。
{{<sidenote>}}
事实上，position-wise FFN 也保证了后面的 Token 不会影响前面的 Token
{{</sidenote>}}

解码器输出的是一个 `(n_batch, len_seq, d_model)` 的 Tensor，经过 Unembedding 层和 Softmax 之后输出的是形状为 `(n_batch, len_seq, n_vocab)` 的概率，每个 Token 位置处的 `(n_vocab)` 维的向量对应的应该是下一个 Token 的概率。因此在将目标序列输入给解码器时，我们要将其向右移动一个 Token，在开头补上 `<s>` 表示开始，这样输出 Tensor 的第一个位置表示的才是目标序列的第一个 Token 的概率。这个操作对应 Transformer 框架图中的 “shifted right”。

在 Attention 层中，当前 Token 不能受到后面 Token 的影响对应着当前 Token 的 Query 不能匹配上后面 Token 的 Key，因此我们可以将 Attention 层里的掩码设计成上三角矩阵：

```python
mask = torch.triu(torch.ones(len_seq, len_seq), diagonal=1)
"""
[[1, 1, 1],
 [0, 1, 1],
 [0, 0, 1]]
"""
```

Mask 的对角线应该是 $1$，因为当前 Token 的输出是下个 Token 的概率，可以与当前 Token 相关。

## 交叉注意力 Cross Attention

解码器在进行和编码器一样的自注意力（Self Attention）层之后，还会额外经过一个交叉注意力（Cross Attention）层，表示解码器从源序列中获取信息。交叉注意力层的结构其实与自注意力层完全一样，只是 $K$ 和 $V$ 是从解码器得到。解码器输出的 `(n_batch, len_seq, d_model)` 维的特征张量，经过解码器内部的两个线性层 $\mathbf{W}_K$ 和 $\mathbf{W}_V$ 转化为 $K$ 和 $V$，而 $Q$ 依然从解码器上个自注意力模块得到。

## 为什么选择注意力？ Why Attention？

到这里我们已经解释完了 Transformer 的框架以及训练和生成过程，除了一些细节问题没有提到，比如 Dropout，Lable Smoothing 等，在实现时可以参考论文[原文](https://arxiv.org/pdf/1706.03762)。更重要的问题是，为什么选择注意力机制？注意力机制相比于之前的框架优势到底在什么地方？Transformer 论文花了一章来比较 Attention 与循环网络（Recurrent Neural Network，RNN）和卷积神经网络（Convolution Neural Network，CNN），总结出了三点：

{{< figure
  src="table.png"
  width=80%
  align=center
  caption="Attention 与 RNN，CNN 的比较"
>}}

**计算复杂度** Attention 层主要的计算开销是 `Q @ K.transpose(-2, -1)` 和 `score @ V` 这两个矩阵乘法，对每个序列计算复杂度为 $O(n^2d)$（$n$ 为序列长度，$d$ 为向量维度）。但是这里没有包含计算 $Q$，$K$，$V$ 的开销，这部分是 $O(nd^2)$。Recurrent 层需要串行 $n$ 步更新每个 Token，复杂度为 $O(nd^2)$。Convolution 层每个 Token 需要 $O(kd^2)$ 次操作（假设卷积核大小为 $k$），那么总的复杂度就为 $O(knd^2)$。假设序列长度 $n$ 小于 $d$，那么 Attention 在计算上就是有优势的，否则要差于 RNN。对于长序列，原文提出将 Attention 的范围缩减到常数 $r$，有点类似于卷积机制，但是这样肯定会限制模型理解文本的能力。提高 Attention 处理长文本的能力是后来很多研究的目标，这里先按下不表。

**并行度** 我们在[之前]({{<ref "#causality">}})说明了 Transformer 可以进行并行理解和推理，这对于 CNN 也是类似的。然而 RNN 必须按顺序遍历整个文本，因此并行度比较低。

**长程影响的距离** 长程距离影响保证了模型能够理解更深层的逻辑关系，因此我们需要让网络中任意两个 Token 之间相互影响的距离尽可能近。对于 Attention 而言，一层就能处理所有 Token 之间的关系，因此距离为 $O(1)$。对于 RNN，序列的头需要等待网络运行 $n$ 步才能影响到末尾，因此最长距离为 $O(n)$。对于 CNN，网络只能一次处理 $k$ 个邻居，因此一般为 $O(n/k)$。如果使用了扩张卷积（dilated convolution），每层可将感受野扩大 $k$ 倍，因此表格里是 $O(\log_k(n))$。

通过这三点我们就大概明白了作者为什么觉得 Attention 是个更好的架构，核心在于我们希望模型有非常高的并行度，并且让不同位置的文本可以以最短的路径相互影响。实验也支撑了作者的判断。同时，作者还提到了 Transformer 具有更好的可解释性。这也是一个非常有意思的问题，深入理解 Transformer 的原理可以帮助我们设计更好的网络结构。比如，Anthropic 的[工作](https://transformer-circuits.pub/2023/monosemantic-features/index.html)深入研究了 Attention 内部各个特征的作用；也有研究者从[数学动力学](https://arxiv.org/abs/2312.10794)的角度解释 Transformer 的原理，将 Transformer 理解为高维空间中的粒子系统。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- [李沐：Transformer论文逐段精读](https://www.bilibili.com/video/BV1pu411o7BE/)
- [deepseek-R1](https://chat.deepseek.com)
- [3Blue1Brown](https://www.youtube.com/@3blue1brown)
- [Dive into Deep Learning](https://d2l.ai/)
- [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer)
- [Hyunwoongko's Transformer Implementation](https://github.com/hyunwoongko/transformer)
- [Transformer Implementation Tutorial](https://huggingface.co/datasets/bird-of-paradise/transformer-from-scratch-tutorial/blob/main/Transformer_Implementation_Tutorial.ipynb)