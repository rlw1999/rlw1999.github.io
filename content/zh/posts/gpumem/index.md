---
date: '2025-05-01T13:43:41+08:00'
draft: true
title: 'LLM GPU Memory'
math: true
ShowReadingTime: false
ShowWordCount: false
---

## 基础情况 Baseline

在最基本的情况下，我们有一个小模型在单卡上训练，使用 32 位浮点数（fp32），模型参数量为 $\Psi$。运行梯度下降算法需要三个阶段：

* 前向过程：输入的 batch 前向经过网络计算，保存每一层的中间结果（激活值，activation），得到 loss
* 反向传播：从 loss 反向经过网络计算，使用激活值计算每个参数的梯度
* 参数更新：使用梯度（如果使用 Adam 等优化器还需要动量、方差等信息）更新每个网络参数

在上面的过程中，我们可以将显存消耗分为两个部分：

* 模型状态：只与网络本身有关，与输入数据无关
  * 网络参数：$4\Psi$（fp32 占 4 bytes）
  * 梯度：$4\Psi$
  * 优化器状态：动量 $4\Psi$，方差 $4\Psi$
  * 总计 $16\Psi$
* 激活值：与输入数据有关

对 GPT 这样的架构来说，我们可以计算模型状态和激活值两个部分的比值。一个 GPT 模型一般可以由下面的参数定义：

1. v - Vocabulary Size（词表大小）模型能识别的唯一token数量。
2. d - Embedding Dimension（嵌入维度）每个token的向量表示维度。
3. L - Number of Layers（层数）Transformer解码器的堆叠层数。
4. h - Number of Attention Heads（注意力头数）多头注意力机制中的头数。
5. f - Feed-Forward Hidden Dimension（前馈网络隐藏层维度）通常为 4*d，是每层前馈网络的中间维度。

分别计算各个部分的参数量：

- Self-attention：$W_Q, W_K, W_V, W_O$ 四个矩阵再加上bias，估计为$4d^2$
- Feed-forward：一个一维卷积先从 $d$ 升到 $4d$，再有一个一维卷积降到$d$，对应两个矩阵和bias，估计为$8d^2$
- LayerNorm：正比于$d$，忽略
- Embedding：$vd$，随着模型变大，层数变多也可以忽略

总参数量大致为 $12Ld^2$，其中 Attention 层占 $1/3$，FFN 层占 $2/3$。

输入数据大小可以用额外两个参数计算：

6. s - Context Length（上下文长度）模型处理的最大序列长度（token数）。
7. b - Batch size（批大小）。

在前向过程中：
- Self-attention 的中间结果有两种
  - $Q,K,V$，output，layernorm 这些正比于序列长度，大小为 $bsd$
  - $Q^TK$，softmax 这些正比于序列长度平方，大小为 $bhs^2$
  - 忽略 dropout，一共 $6bsd + 2bhs^2$
- Feed-forward：$10bsd$

总激活值个数大概为 $(16bsd+2bhs^2)L=bL(16sd+2hs^2)$，占用的显存与模型状态的比值为：

$$\frac{4bL(16sd+2hs^2)}{16\times 12Ld^2}=b\left( \frac{s}{3d}+\frac{hs^2}{24d^2} \right).$$

显然，激活值显存与 batch size 成正比。当上下文长度 $s$ 相比于模型维度 $d$ 比较小时，显存开销主要是模型状态；当上下文长度比较长或者 batch size 比较大时，激活值占显存的主要部分。

## 混合精度 Mix Precision

混合精度训练的基本思想是使用低精度的浮点数（fp16，bfp16）储存模型参数，以减少显存开销。但是为了保证训练时的精度损失和稳定性，优化器依然需要使用 fp32 来保持状态，并且需要额外储存一份 fp32 的模型参数副本用于更新。因此，使用 fp16 进行混合精度训练的模型状态显存开销为：

|  | fp32 | mixed fp16 |
| :-----: | :-----: |:-----: |
| 参数 | $4\Psi$ bytes | $2\Psi$ bytes |
| 梯度 | $4\Psi$ bytes | $2\Psi$ bytes（或 $4\Psi$ bytes） |
| 优化器状态（Adam） | $8\Psi$ bytes | $12\Psi$ bytes |
| 总记 | $16\Psi$ bytes | $16\Psi$ bytes（或 $18\Psi$ bytes） |

可以发现，使用混合精度并不会减小模型状态的显存开销，甚至会占用更多；其主要作用在于减少激活值的开销，可以直接将激活值开销减半。

## 分布式数据并行 DDP



## FSDP

