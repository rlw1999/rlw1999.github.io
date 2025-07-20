---
date: '2025-07-17T13:43:41+08:00'
draft: true
title: 'What Intelligence'
math: true
ShowReadingTime: false
ShowWordCount: false
---

妄图想一篇文章梳理下这波人工智能的主线和现在的困难。

## 语言模型 Language Model

1. Transformer：创新的网络框架，用 Attention + FFN 的模块搭建了 Decoder-Encoder 的框架，在翻译任务中大幅超越 RNN 和 CNN。
2. 预训练（BERT，GPT）：使用 Decoder-Only 的框架，在互联网语料上做预训练，使其经过微调之后可以超越下游任务的专有模型。
3. 扩展定律（GPT3）：通过实验说明了扩展模型参数、数据量、计算资源带来的收益是可预测的，促成了 GPT-3 的诞生，可以直接通过少量示例完成下游任务。
4. 对齐（SFT，RLHF）：通过监督微调（SFT）和基于人类反馈的强化学习（RLHF）可以让模型学会遵循指令，增加生成内容的安全性。
5. 推理（CoT，GPT o1，Deepseek R1）：通过在算术、符号推理等任务上的强化学习激发模型生成长思维链，成功泛化到创意写作等领域。
6. Agent（ReAct，MCP）：让模型可以调用外部工具获得额外信息。

我认同的一些观点：

* GPT构建了一个语言的概率树，实现了对互联网的压缩，输入 prompt 得到回答的过程，就是利用概率树对 prompt 进行解压的过程。
  * [GPT as a Monte Carlo Language Tree: A Probabilistic Perspective](https://arxiv.org/abs/2501.07641)
  * [压缩即智能：为什么 ChatGPT 拥有智能？](https://www.zhihu.com/tardis/zm/art/634194171?source_id=1005)
* 模型就是在学习统计规律，没有内在逻辑。通过例子的学习不能让模型学会真正的逻辑，模型只能进行组合泛化。思维链是概率模型基础上的表象：因为数据集中有思维链的数据，有文字描述的运算规则，模型才能遵传这些 pattern。
  * [Tracing the thoughts of a large language model](https://www.anthropic.com/research/tracing-thoughts-language-model)
  * [Case-Based or Rule-Based: How Do Transformers Do the Math?](https://arxiv.org/abs/2402.17709)
* RL 的作用就是把 pass-K 的能力提升到了 pass-1。在长序列推理中，关键的决策点数量很少，RL 可以让模型可以在这些决策点上反复尝试，提高成功率。稀疏的奖励可以 work 的原因在于搜索的空间其实没有那么大，这是基础模型的能力。
  * [张小珺Jùn｜商业访谈录: 和张祥雨聊，多模态研究的挣扎史和未来两年的2个“GPT-4时刻”](https://www.xiaoyuzhoufm.com/episode/683d2ceb38dcc57c641a7d0f)
  * [Learning to play Minecraft with Video PreTraining](https://openai.com/index/vpt/)
* 在数学任务上训练 RL 的能力可以泛化其他领域，原因是思维链的输出模式对于大部分任务都是有效的，不管是数学，编程，还是文学。

memory，long context length，auto benchmark

## 多模态模型 MultiModal Model

多模态模型的核心目标是将视觉、听觉这些模态与语言统一起来，让模型可以统一理解、生成不同模态的内容。我们主要关注视觉内容。
模型需要将语言与视觉内容联系起来。
一个直接想法是把 LLM 成功的经验复制过来，然而这里面有本质问题。语言模型理解、生成、人类对齐在同一个空间，可以对应到人类思考的空间，但是视觉任务不是。

我们如何相信 LLM 理解了我们的意图，在于 LLM 能够给出正确答案。


多模态的cot应该长成什么样子：step by step的生成，先画什么再画什么，可以擦除修改，在做几何题上可以了，但是不能泛化
如果在现有的模型上做，就是先生成一堆token，然后改变一些token
最终应该是vla的形式

问题是什么
方法是什么

* ImageNet：人工标注一些图片，
* 

## 具身智能 Embodied AI



**以下部分是草稿**

生成模型是另外一条线：VAE->GAN->diffusion

智能不只有说话，还需要有视觉

对比学习：难以 scale
vlm

智能不只有视觉，还要能影响物理世界






理解生成统一

scaling走到头了


vlm在压缩什么
genai是智能吗：不是，本质是压缩已有数据





## 技术路线

## Reference

- [Case-Based or Rule-Based: How Do Transformers Do the Math?](https://arxiv.org/abs/2402.17709)
- [Do Machine Learning Models Memorize or Generalize?](https://pair.withgoogle.com/explorables/grokking/)
- [Jeff Clune's talk in ICLR 2025](https://iclr.cc/virtual/2025/10000096)
- [Ilya Sutskever: Sequence to Sequence Learning with Neural Networks at NeurIPS 2024](https://www.youtube.com/watch?v=WQQdd6qGxNs)
- [OpenAI 研究](https://openai.com/zh-Hans-CN/research/index/)
- [许华哲: 具身智能需要从ImageNet做起吗？](https://zhuanlan.zhihu.com/p/1906157729292219201)
- [张小珺Jùn｜商业访谈录: 和张祥雨聊，多模态研究的挣扎史和未来两年的2个“GPT-4时刻”](https://www.xiaoyuzhoufm.com/episode/683d2ceb38dcc57c641a7d0f)
- [张小珺Jùn｜商业访谈录: 和王鹤聊，具身智能的学术边缘史和资本轰炸后的人为乱象](https://www.xiaoyuzhoufm.com/episode/6857f2174abe6e29cb65d76e)
- [Shunyu Yao: The Second Half](https://ysymyth.github.io/The-Second-Half/)
- [孙浩：通往AGI的四层阶梯](https://zhuanlan.zhihu.com/p/1896382036689810197)