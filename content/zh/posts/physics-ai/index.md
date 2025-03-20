---
date: '2025-02-17T13:43:41+08:00'
draft: false
title: 'Physics for Large Video Model'
math: true
ShowReadingTime: false
ShowWordCount: false
---

尽管现在视频生成模型已经能产生相当不错的效果，但是我们还是能经常发现生成结果中的不自然现象，也就是违反了大家对于几何、物理的常识。近年来出现了很多尝试将物理规则与视频模型相结合的工作，本文大致总结了其中三个方向。

## 物理对齐 Physics Alignment

在语言模型中，对齐（Alignment）指的是通过一系列算法和工程手段，修正模型的行为，使其输出符合预设的安全边界和人类意图。与之对应的，修正视频模型的输出使其满足物理规则的过程就是物理对齐。作为对齐领域的代表性工作，[InstructGPT](https://arxiv.org/abs/2203.02155) 提出了两种对齐方法：
  
**监督微调**（Supervised Fine Tuning，**SFT**）人工标注高质量的提示（prompt）和回答（output）数据集，通过监督学习的方式微调模型。
  
**基于人类反馈的强化学习**（Reinforcement Learning from Human Feedback，**RLHF**）对同一个提示，模型输出多个回答，人工对这些回答进行比较打分；使用打分的结果训练一个反馈模型（Reward Model），用于评价模型输出的好坏；使用反馈模型对模型进行强化学习，比如近端策略优化（Proximal Policy Gradient，PPO）。

将这两种想法用于视频模型中是比较直观的。对于 SFT，我们就需要使用物理真实的视频作为输入。但是真实世界中的视频当然都是物理真实的，可能的问题是视频动态不够，导致模型没有接受到足够的动态信息。因此像 [Cosmos](https://github.com/NVIDIA/Cosmos) 在预训练阶段就会保证数据能够反映真实物理规则。这主要是通过两点做到的：

- 收集包含大量动态的视频：包括驾驶视频、手部动作、第一人称视角、模拟结果等等。
- 对数据进行过滤：剔除质量低的、缺乏动态的、非物理等的视频，并提取一部分高质量视频作为后训练数据集。

尽管 Cosmos [论文](https://research.nvidia.com/publication/2025-01_cosmos-world-foundation-model-platform-physical-ai)在 5.3.2 节专门讨论了物理对齐的问题，但是实际上并没有做更多的尝试，只是在几个场景中测试了 Cosmos 生成的结果是否吻合模拟/真实的物理。

对于 RLHF 而言，首要问题是需要一个反馈模型来判别模型输出的结果是否满足物理规律。这方向一个代表工作是  [VideoPhy](https://videophy.github.io)。这篇工作的核心是对市面上十二个视频模型的生成结果进行人工打分，然后训练一个打分网络 VideoCon-Physics。打分分为两个维度，每个维度得分只有 0 或 1：一个是语义的符合程度（Semantic Adherence，SA），一个是是否符合物理常识（Physical Commonsense，PC），结果如下：

{{< figure
  src="leaderboard.png"
  width=90%
  align=center
  caption="Benchmark 结果"
>}}

可以发现成绩最好的是开源模型 [CogVideoX-5B](https://github.com/THUDM/CogVideo)，但是也只是勉强及格的水平。这方向类似的工作还有 [VideoScore](https://tiger-ai-lab.github.io/VideoScore/)，[PhyGenBench](https://phygenbench2024.github.io)。理论上有了打分模型之后我们就可以对视频模型进行强化学习对齐了，[OnlineVPO](https://onlinevpo.github.io) 就使用了 VideoScore 作为反馈模型微调了 [OpenSora](https://github.com/hpcaitech/Open-Sora) 模型，使其在 VideoScore 得分上超越了其他模型。

最近这个方向又有两篇新文章：[Physics IQ Benchmark](https://physics-iq.github.io/) 和 [WorldModelBench](https://arxiv.org/abs/2502.20694)。Physics-IQ 根据几个生活中的物理场景：固体、流体、光学、热物理、磁现象，测试大模型预测物理过程的能力，最后得到一个 0-100 的 Physics-IQ。结论上来说，所有大模型得分都不高，最高是 [VideoPoet](https://sites.research.google/videopoet/) （和 Physics-IQ 都是 Google 的） 的 29.5 分。并且作者观察到生成真实的视频 （更高的 MLLM 分数）和物理正确（更高的 Physics-IQ）并没有什么关系。相比之下 WorldModelBench 做了更多的工作。WorldModelBench 首先提了几个指标，包括是否遵从指令，是否符合物理等，然后众包人类标准了一个数据集，然后训练了一个判别器，然后用判别器微调了 OpenSora-v1.2 模型。从论文展示的结果来看，微调的提升效果有限。

整体上来说，物理对齐比较依赖预训练大模型的能力。对于语言模型来说，对齐往往会降低模型在基准测试上的分数，称为支付对齐税（Alignment Tax）。对于视频模型情况应该是类似的，增强其在物理动态方面的能力可能导致其他能力的削弱。因此，一个更本质的问题是，通过预训练的方式大模型是否能够足够泛化地学到物理规律？字节的工作 [How Far is Video Generation from World Model? A Physical Law Perspective](https://phyworld.github.io) 是这个方向的一个初步探索。

## 屏幕空间物理 Screen-space Physics

Ok，如果视频模型短期内无法达到我们对于物理规律的需求，那我们是否可以通过显式加入物理模拟的方式增强这方面的能力呢？由于视频模型都是 2D 的，我们可以先从屏幕空间（Screen-space）的模拟开始。

屏幕空间模拟（Screen-space Simulation）指的是绕过 3D 模型，直接在屏幕空间中模拟物体的动态。这方面的一个代表性工作是 [PhysGen](https://stevenlsw.github.io/physgen/)。

{{< figure
  src="physgen.png"
  width=80%
  align=center
  caption="[PhysGen](https://stevenlsw.github.io/physgen/) 流程图"
>}}

大致的流程分为三步：1. 给定一张初始图片，我们先做分割，并使用图像理解模型获取法向贴图、语义等信息，然后使用大语言模型推测对应的材质；2. 使用屏幕空间的 2D 模拟器进行模拟（刚体模拟）；3. 使用视频模型将模拟结果与法向贴图等整合起来，得到最终视频。可以发现，这个流程中其实并不需要大模型生成动态，大模型提供的是一个满足时间连续性的“渲染器”，将模拟结果渲染成视频。最近的 [PhysAnimator](https://xpandora.github.io/PhysAnimator/) 也是这个思路，不过将模拟的对象进一步扩展到了布料这样的软性材料。

这样做的优点很明显，我们能够对生成的结果进行非常精确的控制，并且在小幅度内基本满足我们对物理规则的认知。但是缺点同样很明显，由于我们是在屏幕空间做分割和模拟，我们永远只能生成物体一面的结果，像布料的遮挡褶皱也无法处理。并且，生成结果的视角只能是固定的。这使得这类方法只适用于生成动态壁纸这样比较受限的应用，不能作为通用的视频生成方法。

## 三维物理 3D Physics

既然 2D 的模拟终究是妥协，那不如我们直接回到三维模拟。回顾传统的图形管线，生成视频的过程大致可以分为准备3D 资产、进行模拟、渲染结果这三步。这三步中预训练的视频模型可以充当一个非常好的渲染器，比如下图展示的，[Cosmos](https://github.com/NVIDIA/Cosmos) 可以将三维模拟的结果风格迁移到真实场景。

{{< figure
  src="cosmos.png"
  width=90%
  align=center
  caption="Cosmos 介绍[视频](https://www.youtube.com/watch?v=9Uch931cDx8)中展示的三维模拟渲染成真实视频风格的例子"
>}}

在有 3D 资产和物理参数的基础上，模拟也不是一个困难的问题。传统模拟算法在平衡模拟效果和速度上已经提供了非常多的选择。因此最大的问题在于第一步：对于用户给定的一个语言提示，或者是初始帧，如何获取对应的 3D 资产。对应这两种情况我们可以看到两种解决方法，一是训练文本生成 3D 资产的模型，二是从真实图片中重建。

首先，世面上已经有很多专注做文本生成 3D 资产的 AI，比如 [Rodin](https://hyper3d.ai/?lang=zh)，[Meshy](https://www.meshy.ai) 等,可以直接将这些模型导入到像 Houdini、Blender 这样的图形软件中进行模拟。之前受到很多关注的 [Genesis](https://genesis-embodied-ai.github.io) 想做的就是这个思路。另一方面，过程建模（Procedural Modeling）使用形式语言或者节点化的方式描述模型的生成过程，可以将 3D 模型与文本直接联系起来。比如 SVG 图片使用 html 标记语言，CAD 模型可以完全用代码表示，Houdini 用节点系统描述模型等等。在有了代码化的描述之后，我们就可以通过语言模型去生成这些代码，也就生成了 3D 模型。[Infinigen](https://infinigen.org) 通过 Blender 构建了描述自然和室内场景的过程建模语言，因此可以实现文本生成三维场景。[GPT4Motion](https://gpt4motion.github.io) 通过 Blender 实现了无训练，直接从文本生成视频的整个流水线。

如果我们的任务是从初始帧生成视频，就可以考虑从图片重建出 3D 模型。[PhysMotion](https://supertan0204.github.io/physmotion_website/) 使用的方法就是从单张图片进行 Gaussian Splatting 的重建，然后接入物质点法进行模拟，最后经过视频模型进行渲染。如果我们的单视角重建（本质上是对其他视角的生成任务）足够好，那么生成视频的质量就有保证。但是话又说回来，我们不正是应该利用预训练视频模型的能力来帮助单视角生成的任务吗？为什么反而抛弃了大模型在这方面的能力而只把大模型作为一个渲染器呢？

我们可以发现，如果只是用大模型去增强现有的图形管线，那么不可避免的需要很长的管线，并且没有充分利用大模型的能力。最理想的情况是，我们用最少的规则限制和控制信号，提供最基础的三维物理和几何的保证，其他的交给预训练模型补充细节。在这个方向上，[CineMaster](https://cinemaster-dev.github.io) 是一个很有意思的尝试，只通过最简单的包围盒作为条件，就能实现很好的视频控制生成效果。

## 光流 Optical Flow

光流（Optical Flow）可以理解为屏幕空间像素的速度场，因此是一个和物理非常相关的概念。假设我们有了光流场，通过移动像素加上大模型修正，就能生成动态可控的真实视频。[Go-with-the-Flow](https://eyeline-research.github.io/Go-with-the-Flow/) 就使用了这样的想法，使用光流场变换扩散模型的噪声，就能可控生成高质量的视频，效果非常好。那问题来了，光流场要如何得到？一个非常初步的尝试是 [MotionCraft](https://mezzelfo.github.io/MotionCraft/)，直接从传统模拟器中得到光流场，再拿去变换扩散模型的噪声。

## References

- [PhysGen: Rigid-Body Physics-Grounded Image-to-Video Generation](https://stevenlsw.github.io/physgen/)
- [PhyGenBench: Towards World Simulator: Crafting Physical Commonsense-Based Benchmark for Video Generation](https://phygenbench2024.github.io)
- [VideoPhy: Evaluating Physical Commonsense In Video Generation](https://videophy.github.io)
- [PhysMotion: Physics-Grounded Dynamics From a Single Image](https://supertan0204.github.io/physmotion_website/)
- [Cosmos World Foundation Model Platform for Physical AI](https://github.com/NVIDIA/Cosmos)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [VideoScore: Building Automatic Metrics to Simulate Fine-grained Human Feedback for Video Generation](https://tiger-ai-lab.github.io/VideoScore/)
- [How Far is Video Generation from World Model? A Physical Law Perspective](https://phyworld.github.io)
- [OnlineVPO: Align Video Diffusion Model with Online Video-Centric Preference Optimization](https://onlinevpo.github.io)
- [PhysAnimator: Physics-Guided Generative Cartoon Animation](https://xpandora.github.io/PhysAnimator/)
- [GPT4Motion: Scripting Physical Motions in Text-to-Video Generation via Blender-Oriented GPT Planning](https://gpt4motion.github.io)
- [CineMaster: A 3D-Aware and Controllable Framework for Cinematic Text-to-Video Generation](https://cinemaster-dev.github.io)
- [MotionCraft: Physics-Based Zero-Shot Video Generation](https://mezzelfo.github.io/MotionCraft/)
- [Go-with-the-Flow: Motion-Controllable Video Diffusion Models Using Real-Time Warped Noise](https://eyeline-research.github.io/Go-with-the-Flow/)
- [WorldModelBench: Judging Video Generation Models As World Models](https://arxiv.org/abs/2502.20694)
- [Physics IQ Benchmark: Do generative video models understand physical principles?](https://physics-iq.github.io/)