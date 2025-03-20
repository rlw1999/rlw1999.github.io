---
date: '2025-03-16T13:43:41+08:00'
draft: false
title: 'Reinforcement Learning'
math: true
ShowReadingTime: false
ShowWordCount: false
---

一文速通主流强化学习算法。

## 问题定义 Problem Definition

强化学习（reinforcement learning）整体的目标是优化策略（policy），使得智能体（agent）能够在与环境（environment）的交互中获得最大回报（return）。为了方便讨论，我们只在最简化的情况下考虑问题。

{{< figure
  src="rl_bg.png"
  width=80%
  align=center
  caption="智能体-环境交互循环 from [OpenAI](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)"
>}}

智能体和环境的整体描述称为一个状态（state）$s$。智能体可以通过感受器获得状态的部分观察（observation）$o$，为了简单我们认为智能体可以观察到全部信息 $s$。智能体的策略 $\pi$ 描述为一个神经网络，参数为 $\theta$，根据观测生成行动（action）$a$ 的分布：
$$a \sim \pi_\theta(\cdot | s).$$
$\pi_\theta(a | s)$ 就是在策略 $\pi$ 下，根据观测 $s$ 生成行动 $a$ 的概率。智能体的行动会改变状态，对应状态转移也可以建模为一个概率过程：
$$s_{t+1} = P(\cdot | s_t, a_t).$$
这个状态转移是与策略无关的，并且一般假设与更早的状态无关，也就是对应马尔可夫过程（Markov process）。我们自然假定智能体和环境会一轮一轮地进行交互，形成一条轨迹（trajectory， 或者 episode，rollout）：
$$\tau = (s_0, a_0, s_1, a_1, \cdots).$$
轨迹可能是无穷长的，或者到达终止状态后停止。

智能体的每一步交互都会对应一个反馈（或者奖励，reward）$r$：
$$r_t = R(s_t, a_t, s_{t+1}).$$
这些反馈最终积累起来变成一条轨迹的回报（return）$G$：
$$G(\tau)=\sum_{t=0}^T r_t.$$
这种直接将反馈相加的方式需要设定一个最大窗口长度，或者最大回报阈值。想要考虑无穷长的轨迹的反馈，可以定义一个打折因子 $\gamma\in (0, 1)$，定义回报为：
$$G(\tau)=\sum_{t=0}^\infty \gamma^t r_t.$$
最终我们的目标，是要得到一个策略，可以在所有可能轨迹上得到最大平均回报：
$$\max_\pi J(\pi) = \max_\pi \mathbb{E}_{\tau \sim P(\cdot | \pi)}G(\tau).$$
其中 $P(\tau | \pi)$ 是在策略 $\pi$ 下生成轨迹 $\tau$ 的概率，根据之前的定义，可以写为：
`$$P(\tau | \pi)=p_0(s_0) \prod_{t=0}^{T} P(s_{t+1}|s_t, a_t)\pi(a_t|s_t).$$`

我们可以结合 LLM 的例子理解一下上面的概念。语言模型是一个智能体，动作是生成的下一个 Token，所有的词表组成了全部的动作空间。状态对应用户提供的 prompt 和之前模型输出的上下文，因此状态转移 $P(s_{t+1}|s_t, a_t)$ 是一个确定模型，轨迹的概率由策略概率完全决定。我们的目标，就是优化语言模型本身，提高其获得我们定义的回报，比如正确回答数学问题、正确理解人类意图等。

## 策略梯度 Policy Gradient

策略梯度（policy gradient）是最直接的优化回报的方法，其关键是计算目标 $J$ 对于策略网络的梯度，然后使用梯度上升进行更新：
$$\theta_{k+1} \gets \theta_k + \alpha \nabla_\theta J(\pi_\theta) \big|_{\theta_k}.$$
下面我们来看如何计算这一梯度。由于 $J$ 定义为所有轨迹的期望值，我们自然无法直接计算这一期望，一个常见的想法就是将其转化为采样。也就是我们按照某个概率采样一堆轨迹 $\tau_i$，然后计算平均的 $G_i$。比如，我们可以就使用当前的策略网络 `$\pi_{\theta_k}$` 与环境交互来生成一堆轨迹 $\tau_i$，这样的做法称为 on-policy。与之对应的是 off-policy，也就是用于生成轨迹的策略不是当前的策略。这里我们假定使用 on-policy 的方法，推导 `$\nabla_\theta J$` 的形式：
`$$
\begin{aligned}
  \nabla_\theta J &= \nabla_\theta \int_\tau P(\tau | \pi) G(\tau) \\
  &= \int_\tau \textcolor{blue}{\nabla_\theta P(\tau | \pi)} G(\tau) \\
  &= \int_\tau \textcolor{blue}{P(\tau | \pi) \nabla_\theta \log P(\tau | \pi)} G(\tau) \\
  &= \int_\tau P(\tau | \pi) \left[\nabla_\theta \log \left( \prod_{t=0}^T \pi_\theta(a_t | s_t)\right) + \cancel{\nabla_\theta \log \left(\prod_{t=0}^T P(s_{t+1}|s_t, a_t)\right)} \right] G(\tau) \\
  &= \int_\tau P(\tau | \pi) \sum_{t=0}^T G(\tau) \nabla_\theta \log \pi_\theta(a_t | s_t) \\
  &= \mathbb{E}_{\tau \sim \pi} \sum_{t=0}^T G(\tau) \nabla_\theta \log \pi_\theta(a_t | s_t).
\end{aligned}
$$`

因此，如果我们根据策略 $\pi_{\theta_k}$ 采样了一堆轨迹构成了一个数据集 `$\mathcal{D}=\{\tau_i\}$`，那么得到的梯度为：
$$\hat{g} = \frac{1}{|\mathcal{D}|}\sum_{\tau \in \mathcal{D}}\sum_{t=0}^T G(\tau_i) \nabla_\theta \log \pi_\theta(a_t | s_t).$$
到此我们就推导出了最原始的策略梯度算法的核心公式，这种算法一般称为 REINFORCE 算法。

为了符合 pytorch 等深度学习框架，我们需要定义一个损失函数，使其梯度与上面对应。根据当前策略采样出的一堆轨迹构成了我们的数据集 $\mathcal{D}$，两个求和符号遍历了数据集里的所有 $(s_t, a_t)$ 对，每对对应一次策略网络的 forward。每一对 $(s, a)$ 的 loss 可以定义为：
$$L = - G(\tau) \log \pi_\theta(a | s), $$
对其进行梯度反传就可以优化策略网络了。

我们可以大概理解这个 loss 到底在干什么。对于策略网络输出的每一次决策 $(s_t, a_t)$，如果最终导致 $G(\tau)>0$，那么我们会鼓励这个决策，让网络增大在 $s_t$ 时生成 $a_t$ 的概率，并且 $G(\tau)$ 越大，我们鼓励得越多。$G(\tau)<0$ 类似，负得越多，我们越抑制这个决策。

这个最简单的 REINFORCE 算法理论上已经可以工作了，但是在实践中不是最有效的，后面我们介绍的算法都是 REINFORCE 的改进。很核心的一点是，上面的算法是通过 $G(\tau)$ 调节不同决策的鼓励或者抑制，而 $G(\tau)$ 是未经归一化的，数值相差可以很大；并且 $G(\tau)$ 大并不代表单次决策成功，二者虽然有正相关性，但是也有很大的方差。尽管从期望的角度算法可以收敛，但是较大的数值噪声会减慢收敛的速度，甚至导致不收敛。那么问题就变成了，如何用尽可能低的方差无偏地评价每个动作的好坏程度？

一个直接的减小方差的方程称为 reward-to-go，表示在计算 loss 时只考虑 $(s_t, a_t)$ 之后的回报：
$$L = - \log \pi_\theta(a_t | s_t)\sum_{t'=t}^T r_{t'}. $$
证明的核心在于利用马尔可夫性，$0 \sim t - 1$ 时刻的反馈与 $t$ 时刻的决策是完全独立的。具体证明可以参考[文章](https://zhuanlan.zhihu.com/p/21372386742)。
{{< sidenote >}}
这里我们为了简化公式没有乘衰减因子 $\gamma$，有衰减时结论类似。
{{< /sidenote >}}

另一个非常重要的想法是引入基线（baseline），也就是对 $G(\tau)$ 均值的估计。假设我们对状态 $s_t$ 所有可能反馈的均值有估计 $b_t$，与策略无关，那么我们可以从反馈中减去基线 $b_t$，重新定义损失函数为：
$$L = - \log \pi_\theta(a_t | s_t)\sum_{t'=t}^T (r_{t'} - b_{t}). $$
注意这里是 $b_t$ 而不是 $b_{t'}$。这个损失函数的好处在于，有了基线之后，我们就知道哪些行动是好的，可以鼓励，哪些是不好的，可以抑制，帮助算法稳定收敛，而不是所有行动都加上不同程度的鼓励。这个损失函数的正确性证明这里也不展开了，详细请参考[文章](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)。

那么到底基线要如何计算呢？这里我们先给出形式化的定义，在下面一节再给出具体的计算方法。对于一个强化学习问题，我们可以定义状态值（state value）$v$ 和动作值（action value） $q$：
`$$
\begin{aligned}
  v_\pi(s) &= \mathbb{E}_{\tau \sim \pi} [G(\tau)|s_0=s], \\
  q_\pi(s,a) &= \mathbb{E}_{\tau \sim \pi} [G(\tau)|s_0=s,a_0=a].
\end{aligned}
$$`
那么 $v_\pi(s_t)$ 就是我们想要的基线 $b_t$。根据定义，我们有 `$v_\pi(s)=\mathbb{E}_{a \sim \pi} q_\pi(s, a)$`。除此之外，我们还可以定义优势（advantage）：
`$$ A_\pi (s,a) = q_\pi(s,a) - v_\pi(s), $$`
反映了当前动作相对于该状态下平均策略动作的“优势”程度。

有了这些定义之后，我们可以重写上面的损失函数形式。首先，reward-to-go 的损失函数等价于使用动作值：
$$L = - q_{\pi_\theta}(s_t, a_t) \log \pi_\theta(a_t|s_t).$$
具体证明参考[材料](https://spinningup.openai.com/en/latest/spinningup/extra_pg_proof2.html)，核心依然是马尔代夫链的无后效性。其次，引入基线之后，等价于使用优势函数：
$$L = - A_{\pi_\theta}(s_t, a_t) \log \pi_\theta(a_t|s_t).$$
因此，后面的算法我们都默认使用优势函数的损失函数形式。

## 贝尔曼方程 Bellman Equation

如果我们想拟合 $v_\pi(s)$，最简单的想法是根据定义采样一堆轨迹，计算回报的期望，也就是蒙特卡洛法（Monte Carlo，MC）。不过这里我们会遇到和上面类似的问题，尽管这个估计是无偏的，但是方差太大了。而且对每个状态我们都需要如此采样，对应大规模的问题计算量无法承受。幸运的是，我们可以利用问题的马尔可夫性，得到 $v_\pi$ 的递推关系，也就是贝尔曼方程（Bellman equation）：
`$$
v_\pi (s) = \mathbb{E}_{a\sim \pi, s' \sim P} [r(s,a) + \gamma v_\pi (s')].
$$`
这个方程理解起来并不复杂，根据回报 $G$ 的定义和马尔可夫性可以直接得到。相对应的，对于动作值我们也有：
`$$
q_\pi (s,a) = \mathbb{E}_{s'\sim P}[r(s,a) + \gamma v_\pi (s')] = \mathbb{E}_{s'\sim P, a' \sim \pi}[r(s,a) + \gamma q_\pi (s', a')].
$$`

我们使用一个参数为 $\phi$ 的网络 $v_\phi$ 近似状态值，下面考虑如何使用贝尔曼方程使其收敛到真值 $v_\pi$。假设我们在策略网络 $\pi_\theta$ 的一次 forward 之后，从状态 $s_t$ 更新到了状态 $s_{t+1}$，获得了反馈 $r_t$，那么从期望的角度，贝尔曼方程右侧的估计就为 $r_t + \gamma v_\phi(s_{t+1})$，而左侧的估计为 $v_\phi$。由于网络还没有收敛，这两个估计存在差别 $\delta_t$：
`$$
\delta_t = r_t + \gamma v_\phi(s_{t+1}) - v_\phi(s_t).
$$`

TBD