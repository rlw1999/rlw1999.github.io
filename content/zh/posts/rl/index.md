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
于是自然我们可以使用 $\delta_t$ 来定义损失函数：
$$L = \frac{1}{2} \delta_t^2 = \frac{1}{2} [r_t + \gamma v_\phi(s_{t+1}) - v_\phi(s_t)]^2.$$
当损失函数完全降到 $0$ 时，值网络收敛到真值。这里注意一个细节，尽管这个损失函数的定义需要网络 forward 两次，但是在梯度反传的时候我们往往会固定 $v_\phi(s_{t+1})$ 不反传梯度，只反传 $v_\phi(s_t)$。这样不仅能简化实现，还能稳定训练。我们这里其实使用的是一个自回归的思想，或者说是机器学习领域的 bootstrapping 思想。每次我们让网络值更接近真值一些，但是不直接用真值作为监督，以减小方差。这种利用贝尔曼方程回归值函数的方法称为时序差分（time difference，TD）。我们还可以多展开（rollout）几步，计算贝尔曼方程左右估计的差别，得到：
$$L = \frac{1}{2}\left[G^n(s_t) - v_\phi(s_t) \right]^2, \ G^{(n)}(s_t) = \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n v_\phi(s_{t+n}).$$
这样的算法称为 n-step TD。可以发现，当 $n$ 趋向正无穷时，n-step TD 其实就是蒙特卡洛估计。因此 $n$ 不宜过大，也不宜过小。进一步，我们可以发现上面我们虽然展开了多步，但是值网络只固定 forward 了两次，并没有用到中间状态 $v_\phi(s_{t+i})$ 的估计值。我们可以考虑计算每一步的 $G^{(n)}$，然后引入一个新的衰减因子 $\lambda$，定义总回报的估计为：
$$G^\lambda(s_t) = (1-\lambda) \sum_{i=1}^{\infty} \lambda^{i-1} G^{(i)}(s_t).$$
可以发现我们就是对 $G^{i}$ 进行了加权平均，展开越长权重越小。这样的做法称为 TD($\lambda$)。当 $\lambda=0$ 时，$G^0(s_t)$ 回退为单步展开，因此单步 TD 也称为 TD($0$)。

在实际应用时，我们可以一直展开到结束再回头更新值网络，这样的视角称为前向视角。反过来，每当我们展开一步，这对应前一状态的 $G^{(1)}$，前两步状态的 $G^{(2)}$，前三步状态的 $G^{(3)}$……因此我们可以用一步展开去更新之前所有状态的值，这样的视角称为后向视角。我们这里不进一步展开细节了，因为下面我们将了解 $TD(\lambda)$ 的思想在策略梯度中更常用的形式：广义优势估计（general advantage estimation，GAE）。

## 广义优势估计 GAE

在策略梯度中，我们最后使用优势函数定义带基线的损失函数，然后在贝尔曼方程部分我们了解了可以使用 TD($\lambda$) 估计基线，那自然想到可不可以直接用 TD($\lambda$) 的思想估计优势函数 $A$。我们可以根据定义直接推导：
`$$
\begin{aligned}
  A(s_t, a_t) =& q(s_t, a_t) - v(s_t) \\
  \approx & G^\lambda(s_t, a_t) - v_\phi(s_t) \\
  =& - v_\phi(s_t) + (1-\lambda)[\textcolor{blue}{r_t} + \gamma v_\phi(s_{t+1})] + \\
  &(1-\lambda)\lambda[\textcolor{blue}{r_t} + \gamma \textcolor{red}{r_{t+1}} + \gamma^2 v_\phi(s_{t+2})] + \\
  &(1-\lambda)\lambda^2[\textcolor{blue}{r_t} + \gamma \textcolor{red}{r_{t+1}} + \gamma^2 r_{t+2} + \gamma^3 v_\phi(s_{t+3})] + \cdots \\
  =& - v_\phi(s_t) + \textcolor{blue}{r_t} + \lambda\gamma \textcolor{red}{r_{t+1}} + (\lambda\gamma)^2 r_{t+2} + \cdots + \\
  & (1-\lambda)\gamma v_\phi(s_{t+1}) + (1-\lambda)\lambda\gamma^2 v_\phi(s_{t+2}) + \cdots \\
  =& [r_t + \gamma v_\phi(s_{t+1}) - v_\phi(s_t)] + \\
  &(\lambda\gamma)[r_{t+1} + \gamma v_\phi(s_{t+2}) - v_\phi(s_{t+1})] + \\
  &(\lambda\gamma)^2[r_{t+2} + \gamma v_\phi(s_{t+3}) - v_\phi(s_{t+2})] + \cdots \\
  =& \delta_t + (\lambda\gamma)\delta_{t+1} + (\lambda\gamma)^2 \delta_{t+2} + \cdots 
\end{aligned}
$$`
这个推导的过程比较复杂，但结果是清晰的：每一步更新我们都可以计算一个 $\delta_t$（称为 TD error），累计起来就可以得到优势函数的估计，记为 $A^{GAE}_t$。并且在实现中，我们还可以将上面的公式写为递推形式。假设我们展开了一条轨迹，从后往前遍历，`$A^{GAE}_t$` 与 `$A^{GAE}_{t+1}$` 的关系为：
`$$A^{GAE}_t = \delta_t + (\lambda \gamma) A^{GAE}_{t+1}.$$`
所以我们只需要从后往前遍历一次轨迹就可以得到优势函数的估计，用于训练策略网络。对于值网络而言，我们可以用 `$A^{GAE}_t + v_\phi(s_t)$` 作为监督，定义损失函数：
`$$
L = \frac{1}{2} [v^{tar} - v_\phi(s_t)]^2, \ \ v^{tar} = A^{GAE}_t + v_\phi(s_t).
$$`

我们可以从两个角度理解这个公式。第一个角度是我们其实在最小化 $A^{GAE}$，也就是最小化 TD error $\delta$，这跟我们之前理解训练值网络的方向相同。第二个角度是 $A^{GAE} + v$ 其实是对动作值 $q$ 的估计，由于我们的所有轨迹都是从策略 $\pi$ 中采的，因此采样的动作值应该和值网络的输出是一样的。

到这里我们就得到了一个带基线的策略梯度算法，使用了两个网络：$\pi_\theta$ 用于生成策略分布，$v_\phi$ 用于估计状态值。这样的算法称为演员-裁判（actor-critic，AC）算法，更具体的是使用优势函数的演员-裁判（advantage actor-critic, A2C）算法，每步训练的伪代码如下：
```python
# 伪代码由 deepseek-R1 生成
# 1. 收集轨迹数据
states, actions, rewards, dones, next_states = [], [], [], [], []
for _ in range(num_steps):
    # 并行执行多个环境（若num_envs > 1）
    for env in 0, 1, ..., num_envs-1:
        s = current_state[env]
        a ~ PolicyNN(s)                 # 采样动作
        s1, r, done = env.step(a)  # 与环境交互
        # 存储数据
        states.append(s)
        actions.append(a)
        rewards.append(r)
        dones.append(done)
        next_states.append(s1)
        # 重置环境（如果终止）
        if done:
            current_state[env] = env.reset()

# 2. 计算GAE和值目标
values = ValueNN(states)                # 当前状态值V(s_t)
next_values = ValueNN(next_states)      # 下一状态值V(s_{t+1})

# 初始化GAE和回报计算
advantages = []
returns = []
A_next = 0
V_target_next = 0

# 反向遍历时间步
for t in reversed(range(num_steps * num_envs)):
    # 计算 TD error
    delta = rewards[t] + gamma * (1 - dones[t]) * next_values[t] - values[t]
    # 计算 GAE 优势 A_t
    A_t = delta + gamma * lambda_ * (1 - dones[t]) * A_next
    advantages.append(A_t)
    # 计算值目标 V_target
    V_target = values[t] + A_t
    returns.append(V_target)
    # 更新下一时间步的A_next和V_target_next
    A_next = A_t
    V_target_next = V_target if not dones[t] else 0
# 反转列表以对齐时间顺序
advantages = advantages[::-1]
returns = returns[::-1]

# 3. 归一化优势（可选）
advantages = (advantages - mean(advantages)) / (std(advantages) + 1e-8)

# 4. 更新Actor网络（策略梯度）
log_probs = PolicyNN(states).log_prob(actions)   # 计算动作对数概率
actor_loss = - (log_probs * advantages).mean() - entropy_coef * PolicyNN.entropy().mean()
optimizer_actor.zero_grad()
actor_loss.backward()
optimizer_actor.step()

# 5. 更新Critic网络（值函数回归）
critic_loss = (ValueNN(states) - returns).pow(2).mean()
optimizer_critic.zero_grad()
critic_loss.backward()
optimizer_critic.step()
```

## 近端策略优化 PPO

强化学习问题与监督学习问题不同，监督学习的梯度方向是显式定义且无偏的，但强化学习的梯度方向必须通过采样得到，从而引入大的方差。一个例子是，一个高手下出的一步棋可能是非常好的，但是让新手来判断，来采样这步棋之后怎么走，可能根本不能明白这步棋妙在哪里。因此，我们历史的策略同时影响着我们对于每个动作的判断，演员和裁判是在共同进步。这反映了强化学习的两个问题：

1. 策略更新的非平稳性：策略更新会导致采样的偏移，从而导致策略评价的改变。
2. 探索-利用困境：探索（随机采样）可以获得更多无偏大信息，但增大方差；利用（已有的策略）可以减小方差，但带来偏差。

近端策略优化（proximal policy optimization，PPO）算法相比于 A2C 算法可以缓解这些问题，实现更稳定的收敛。近端（proximal）这个词来源于优化算法中的近端算子（proximal operator）。在凸优化中，近端算子用于解决含非光滑项的优化问题，其形式为：
$$
\text{prox}_{\lambda f}(v) = \arg\min_x \left( f(x) + \frac{1}{2\lambda} \|x - v\|^2 \right)
$$
近端算子的核心思想是：在每一步迭代中，既优化目标函数 $f(x)$，又保证解 $x$ 不会偏离当前点 $v$ 太远（通过二次项约束）。在 PPO 中我们借用这个思想，保证策略每步更新时不要离上一步太远，避免策略过早放弃探索，陷入局部困境。在具体实现中，有两种 PPO 算法：
1. PPO-penalty：类似于近端算子的做法，在损失函数上加上原策略 $\pi_\theta$ 和新策略 $\pi_{\theta + \Delta \theta}$ 的 KL 散度，保证策略更新步长不要太大。
2. PPO-clip：直接对梯度进行裁剪，避免更新步长超过某个阈值。

实际中 PPO-clip 实现更简单，效果也更稳定，因此下面只介绍 PPO-clip 算法。

假设我们基于 $\pi_{\theta_k}$ 采样了一堆轨迹，我们希望使用这些轨迹更新策略 $\pi_\theta$，并且不要离开 $\pi_{\theta_k}$ 太远。由于我们要限制策略更新的步长，所以希望能尽可能重复使用基于 $\pi_{\theta_k}$ 采样的轨迹，避免重复采样相似的策略带来的计算开销。这个过程类似于 off-policy 的思想，即采样数据的策略不同于当前策略，但是区别在于这二者非常接近。对于 A2C 算法而言，单步策略的梯度可以写为：
`$$g = \mathbb{E}_{(s_t, a_t) \sim \pi_\theta} A_{\pi_\theta} (s_t, a_t) \nabla_\theta \log \pi_\theta (a_t | s_t). $$`

对于 PPO，我们数据分布不再是 $\pi_\theta$ 采样得到的，而是 $\pi_{\theta_k}$ 采样的结果。因此，我们可以使用重要性采样，将其转化为对 $\pi_{\theta_k}$ 的期望，得到：
`$$
\begin{aligned}
  g &= \mathbb{E}_{(s_t, a_t) \sim \pi_\theta} A_{\pi_\theta} (s_t, a_t) \nabla_\theta \log \pi_\theta (a_t | s_t) \\
  &= \mathbb{E}_{(s_t, a_t) \sim \textcolor{blue}{\pi_{\theta_k}}} \frac{P_\theta(a_t, s_t)}{\textcolor{blue}{P_{\theta_k}(a_t, s_t)}} A_{\pi_\theta} (s_t, a_t) \nabla_\theta \log \pi_\theta (a_t | s_t) \\
  &= \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta_k}} \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_k}(a_t|s_t)} \frac{P_\theta(s_t)}{P_{\theta_k}(s_t)} A_{\pi_\theta} (s_t, a_t) \nabla_\theta \log \pi_\theta (a_t | s_t) \\
  &\approx \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta_k}} \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_k}(a_t|s_t)} A_{\pi_{\theta_k}} (s_t, a_t) \nabla_\theta \log \pi_\theta (a_t | s_t) \\
  &= \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta_k}} \frac{\nabla_\theta \pi_\theta(a_t|s_t)}{\pi_{\theta_k}(a_t|s_t)} A_{\pi_{\theta_k}} (s_t, a_t).
\end{aligned} 
$$`
这一步近似中，我们利用了 $\theta$ 与 $\theta_k$ 接近的假设，认为 $P_\theta(s_t) \approx P_{\theta_k}(s_t)$ 以及 $A_{\pi_{\theta}} \approx A_{\pi_{\theta_k}}$。因此我们可以定义在做梯度裁减之前的目标函数为：
`$$L = - \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_k}(a_t|s_t)} A_{\pi_{\theta_k}} (s_t, a_t).$$`
在这个形式之上我们可以定义带裁减的 PPO 的损失函数形式：
`$$
L = - \min \left[\frac{\pi_\theta}{\pi_{\theta_k}} A_{\pi_{\theta_k}}, \text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_k}}, 1 - \varepsilon, 1 + \varepsilon\right)A_{\pi_{\theta_k}} \right].
$$`
这个形式写得很复杂，我们可以分成两种情况来理解。如果 $A_{\pi_{\theta_k}} > 0$，表示这个行动应该是好的，此时损失函数为：
`$$
L = - \min \left(\frac{\pi_\theta}{\pi_{\theta_k}}, 1 + \varepsilon \right)A_{\pi_{\theta_k}}.
$$`
涵义是假设新策略已经好到超过旧策略的 $1+\varepsilon$ 倍时，梯度为 $0$，别再进一步鼓励这个行动了。与之相对的，如果 $A_{\pi_{\theta_k}} < 0$，表示这个行动不行，此时损失函数为：
`$$
L = - \max \left(\frac{\pi_\theta}{\pi_{\theta_k}}, 1 - \varepsilon \right)A_{\pi_{\theta_k}}.
$$`
涵义是假设新策略已经减小这个坏行动的概率到旧策略的 $1-\varepsilon$ 倍时，梯度为 $0$，不再进一步惩罚这个行动了。

在实现中，PPO 算法会在采样得到轨迹之后，少量多次地更新策略，与 A2C 算法核心的差别由下面的伪代码给出：
```python
# 代码由 deepseek-R1 生成
# === PPO 独有部分 ===

# 1. 记录旧策略的动作概率（重要性采样基础）
old_probs = actor(states).detach()  # 旧策略概率（固定梯度）

# 2. 多次小批量更新（数据重用）
for _ in range(K_EPOCHS):  # 通常 K_EPOCHS=3~10
    
    # 3. 计算新策略比率（核心机制）
    new_probs = actor(states)
    ratios = new_probs[actions] / old_probs[actions]  # 策略比率 r(θ)
    
    # 4. 剪切目标函数（信赖域约束）
    clipped_ratios = torch.clamp(ratios, 1 - CLIP_EPS, 1 + CLIP_EPS)
    surr1 = ratios * advantages
    surr2 = clipped_ratios * advantages
    actor_loss = -torch.min(surr1, surr2).mean()  # 剪切后的损失
    
    # 5. 熵正则化（鼓励探索）
    entropy = -torch.sum(new_probs * torch.log(new_probs), dim=1).mean()
    actor_loss += ENTROPY_COEF * entropy  # 通常 ENTROPY_COEF=0.01
    
    # 6. 仅更新 Actor（Critic 更新与A2C相同）
    optimizer_actor.zero_grad()
    actor_loss.backward()
    optimizer_actor.step()
```