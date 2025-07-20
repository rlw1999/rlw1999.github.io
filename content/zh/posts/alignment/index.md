---
date: '2025-03-16T13:43:41+08:00'
draft: true
title: 'Alignment'
math: true
ShowReadingTime: false
ShowWordCount: false
---

## RLHF

instructGPT 里的 reward model $R$ 实际对应 RL 里的回报（return），是对整条轨迹定义的值。RL 定义的目标函数就是最大化回报：
`$$\max_{\pi_\theta} \mathbb{E}_{\tau \sim \pi_\theta} R(\tau). $$`
instructGPT 多加了一项保证优化之后不要离 SFT 的结果太远：
`$$\max_{\pi_\theta} \mathbb{E}_{(x, y) \sim \pi_\theta} R(x, y) - \beta \mathcal{D}_{KL}\left[\pi_\theta(\cdot | x) \Vert \pi_{ref}(\cdot | x)\right]. $$`
根据 KL 散度的定义上面的定义等价于：
`$$\max_{\pi_\theta} \mathbb{E}_{(x, y) \sim \pi_\theta} \left[ R(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} \right]. $$`
因此我们不需要对 RL 算法本身做任何改变，只需要重新定义回报为：
`$$R(x, y) \gets R(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}.$$`
这个优化目标有闭式解：
`$$
\pi_\theta = \frac{1}{Z(x)}\pi_{ref}(y|x) \exp \left( \frac{1}{\beta} R(x,y) \right).
$$`

## DPO

## GRPO

