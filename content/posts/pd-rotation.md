---
date: '2025-01-27T10:45:44+08:00'
draft: true
title: 'Fast Convergence of Large Rotations'
summary: survey of some graphics papers on this topic
math: true
---

...

## Projective Dynamics

[Projective dynamics](https://www.projectivedynamics.org/Projective_Dynamics/index.html) (PD) is often considered easy to implement and fast to converge:smiley:. This statement is true for real-time simulations, but in the quasi-static setting where the initial position and the converged position are very different, PD can be very slow:worried:. We can show this with a simple mass-spring example ([taichi code](code/pd-rotation/ms-2d.py)).

{{< figure
  src="/image/pd-rotation/ms-1.png"
  width="50%"
  align="center"
  caption="Mass-spring quasi-static example: the bar is bended from the horizontal rest position due to gravity with one end fixed."
>}}

Vanilla Newton's method requires around 10 iterations to converge, while PD requires around 1000 iterations. Though PD has much lower per-iteration cost, it still requires more solution in total. Why does this happen?

From [[Liu et al. 2017]](https://tiantianliu.cn/papers/liu17quasi/liu17quasi.html), PD can be seen as a quasi-Newton method, meaning in each iteration, PD constructs an accurate right-hand-side vector (i.e. the accurate gradient of total energy), but constructs an approximate left-hand-side Hessian. The accurate Hessian of each spring is:

$$ \mathbf{H}_{\text{N} } = k (1 - \frac{l_s}{l_d}) (\mathbf{I}-\bm{n}\bm{n}^T) + k \bm{n}\bm{n}^T, $$

where $k$ is the stiffness, $l_s$ is the static length of the spring, $l_d$ is the deformed length, $\bm{n}$ is the deformed direction of the spring. PD simplifies the accurate Hessian to a constant matrix:

$$ \mathbf{H}_{\text{P} } = k \mathbf{I} = k (\mathbf{I}-\bm{n}\bm{n}^T) + k \bm{n}\bm{n}^T. $$

After we expand the expression of $\mathbf{H}_{\text{P} }$, both Hessians look very similar. The only difference is the coefficient before $\mathbf{I}-\bm{n}\bm{n}^T$. The coefficient $k (1 - \frac{l_s}{l_d})$ in Newton's Hessian is expected to be near zero if $l_s \approx l_d$, while $k$ in PD's Hessian is constantly large. This means for a deformation direction $\bm{d}$ othogonal to $\bm{n}$ (i.e. spinning of the spring), Newton's Hessian has a near zero penalty, while PD's Hessian has a large penalty. In the example we show, the main deformation is global rotation of springs, thus the unwanted large penalty introduced by PD causes slow convergence.

This observation can be generalized to FEM. We do not show the full derivation here, but only the key idea. For each element, the energy is a function of its deformation gradient $\mathbf{F}$, but only the symmetric part $\mathbf{S}$ in the polar decomposition $\mathbf{F}=\mathbf{RS}$ takes a real effect for isotropic energies. Pure rotation deformation should not be penalized, just like the othogonal deformation for springs. PD uses a diagonal Hessian for each element, penalizes all deformations including rotations, thus converges slowly for quasi-static cases.

On the one hand, we don't want unphysical penalties in all directions, on the other hand, it's exactly the reason PD can have a constant Hessian for precomputation acceleration. So how can we solve this dilemma?

## WRAPD

[WRAPD](https://georgbrown.github.io/wrapd/) proposes a way to accelerate PD for quasi-static problems. This work stems from their previous work [ADMM $\supseteq$ PD](https://www.cse.iitd.ac.in/~narain/admm-pd/). 

## Reference

