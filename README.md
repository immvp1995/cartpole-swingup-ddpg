# Introduction
In recent years, researchers in the field of artificial intelligence have made substantial progress in solving 
high-dimensional complex tasks using large, non-linear function approximators to learn value or action-value functions. 

Much of this progress has been achieved by combining advances in deep learning with reinforcement learning. Mnih et al. 
introduced an algorithm called “Deep Q Network” (DQN) that uses a deep neural network to estimate the action-value 
function, which can solve problems with high-dimensional observation spaces and discrete low-dimensional action spaces 
[1]. Prior to this achievement, non-linear function approximators were avoided as guaranteed performance and stable 
learning was deemed impossible. By adapting advancements made by Mnih et al., Lillicrap et al. introduced a model-free, 
off-policy actor-critic named “Deep Deterministic Policy Gradient” (DDPG), which expands the idea of 
“Deterministic Policy Gradients” (DPG) by Silver et al. and can operate in continuous action spaces [1] [2] [3].

In this work we apply DDPG to solve the cartpole swing-up problem.

# Background
The standard setup of reinforcement learning consists of an agent that interacts with 
the environment <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;\bg_white&space;\varepsilon" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{100}&space;\bg_white&space;\varepsilon" title="\varepsilon" /></a> by selecting actions over a sequence of time in order to maximize a cumulative reward. The environment 
may be stochastic. Therefore, we model this as a finite Markov Decision Process (MDP) that comprises of a state space <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;\bg_white&space;S" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{100}&space;\bg_white&space;S" title="S" /></a>, 
and action space <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;\bg_white&space;A" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{100}&space;\bg_white&space;A" title="A" /></a>, an initial distribution <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;\bg_white&space;p_1(s_1)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{100}&space;\bg_white&space;\small&space;p_1(s_1)" title="p_1(s_1)" /></a>, a stationary transition dynamics distribution with conditional density 
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;\bg_white&space;p(s_{t&plus;1}|s_t,&space;a_t)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{100}&space;\bg_white&space;\small&space;p(s_{t&plus;1}|s_t,&space;a_t)" title="p(s_{t+1}|s_t, a_t)" /></a>
that satisfies the Markov Property <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;\bg_white&space;\small&space;p(s_{t&plus;1}|s_1,&space;a_1,&space;...,&space;s_t,&space;a_t)=p(s_{t&plus;1}|s_t,&space;a_t)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{100}&space;\bg_white&space;\small&space;p(s_{t&plus;1}|s_1,&space;a_1,&space;...,&space;s_t,&space;a_t)=p(s_{t&plus;1}|s_t,&space;a_t)" title="\small p(s_{t+1}|s_1, a_1, ..., s_t, a_t)=p(s_{t+1}|s_t, a_t)" /></a> and a reward function <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;\bg_white&space;\small&space;r(s_t,&space;a_t)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{100}&space;\bg_white&space;\small&space;r(s_t,&space;a_t)" title="\small r(s_t, a_t)" /></a>
which maps a state-action pair to a scalar [2] [5].

In general, the agent follows a stochastic policy <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;\bg_white&space;\pi" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{100}&space;\bg_white&space;\pi" title="\pi" /></a>
that maps a state to a set of probability measures on <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;\bg_white&space;A" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{100}&space;\bg_white&space;A" title="A" /></a>. The agent uses <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;\bg_white&space;\pi" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{100}&space;\bg_white&space;\pi" title="\pi" /></a>
to interact with the MDP to generate trajectories of states, action and rewards. In each step these rewards are discounted by a factor <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;\bg_white&space;\small&space;\gamma&space;\in&space;[0,&space;1]" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{100}&space;\bg_white&space;\small&space;\gamma&space;\in&space;[0,&space;1]" title="\small \gamma \in [0, 1]" /></a>
such that the future discounted return at time <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;\bg_white&space;t" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{100}&space;\bg_white&space;t" title="t" /></a> can be defined as <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;\bg_white&space;\small&space;R_t&space;=&space;\sum_{i=t}^{T}&space;\gamma^{i-t}r_i" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{100}&space;\bg_white&space;\small&space;R_t&space;=&space;\sum_{i=t}^{T}&space;\gamma^{i-t}r_i" title="\small R_t = \sum_{i=t}^{T} \gamma^{i-t}r_i" /></a>.
The goal of the agent is to learn a policy that maximizes the expected return from the start distribution
[2][5].

# Environment

# Algorithm

# Results 

# References
[1] V. Mnih et al., “Playing Atari with deep reinforcement learning” in NIPS Deep Learning Workshop 2013, 2013.

[2] D. Silver et al., “Deterministic policy gradient algorithms” in ICML, 2014.

[3] T. P. Lillicrap, “Continuous control with deep reinforcement learning” in ICLR, 2016.
