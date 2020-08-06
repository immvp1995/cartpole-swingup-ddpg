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
The standard setup of reinforcement learning consists of an agent that interacts with the environment 
by selecting actions over a sequence of time in order to maximize a cumulative reward. The environment
may be stochastic. Therefore, we model this as a finite Markov decision process (MDP) that comprises of a state space <a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\fn_phv&space;\small&space;S" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bg_white&space;\fn_phv&space;\small&space;S" title="\small S" /></a>, 
an action space <a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\fn_phv&space;\small&space;A" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bg_white&space;\fn_phv&space;\small&space;A" title="\small A" /></a>, an initial distribution with density <a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\fn_phv&space;\small&space;p_1(s_1)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bg_white&space;\fn_phv&space;\small&space;p_1(s_1)" title="\small p_1(s_1)" /></a>, a stationary transition dynamics distribution with conditional density
<a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\fn_phv&space;\small&space;p(s_{t&plus;1}|s_t,&space;a_t)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bg_white&space;\fn_phv&space;\small&space;p(s_{t&plus;1}|s_t,&space;a_t)" title="\small p(s_{t+1}|s_t, a_t)" /></a>
that satisfies the Markov property <a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\fn_phv&space;\small&space;p(s_{t&plus;1}|s_1,&space;a_1,&space;...,&space;s_t,&space;a_t)=p(s_{t&plus;1}|s_t,&space;a_t)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bg_white&space;\fn_phv&space;\small&space;p(s_{t&plus;1}|s_1,&space;a_1,&space;...,&space;s_t,&space;a_t)=p(s_{t&plus;1}|s_t,&space;a_t)" title="\small p(s_{t+1}|s_1, a_1, ..., s_t, a_t)=p(s_{t+1}|s_t, a_t)" /></a>
and a reward function <a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\fn_phv&space;\small&space;r(s_t,&space;a_t)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bg_white&space;\fn_phv&space;\small&space;r(s_t,&space;a_t)" title="\small r(s_t, a_t)" /></a>
which maps a state-action pair to a scalar [2] [5].

# Environment

# Algorithm

# Results 

# References
[1] V. Mnih et al., “Playing Atari with deep reinforcement learning” in NIPS Deep Learning Workshop 2013, 2013.

[2] D. Silver et al., “Deterministic policy gradient algorithms” in ICML, 2014.

[3] T. P. Lillicrap, “Continuous control with deep reinforcement learning” in ICLR, 2016.
