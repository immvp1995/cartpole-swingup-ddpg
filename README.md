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
the environment by selecting actions over a sequence of time in order to maximize a cumulative reward. The environment 
may be stochastic. Therefore, we model ths as a finite Markov Decision Process (MDP) that comprises of a state space S, 
and action space A, an initial distribution with density, a stationary transition dynamics distribution with conditional 
density that satisfies the Markov Property and a reward function which maps a state-action pair to a scalar [2] [5].

# Environment

# Algorithm

# Results 

# References
[1] V. Mnih et al., “Playing Atari with deep reinforcement learning” in NIPS Deep Learning Workshop 2013, 2013.

[2] D. Silver et al., “Deterministic policy gradient algorithms” in ICML, 2014.

[3] T. P. Lillicrap, “Continuous control with deep reinforcement learning” in ICLR, 2016.
