from environment import CartPoleSwingUpContinuousEnv
from gym import make as gym_make


def make(env_name, *make_args, **make_kwargs):
    if env_name == "CartPoleSwingUpContinuous":
        return CartPoleSwingUpContinuousEnv()
    else:
        return gym_make(env_name, *make_args, **make_kwargs)


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []
    counter = 0

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state, (episode + 1) * (step + 1))
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            # update the agent if enough transitions are stored in replay buffer
            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward)
                # Count number of consecutive games with cumulative rewards >-55 for early stopping
                if episode_reward > -55:
                    counter += 1
                else:
                    counter = 0
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state
        # Early stopping, if cumulative rewards of 10 consecutive games were >-55
        if counter == 10:
            break

    return episode_rewards