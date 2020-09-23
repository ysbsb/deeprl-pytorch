"""Play Q-table frozenlake
"""
import time
import gym
import numpy as np
import utils_prints as print_utils

N_ACTIONS = 4
N_STATES = 16

# Set learning parameters
LEARNING_RATE = .5
DISCOUNT_RATE = .98

N_EPISODES = 2000


def main():
    """Main"""
    frozen_lake_env = gym.make("FrozenLake-v0")

    # Initialize table with all zeros
    Q = np.zeros([N_STATES, N_ACTIONS])

    # Create lists to contain total rewards and steps per episode
    rewards = []

    for i in range(N_EPISODES):
        # Reset environment and get first new observation
        state = frozen_lake_env.reset()
        episode_reward = 0
        done = False

        # The Q-Table learning algorithm
        while not done:
            # Choose an action by greedily (with noise) picking from Q table
            noise = np.random.randn(1, N_ACTIONS) / (i+1)
            action = np.argmax(Q[state, :] + noise)

            # Get new state and reward from environment
            new_state, reward, done, _ = frozen_lake_env.step(action)

            reward = -1 if done and reward < 1 else reward

            # Update Q-Table with new knowledge using learning rate
            Q[state, action] = (
                1 - LEARNING_RATE) * Q[state, action] + LEARNING_RATE * (reward + DISCOUNT_RATE * np.max(Q[new_state, :]))

            episode_reward += reward
            state = new_state

        rewards.append(episode_reward)

    print("Score over time: " + str(sum(rewards) / N_EPISODES))
    print("Final Q-Table Values")

    for i in range(10):
        # Reset environment and get first new observation
        state = frozen_lake_env.reset()
        episode_reward = 0
        done = False

        # The Q-Table learning algorithm
        while not done:
            # Choose an action by greedily (with noise) picking from Q table
            action = np.argmax(Q[state, :])

            # Get new state and reward fromm environment
            new_state, reward, done, _ = frozen_lake_env.step(action)
            print_utils.clear_screen()
            frozen_lake_env.render()
            time.sleep(.1)

            episode_reward += reward
            state = new_state

            if done:
                print("Episode Reward: {}".format(episode_reward))
                print_utils.print_result(episode_reward)

        rewards.append(episode_reward)

    frozen_lake_env.close()


if __name__ == '__main__':
    main()