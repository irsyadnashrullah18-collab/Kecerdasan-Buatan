# Example 3.23 OpenAI Gym CartPole (Fixed for Gym updates)
import gym

env = gym.make('CartPole-v0')

for i_episode in range(20):
    # Penyesuaian reset() versi baru
    state = env.reset()
    observation = state[0] if isinstance(state, tuple) else state

    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()

        # Penyesuaian step() (menangani 4 atau 5 nilai kembalian)
        step_res = env.step(action)
        if len(step_res) == 5:
            observation, reward, terminated, truncated, info = step_res
            done = terminated or truncated
        else:
            observation, reward, done, info = step_res

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

env.close()