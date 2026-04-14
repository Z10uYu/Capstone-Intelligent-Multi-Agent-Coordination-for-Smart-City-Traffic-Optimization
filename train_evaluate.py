"""
train_evaluate.py
Trains DQN agents on the traffic environment and evaluates performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from traffic_env import TrafficGymEnv, create_grid_network
from dqn_agent import DQNAgent

def train(num_episodes=100, steps_per_episode=200, arrival_rate=0.4):
    # Create environment
    G = create_grid_network(rows=2, cols=2)   # 4 intersections
    traffic_lights = {node: ['NS_GREEN', 'EW_GREEN'] for node in G.nodes}
    env = TrafficGymEnv(G, traffic_lights, arrival_rate=arrival_rate, max_steps=steps_per_episode)

    # Create agents for each intersection
    agents = {}
    state_dim = 6   # 4 queues + 2 phase one-hot
    action_dim = 2
    for aid in env.agent_ids:
        agents[aid] = DQNAgent(state_dim, action_dim)

    episode_rewards = []
    avg_waiting_times = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = {aid: 0.0 for aid in agents}
        done = False
        step = 0
        while not done:
            # Select actions
            actions = {}
            for aid, agent in agents.items():
                state = obs[aid]
                actions[aid] = agent.act(state)
            # Step environment
            next_obs, rewards, done, truncated, info = env.step(actions)
            # Store transitions and learn
            for aid, agent in agents.items():
                agent.remember(obs[aid], actions[aid], rewards[aid], next_obs[aid], done)
                agent.learn()
                total_reward[aid] += rewards[aid]
            obs = next_obs
            step += 1
            if step >= steps_per_episode:
                done = True

        # Log episode performance
        ep_total_reward = sum(total_reward.values())
        episode_rewards.append(ep_total_reward)
        avg_wait = info['total_waiting_time'] / max(1, info['total_vehicles_served'])
        avg_waiting_times.append(avg_wait)
        print(f"Episode {ep+1}: Total Reward = {ep_total_reward:.2f}, Avg Wait = {avg_wait:.2f}s, Vehicles={info['total_vehicles_served']}")

    # Save trained models
    for aid, agent in agents.items():
        agent.save(f"dqn_traffic_{aid}.pth")
    return agents, episode_rewards, avg_waiting_times

def evaluate(agents, num_episodes=10, steps_per_episode=200, arrival_rate=0.4):
    G = create_grid_network(2,2)
    traffic_lights = {node: ['NS_GREEN', 'EW_GREEN'] for node in G.nodes}
    env = TrafficGymEnv(G, traffic_lights, arrival_rate=arrival_rate, max_steps=steps_per_episode)

    fixed_timing_metrics = []
    mas_metrics = []

    for ep in range(num_episodes):
        # Fixed timing baseline (alternating phases without learning)
        obs, _ = env.reset()
        total_wait_fixed = 0
        total_veh_fixed = 0
        done = False
        step = 0
        while not done:
            # Fixed timing: switch every 10 steps
            actions = {}
            for aid in env.agent_ids:
                actions[aid] = 1 if (step % 10) < 5 else 0   # alternate
            _, _, done, _, info = env.step(actions)
            step += 1
            if step >= steps_per_episode:
                done = True
        total_wait_fixed += info['total_waiting_time']
        total_veh_fixed += info['total_vehicles_served']

        # MAS with trained agents (eval mode, no exploration)
        obs, _ = env.reset()
        total_wait_mas = 0
        total_veh_mas = 0
        done = False
        step = 0
        while not done:
            actions = {}
            for aid, agent in agents.items():
                state = obs[aid]
                actions[aid] = agent.act(state, eval_mode=True)
            _, _, done, _, info = env.step(actions)
            step += 1
            if step >= steps_per_episode:
                done = True
        total_wait_mas += info['total_waiting_time']
        total_veh_mas += info['total_vehicles_served']

        fixed_timing_metrics.append(total_wait_fixed / max(1, total_veh_fixed))
        mas_metrics.append(total_wait_mas / max(1, total_veh_mas))

    avg_fixed = np.mean(fixed_timing_metrics)
    avg_mas = np.mean(mas_metrics)
    print(f"\nEvaluation over {num_episodes} episodes:")
    print(f"Fixed Timing Avg Waiting Time: {avg_fixed:.2f} s")
    print(f"MAS (trained) Avg Waiting Time: {avg_mas:.2f} s")
    print(f"Improvement: {(avg_fixed - avg_mas)/avg_fixed*100:.1f}%")
    return avg_fixed, avg_mas

def plot_results(episode_rewards, avg_waiting_times):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    ax1.plot(episode_rewards)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Reward per Episode')
    ax2.plot(avg_waiting_times)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Waiting Time (s)')
    ax2.set_title('Average Waiting Time')
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

if __name__ == "__main__":
    # Train agents
    agents, rewards, waits = train(num_episodes=50, steps_per_episode=200, arrival_rate=0.5)
    plot_results(rewards, waits)

    # Evaluate against fixed timing
    evaluate(agents, num_episodes=10, steps_per_episode=200, arrival_rate=0.5)
