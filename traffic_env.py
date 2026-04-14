"""
traffic_env.py
Gymnasium-compatible traffic simulation environment using Mesa as the backend.
"""

import numpy as np
import networkx as nx
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from gymnasium import Env, spaces
from typing import Dict, List, Tuple, Optional
import json

# -------------------------------
# Vehicle Agent
# -------------------------------
class Vehicle(Agent):
    """Vehicle agent that moves along a predefined route."""
    def __init__(self, unique_id, model, route, start_time):
        super().__init__(unique_id, model)
        self.route = route          # list of node IDs
        self.current_node_index = 0
        self.current_node = route[0]
        self.start_time = start_time
        self.waiting_time = 0
        self.total_delay = 0

    def step(self):
        # Vehicle movement logic is handled by the model's step
        pass

# -------------------------------
# Traffic Light Agent
# -------------------------------
class TrafficLight(Agent):
    """Traffic light agent with DQN control (learning done outside)."""
    def __init__(self, unique_id, model, phases=['NS_GREEN', 'EW_GREEN'], phase_duration=3):
        super().__init__(unique_id, model)
        self.phases = phases
        self.current_phase = phases[0]
        self.phase_duration = phase_duration
        self.counter = 0
        self.queue_lengths = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
        self.neighbors = []   # list of neighbor TrafficLight IDs

    def get_state(self):
        """Return observation: queue lengths on all approaches + current phase."""
        queues = [self.queue_lengths[d] for d in ['N', 'S', 'E', 'W']]
        phase_onehot = [1 if self.current_phase == p else 0 for p in self.phases]
        return np.array(queues + phase_onehot, dtype=np.float32)

    def step(self):
        self.counter += 1
        if self.counter >= self.phase_duration:
            # Change phase
            idx = self.phases.index(self.current_phase)
            self.current_phase = self.phases[(idx + 1) % len(self.phases)]
            self.counter = 0

    def send_message(self):
        """Return congestion info to share with neighbors."""
        total_queue = sum(self.queue_lengths.values())
        return {"from": self.unique_id, "queue_length": total_queue, "phase": self.current_phase}

    def receive_message(self, msg):
        """Store neighbor's queue info (used in state extension)."""
        # In this implementation, the model will aggregate messages for all agents
        pass

# -------------------------------
# Traffic Model (Mesa + Gym)
# -------------------------------
class TrafficModel(Model):
    """Mesa model that manages vehicles and traffic lights."""
    def __init__(self, graph: nx.Graph, traffic_lights: Dict[int, List[str]],
                 arrival_rate: float = 0.5, seed=None):
        super().__init__(seed=seed)
        self.graph = graph
        self.arrival_rate = arrival_rate   # Poisson lambda (vehicles per step)
        self.schedule = RandomActivation(self)
        self.grid = NetworkGrid(graph)

        # Create traffic light agents
        self.traffic_lights = {}
        for node_id, phases in traffic_lights.items():
            tl = TrafficLight(node_id, self, phases)
            self.schedule.add(tl)
            self.traffic_lights[node_id] = tl
            self.grid.place_agent(tl, node_id)

        # Find neighbors for each traffic light (adjacent intersections)
        for node_id, tl in self.traffic_lights.items():
            tl.neighbors = [n for n in graph.neighbors(node_id) if n in self.traffic_lights]

        # Vehicle queue per edge (simplified: store vehicles waiting at each node's approaches)
        self.vehicle_queues = {node: {'N': [], 'S': [], 'E': [], 'W': []} for node in graph.nodes}
        self.vehicle_counter = 0
        self.step_count = 0

        # Metrics
        self.total_waiting_time = 0
        self.total_vehicles_served = 0
        self.vehicle_log = []   # (arrival_time, departure_time, origin, destination)

    def get_direction_from_edge(self, u, v):
        """Return compass direction of edge (u->v) relative to node u."""
        # Simplified: use vector difference
        pos_u = self.graph.nodes[u].get('pos', (0,0))
        pos_v = self.graph.nodes[v].get('pos', (0,0))
        dx = pos_v[0] - pos_u[0]
        dy = pos_v[1] - pos_u[1]
        if abs(dx) > abs(dy):
            return 'E' if dx > 0 else 'W'
        else:
            return 'N' if dy > 0 else 'S'

    def spawn_vehicle(self):
        """Generate a new vehicle at a random node with Poisson probability."""
        if np.random.rand() < self.arrival_rate:
            # Choose random origin and destination (must be different)
            nodes = list(self.graph.nodes)
            origin = np.random.choice(nodes)
            destinations = [n for n in nodes if n != origin]
            if not destinations:
                return
            dest = np.random.choice(destinations)
            # Compute shortest path (simplified)
            try:
                route = nx.shortest_path(self.graph, origin, dest)
            except nx.NetworkXNoPath:
                return
            veh = Vehicle(self.vehicle_counter, self, route, self.step_count)
            self.vehicle_counter += 1
            # Place vehicle at origin node
            self.grid.place_agent(veh, origin)
            self.schedule.add(veh)
            # Add to appropriate queue at origin
            # Determine direction of first edge
            if len(route) > 1:
                dir = self.get_direction_from_edge(route[0], route[1])
                self.vehicle_queues[origin][dir].append(veh)
            else:
                # Already at destination? remove immediately
                pass

    def step(self):
        self.step_count += 1

        # 1. Spawn vehicles
        self.spawn_vehicle()

        # 2. Update traffic lights (phase changes)
        for tl in self.traffic_lights.values():
            tl.step()

        # 3. Move vehicles through intersections based on current phase
        for node_id, tl in self.traffic_lights.items():
            # Determine which directions have green
            green_dirs = []
            if tl.current_phase == 'NS_GREEN':
                green_dirs = ['N', 'S']
            elif tl.current_phase == 'EW_GREEN':
                green_dirs = ['E', 'W']

            # For each green direction, release one vehicle (if any)
            for dir in green_dirs:
                queue = self.vehicle_queues[node_id][dir]
                if queue:
                    veh = queue.pop(0)
                    # Move vehicle to next node in its route
                    idx = veh.route.index(node_id)
                    if idx + 1 < len(veh.route):
                        next_node = veh.route[idx+1]
                        self.grid.move_agent(veh, next_node)
                        # Record departure time for waiting time
                        delay = self.step_count - veh.start_time
                        self.total_waiting_time += delay
                        self.total_vehicles_served += 1
                        self.vehicle_log.append((veh.start_time, self.step_count, veh.route[0], veh.route[-1]))
                        # If vehicle not at destination, add to queue of next node
                        if idx+2 < len(veh.route):
                            next_dir = self.get_direction_from_edge(next_node, veh.route[idx+2])
                            self.vehicle_queues[next_node][next_dir].append(veh)
                        # else vehicle removed (reached destination)
                    else:
                        # Destination reached
                        delay = self.step_count - veh.start_time
                        self.total_waiting_time += delay
                        self.total_vehicles_served += 1
                        self.vehicle_log.append((veh.start_time, self.step_count, veh.route[0], veh.route[-1]))
                        self.schedule.remove(veh)

        # 4. Update queue length observations for traffic lights
        for node_id, tl in self.traffic_lights.items():
            queues = self.vehicle_queues[node_id]
            tl.queue_lengths = {k: len(v) for k, v in queues.items()}

    def get_agent_states(self):
        """Return states for all traffic light agents."""
        return {aid: tl.get_state() for aid, tl in self.traffic_lights.items()}

    def get_rewards(self):
        """Reward = - (total waiting time increment + queue length sum)."""
        rewards = {}
        for node_id, tl in self.traffic_lights.items():
            total_queue = sum(tl.queue_lengths.values())
            # Simplified: negative queue length as immediate reward
            rewards[node_id] = -total_queue
        return rewards

    def get_communication_messages(self):
        """Aggregate messages from all traffic lights."""
        messages = {}
        for node_id, tl in self.traffic_lights.items():
            messages[node_id] = tl.send_message()
        return messages

# -------------------------------
# Gymnasium Environment Wrapper
# -------------------------------
class TrafficGymEnv(Env):
    """Gymnasium environment for multi-agent traffic control."""
    def __init__(self, graph: nx.Graph, traffic_lights: Dict[int, List[str]],
                 arrival_rate=0.5, max_steps=1000):
        super().__init__()
        self.model = TrafficModel(graph, traffic_lights, arrival_rate)
        self.max_steps = max_steps
        self.current_step = 0

        # Observation and action spaces for each agent (we'll use a dict)
        self.num_agents = len(traffic_lights)
        self.agent_ids = list(traffic_lights.keys())
        # Example state: 4 queue lengths + 2 phase one-hot = 6 dims
        obs_dim = 6
        self.observation_space = spaces.Dict({
            aid: spaces.Box(low=0, high=100, shape=(obs_dim,), dtype=np.float32)
            for aid in self.agent_ids
        })
        # Action: 0 = stay in current phase, 1 = switch to next phase
        self.action_space = spaces.Dict({
            aid: spaces.Discrete(2) for aid in self.agent_ids
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.model = TrafficModel(self.model.graph, {aid: tl.phases for aid, tl in self.model.traffic_lights.items()},
                                  self.model.arrival_rate, seed=seed)
        self.current_step = 0
        obs = self.model.get_agent_states()
        return obs, {}

    def step(self, actions: Dict[int, int]):
        # Apply actions: 1 = force phase change (override counter)
        for aid, act in actions.items():
            tl = self.model.traffic_lights[aid]
            if act == 1:  # switch phase
                idx = tl.phases.index(tl.current_phase)
                tl.current_phase = tl.phases[(idx+1) % len(tl.phases)]
                tl.counter = 0

        # Advance model one step
        self.model.step()
        self.current_step += 1

        # Get observations, rewards, done
        obs = self.model.get_agent_states()
        rewards = self.model.get_rewards()
        done = self.current_step >= self.max_steps
        truncated = False
        info = {
            'total_waiting_time': self.model.total_waiting_time,
            'total_vehicles_served': self.model.total_vehicles_served,
            'messages': self.model.get_communication_messages()
        }
        return obs, rewards, done, truncated, info

    def render(self):
        pass

# -------------------------------
# Example: create a simple grid network
# -------------------------------
def create_grid_network(rows=2, cols=2):
    """Create a grid road network with positions."""
    G = nx.grid_2d_graph(rows, cols)
    pos = {node: (node[1], -node[0]) for node in G.nodes()}  # (x,y)
    nx.set_node_attributes(G, pos, 'pos')
    # Rename nodes to integers for simplicity
    G = nx.convert_node_labels_to_integers(G)
    return G

if __name__ == "__main__":
    # Quick test
    G = create_grid_network(2,2)
    # Define traffic lights at all nodes with two phases
    traffic_lights = {node: ['NS_GREEN', 'EW_GREEN'] for node in G.nodes}
    env = TrafficGymEnv(G, traffic_lights, arrival_rate=0.3, max_steps=50)
    obs, _ = env.reset()
    for _ in range(10):
        actions = {aid: np.random.randint(2) for aid in env.agent_ids}
        obs, rew, done, trunc, info = env.step(actions)
        print(f"Step: {_}, Rewards: {rew}")
    print("Simulation test completed.")
