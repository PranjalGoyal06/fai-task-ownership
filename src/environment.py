"""Repository topology, graph dependencies, and task generation."""

from __future__ import annotations

import networkx as nx
import numpy as np

from src.types import ResolutionRule, Task


class RepositoryEnvironment:
    """Manages the software repository graph and module profiles."""

    def __init__(
        self,
        num_agents: int = 10,
        embedding_dim: int = 16,
        graph_type: str = "scale_free",  # "scale_free" or "erdos_renyi"
        avg_degree: int = 2,
        seed: int | None = 42,
    ):
        self.num_agents = num_agents
        self.embedding_dim = embedding_dim
        self.rng = np.random.default_rng(seed)

        # 1. Generate module profile vectors w_i (unit norm)
        raw_profiles = self.rng.normal(0, 1.0, size=(num_agents, embedding_dim))
        norms = np.linalg.norm(raw_profiles, axis=1, keepdims=True)
        self.profiles = raw_profiles / np.maximum(norms, 1e-9)

        # 2. Generate dependency graph G
        if graph_type == "scale_free" and num_agents > avg_degree:
            self.G = nx.barabasi_albert_graph(num_agents, avg_degree, seed=seed)
        else:
            p = min(1.0, (2.0 * avg_degree) / max(1, num_agents - 1))
            self.G = nx.erdos_renyi_graph(num_agents, p, seed=seed)

        # Ensure G is undirected and has at least self-loops or minimal edges
        self.adj_matrix = nx.to_numpy_array(self.G)
        self._precompute_distances()

    def _precompute_distances(self) -> None:
        """Precompute shortest path lengths between all nodes."""
        self.distances = np.full((self.num_agents, self.num_agents), 999, dtype=int)
        np.fill_diagonal(self.distances, 0)
        for i, lengths in nx.all_pairs_shortest_path_length(self.G):
            for j, length in lengths.items():
                self.distances[i, j] = length

    def get_neighbors(self, agent_id: int) -> list[int]:
        """Returns neighbors of agent_id in graph G."""
        return list(self.G.neighbors(agent_id))

    def get_degree(self, agent_id: int) -> int:
        """Returns degree m_i = |N(i)|."""
        return self.G.degree[agent_id]


class TaskGenerator:
    """Generates incoming issues/tasks with root-cause weighted embeddings."""

    def __init__(
        self,
        env: RepositoryEnvironment,
        coalition_size_k: int = 2,
        cluster_prob: float = 0.7,
        hop_decay: float = 0.5,
        task_noise_sigma: float = 0.2,
        resolution_reward: float = 10.0,
        default_cost: float = 1.0,
        rule: ResolutionRule = ResolutionRule.CONJUNCTIVE,
        seed: int | None = 42,
    ):
        self.env = env
        self.k = min(coalition_size_k, env.num_agents)
        self.cluster_prob = cluster_prob
        self.hop_decay = hop_decay
        self.task_noise_sigma = task_noise_sigma
        self.R = resolution_reward
        self.default_cost = default_cost
        self.rule = rule
        self.rng = np.random.default_rng(seed)

    def generate_task(self, task_id: int) -> Task:
        """Synthesizes one task/issue instance."""
        N = self.env.num_agents

        # 1. Pick root-cause seed module s_0
        root_module = int(self.rng.integers(0, N))
        S_star = {root_module}

        # 2. Expand S_star (cluster-based along G vs random)
        if self.k > 1:
            use_cluster = self.rng.random() < self.cluster_prob
            if use_cluster:
                # Explore neighbors of root_module and connected components
                candidates = list(self.env.get_neighbors(root_module))
                self.rng.shuffle(candidates)
                for cand in candidates:
                    if len(S_star) >= self.k:
                        break
                    S_star.add(cand)

                # If still need more, explore 2-hop neighbors
                if len(S_star) < self.k:
                    two_hop = []
                    for s in list(S_star):
                        for nbr in self.env.get_neighbors(s):
                            if nbr not in S_star:
                                two_hop.append(nbr)
                    self.rng.shuffle(two_hop)
                    for cand in two_hop:
                        if len(S_star) >= self.k:
                            break
                        S_star.add(cand)

            # Fallback to random if still under k
            if len(S_star) < self.k:
                remaining = list(set(range(N)) - S_star)
                self.rng.shuffle(remaining)
                needed = self.k - len(S_star)
                S_star.update(remaining[:needed])

        # 3. Indicator vector z
        z = np.zeros(N, dtype=int)
        for s in S_star:
            z[s] = 1

        # 4. Weighted issue embedding synthesis
        # Root module has weight 1.0; other members decay by graph distance
        t_raw = np.zeros(self.env.embedding_dim)
        for s in S_star:
            dist = self.env.distances[root_module, s]
            weight = 1.0 if s == root_module else (self.hop_decay ** dist)
            t_raw += weight * self.env.profiles[s]

        # Add Gaussian synthesis noise
        noise = self.rng.normal(0, self.task_noise_sigma, size=self.env.embedding_dim)
        t_vec = t_raw + noise
        t_norm = np.linalg.norm(t_vec)
        t_embedding = t_vec / max(t_norm, 1e-9)

        # 5. Costs vector
        costs = np.full(N, self.default_cost, dtype=float)

        return Task(
            task_id=task_id,
            t_embedding=t_embedding,
            root_module=root_module,
            S_star=S_star,
            z=z,
            R=self.R,
            costs=costs,
            rule=self.rule,
        )
