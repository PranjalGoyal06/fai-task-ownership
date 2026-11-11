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
        embedding_dim: int = 32,
        graph_type: str = "scale_free",  # "scale_free", "erdos_renyi", "complete", or "empty"
        avg_degree: int = 4,
        seed: int | None = 42,
    ):
        self.num_agents = num_agents
        self.embedding_dim = embedding_dim
        self.rng = np.random.default_rng(seed)

        # 1. Generate module profile vectors w_i (unit norm Gaussian; mostly orthogonal in dim >= 32)
        raw_profiles = self.rng.normal(0, 1.0, size=(num_agents, embedding_dim))
        norms = np.linalg.norm(raw_profiles, axis=1, keepdims=True)
        self.profiles = raw_profiles / np.maximum(norms, 1e-9)

        # 2. Generate dependency graph G with matched expected degree density
        if graph_type == "empty":
            self.G = nx.empty_graph(num_agents)
        elif graph_type == "complete":
            self.G = nx.complete_graph(num_agents)
        elif graph_type == "scale_free" and num_agents > 2:
            m_attach = min(max(1, avg_degree // 2), num_agents - 1)
            self.G = nx.barabasi_albert_graph(num_agents, m_attach, seed=seed)
        else:
            p = min(1.0, avg_degree / max(1, num_agents - 1))
            self.G = nx.erdos_renyi_graph(num_agents, p, seed=seed)

        self._precompute_distances()

    def _precompute_distances(self) -> None:
        """Precompute shortest path lengths between all reachable module pairs."""
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
        min_member_weight: float = 0.4,
        task_noise_sigma: float = 0.2,
        resolution_reward: float = 10.0,
        default_cost: float = 1.0,
        agent_costs: np.ndarray | None = None,
        rule: ResolutionRule = ResolutionRule.CONJUNCTIVE,
        seed: int | None = 42,
    ):
        self.env = env
        self.k = min(coalition_size_k, env.num_agents)
        self.cluster_prob = cluster_prob
        self.hop_decay = hop_decay
        self.min_member_weight = min_member_weight
        self.task_noise_sigma = task_noise_sigma
        self.R = resolution_reward
        self.default_cost = default_cost
        if agent_costs is not None:
            if len(agent_costs) != env.num_agents:
                raise ValueError("agent_costs length must match env.num_agents")
            self.agent_costs = np.asarray(agent_costs, dtype=float)
        else:
            self.agent_costs = np.full(env.num_agents, default_cost, dtype=float)
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
                # 1-hop neighbors of root_module
                candidates = list(self.env.get_neighbors(root_module))
                self.rng.shuffle(candidates)
                for cand in candidates:
                    if len(S_star) >= self.k:
                        break
                    S_star.add(cand)

                # Unbiased 2-hop expansion using a deduplicated set
                if len(S_star) < self.k:
                    two_hop_set: set[int] = set()
                    for s in S_star:
                        for nbr in self.env.get_neighbors(s):
                            if nbr not in S_star:
                                two_hop_set.add(nbr)
                    two_hop = sorted(two_hop_set)
                    self.rng.shuffle(two_hop)
                    for cand in two_hop:
                        if len(S_star) >= self.k:
                            break
                        S_star.add(cand)

            # Fallback to uniform random if unclustered or neighborhood exhausted
            if len(S_star) < self.k:
                remaining = sorted(set(range(N)) - S_star)
                self.rng.shuffle(remaining)
                needed = self.k - len(S_star)
                S_star.update(remaining[:needed])

        # 3. Indicator vector z
        z = np.zeros(N, dtype=int)
        for s in S_star:
            z[s] = 1

        # 4. Weighted issue embedding synthesis
        # Root module gets weight 1.0; co-owners decay by hop distance with positive floor
        t_raw = np.zeros(self.env.embedding_dim)
        for s in S_star:
            if s == root_module:
                weight = 1.0
            else:
                dist = self.env.distances[root_module, s]
                weight = max(self.hop_decay ** dist, self.min_member_weight)
            t_raw += weight * self.env.profiles[s]

        # Add Gaussian synthesis noise
        noise = self.rng.normal(0, self.task_noise_sigma, size=self.env.embedding_dim)
        t_vec = t_raw + noise
        t_norm = np.linalg.norm(t_vec)
        t_embedding = t_vec / max(t_norm, 1e-9)

        return Task(
            task_id=task_id,
            t_embedding=t_embedding,
            root_module=root_module,
            S_star=S_star,
            z=z,
            R=self.R,
            costs=self.agent_costs.copy(),
            rule=self.rule,
        )
