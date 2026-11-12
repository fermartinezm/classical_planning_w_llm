from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the parts of a PDDL fact string by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (u2gh8ts5).  The domain
    models movement of "agents" (predicate: wuqjn5r5 <agent> <node>), a
    movable token/resource (predicate: odqd6scj <node>), and a predicate
    ajcpzwcf <node> that marks nodes as "free/available".  The static
    predicate k21e3pdq encodes directed adjacency between z1mgqxkc nodes.
    Goals require certain agents to be located at certain nodes (wuqjn5r5 facts).

    The heuristic estimates the number of actions required by:
    - computing a shortest-path distance (in number of agent-steps) for each
      agent from its current location to its goal location using the directed
      adjacency graph extracted from k21e3pdq static facts;
    - adding an estimate for the cost of bringing a token close enough to the
      agent to enable the first agent-step (tokens are required by the domain
      actions to move an agent);
    - adding a small penalty for nodes along the path that are not currently
      marked as free (ajcpzwcf), since the domain's actions require targets to
      be free in order to move into them.

    The heuristic returns the sum over all goal agents of these estimated costs.
    It is not required to be admissible; the design deliberately biases toward
    a more informed (possibly optimistic or pessimistic) estimate that helps
    greedy best-first search focus on promising states.

    Assumptions
    -----------
    - Agent movement is along edges described by static k21e3pdq facts (we treat
      the first two arguments as a directed edge from the first node to the
      second node).
    - At least one token (odqd6scj) exists somewhere in the state in solvable
      problems. If no token is present, the heuristic will still produce a
      finite (large) estimate for unreachable situations.
    - Nodes with ajcpzwcf are "free" and are required as move targets. Moving
      into nodes currently not free will typically require additional actions;
      we model that as a small per-node penalty.
    - The domain's detailed interplay of token moves and agent moves is
      approximated by token distance and by the number of hops for the agent.
      This keeps the heuristic efficient to compute while capturing the main
      structure of the problem.

    Heuristic Initialization
    ------------------------
    During initialization the following static data is extracted from the task:
    - self.adjacency: directed adjacency map for z1mgqxkc nodes computed from
      static facts of the form (k21e3pdq A B LABEL). An edge A -> B is added.
    - self.goal_agent_map: mapping of agent (kukxix7x) to the goal node
      (z1mgqxkc) as given by wuqjn5r5 facts in the task goals.
    - self.all_nodes: the set of all nodes known from static facts so BFS knows
      the universe of vertices.

    Step-By-Step Thinking for Computing Heuristic (for a given state)
    -----------------------------------------------------------------
    1. If the state's facts already satisfy all goals, return 0.
    2. Extract current agent locations by scanning wuqjn5r5 facts in the state.
       If an agent mentioned in goals is not present in the state, treat that
       agent as having no known location (yielding a large cost).
    3. Gather token locations (all nodes that have odqd6scj in the state) and
       nodes that are currently free (ajcpzwcf in the state).
    4. For each agent with a goal:
       a. Compute the directed shortest path (BFS) from the agent's current
          node to its goal node using self.adjacency. If none exists, return a
          large cost (this goal is unreachable under the static graph).
       b. Reconstruct the path and count how many intermediate/target nodes on
          the path are currently not free (not in ajcpzwcf). Each such node
          gets a small penalty (1) to account for the extra actions required
          to make it free or to move resources around.
       c. Compute the minimal token distance: the shortest directed distance
          from any current token node to the agent's current node. This
          reflects the minimal number of token-move actions required to bring
          a token into range for the agent to start moving. If no tokens are
          present, a conservative large penalty is added to reflect the need
          to create/obtain a token (keeps heuristic finite but large).
       d. The estimated cost for this agent is:
            agent_hops (number of agent-step actions) +
            token_distance (moves to bring token near agent) +
            blocked_nodes_penalty
    5. Sum the estimates for all goal agents and return this integer estimate.

    Notes on efficiency
    -------------------
    - BFS is used for shortest-path computations; adjacency is stored as a
      dictionary of sets for O(1) neighbor access.
    - All parsing of facts is string-splitting; this is light-weight and
      consistent with planner internal representation.
    """

    def __init__(self, task):
        """Extract static adjacency and goal agent mapping from the task."""
        self.goals = task.goals
        self.static = task.static

        # Build directed adjacency map from static k21e3pdq facts:
        # (k21e3pdq src dst label) -> directed edge src -> dst
        self.adjacency = {}
        self.all_nodes = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "k21e3pdq" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.all_nodes.add(src)
                self.all_nodes.add(dst)
                self.adjacency.setdefault(src, set()).add(dst)

        # Extract goal mapping: agent -> target node for wuqjn5r5 goals
        self.goal_agent_map = {}
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "wuqjn5r5" and len(parts) >= 3:
                agent, node = parts[1], parts[2]
                self.goal_agent_map[agent] = node

        # A conservative large cost to indicate unreachable / heavy penalty.
        self.LARGE_COST = 1000

    def _bfs_distance_and_path(self, start, goal):
        """
        BFS on directed graph self.adjacency. Returns (distance, path_list)
        where path_list is [start, ..., goal]. If no path, returns (None, []).
        """
        if start == goal:
            return 0, [start]
        visited = set()
        queue = deque([(start, [start])])
        visited.add(start)
        while queue:
            node, path = queue.popleft()
            for nb in self.adjacency.get(node, ()):
                if nb in visited:
                    continue
                visited.add(nb)
                new_path = path + [nb]
                if nb == goal:
                    return len(new_path) - 1, new_path
                queue.append((nb, new_path))
        return None, []

    def _multi_source_bfs_min_dist(self, sources, target):
        """
        Compute the minimum directed distance from any source in 'sources' to 'target'.
        If sources is empty or no source can reach target, returns None.
        """
        if not sources:
            return None
        visited = set()
        queue = deque()
        # Multi-source: initialize all sources at distance 0
        for s in sources:
            queue.append((s, 0))
            visited.add(s)
        while queue:
            node, dist = queue.popleft()
            if node == target:
                return dist
            for nb in self.adjacency.get(node, ()):
                if nb in visited:
                    continue
                visited.add(nb)
                queue.append((nb, dist + 1))
        return None

    def __call__(self, node):
        """Compute the heuristic estimate for the given search node."""
        state = node.state

        # 1) If goals already satisfied -> heuristic is 0
        if self.goals <= state:
            return 0

        # 2) Extract current agent locations: wuqjn5r5 <agent> <node>
        agent_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "wuqjn5r5" and len(parts) >= 3:
                agent, loc = parts[1], parts[2]
                agent_locations[agent] = loc
                self.all_nodes.add(loc)

        # 3) Token nodes and free nodes
        token_nodes = [get_parts(f)[1] for f in state if get_parts(f)[0] == "odqd6scj"]
        free_nodes = {get_parts(f)[1] for f in state if get_parts(f)[0] == "ajcpzwcf"}

        total_estimate = 0

        # Precompute distances from tokens to all nodes using multi-source BFS per goal when needed.
        for agent, goal_node in self.goal_agent_map.items():
            # If agent already at goal, skip
            curr_loc = agent_locations.get(agent)
            if curr_loc is None:
                # Unknown location -> heavy penalty (agent must somehow be created/placed)
                total_estimate += self.LARGE_COST
                continue

            if curr_loc == goal_node:
                continue

            # Compute shortest directed path from current location to goal location
            dist, path = self._bfs_distance_and_path(curr_loc, goal_node)
            if dist is None:
                # No directed path -> unreachable under static graph, assign large cost
                total_estimate += self.LARGE_COST
                continue

            agent_hops = dist

            # Penalty for nodes along path that are not currently free (except the start node)
            # (Targets of movement generally require ajcpzwcf according to domain actions)
            blocked_penalty = 0
            # skip the starting node: we care about intermediate and target nodes
            for node_on_path in path[1:]:
                if node_on_path not in free_nodes:
                    blocked_penalty += 1

            # Compute minimal token distance to the agent's current location
            token_dist = self._multi_source_bfs_min_dist(token_nodes, curr_loc)
            if token_dist is None:
                # No token present or cannot reach: give a conservative penalty
                # (keeps the heuristic finite but large for problematic states)
                token_dist = max(1, min(self.LARGE_COST // 10, agent_hops + blocked_penalty + 1))

            # Sum estimate for this agent
            estimate_agent = agent_hops + token_dist + blocked_penalty
            total_estimate += estimate_agent

        return int(total_estimate)