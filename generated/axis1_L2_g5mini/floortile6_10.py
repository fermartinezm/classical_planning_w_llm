from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return list of tokens of a PDDL fact string, ignoring surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic for the secret_domain (hi4l5zmb). The domain's goals are facts of the
    form (cb6fw2xd <location> <resource>). Producing such a goal fact requires:
      - a location <location> to be "free" (predicate fguw2142 <location>),
      - an agent <agent> to hold the desired resource (qp2t6u4s <agent> <resource>),
      - the agent to be at a position <pos> that is related to <location> via
        either ga4l9do0 (<location> <pos>) or s4jrpk2b (<location> <pos>).
    The domain also contains several movement relations (vjcqgdgn, s4jrpk2b, vom4pfig,
    ga4l9do0) which can be used to move the agent through the map; movement
    preconditions require the target node to be "free", and movement swaps the
    free marker to the agent's previous location (sliding-like behavior).

    Assumptions
    -----------
    - The heuristic treats the map as an undirected graph composed of all relation
      facts in the static information (vjcqgdgn, s4jrpk2b, vom4pfig, ga4l9do0).
    - Any agent can be used to fulfill any goal; the heuristic picks the agent
      with minimal estimated cost for each goal.
    - Changing an agent's resource to the desired resource can be done in a
      single action (wvswfzmn) if that resource exists in the static facts
      (fezojtj3 <resource>). If the resource is already held by the agent,
      no change action is needed.
    - The free marker (fguw2142) can be "moved" across the graph; the cost to
      make a specific location free is approximated by the shortest-path
      distance from any currently free location to that target location.
    - Costs are counted in action-unit steps (each operator counts as 1).
      The final cb6fw2xd-producing action is counted as 1.

    Heuristic Initialization
    ------------------------
    The constructor extracts and stores:
      - the set of goal facts (self.goals),
      - an undirected adjacency graph combining static relation facts
        (vjcqgdgn, s4jrpk2b, vom4pfig, ga4l9do0),
      - for every location L, the set of positions P such that
        (ga4l9do0 L P) or (s4jrpk2b L P). These are the candidate agent
        positions from which the goal-creating operators can be applied,
        and are stored in self.goal_neighbors[L].
      - the set of available resources (from fezojtj3 facts). These are
        resources that agents can switch to in one step.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state the heuristic proceeds as follows:

    1. If all goals are already satisfied in the state, return 0.

    2. Parse the state to obtain:
       - current agent positions via ypt9mrgu <agent> <pos>,
       - current resources held by agents via qp2t6u4s <agent> <resource>,
       - all currently free locations via fguw2142 <loc>.

    3. For each goal (cb6fw2xd <loc> <res>) that is not yet true:
       a) Identify the candidate agent positions 'pos' from which the goal can
          be created. These are the positions P with a static fact
          ga4l9do0 <loc> P or s4jrpk2b <loc> P.
          If no such positions exist in the static facts, the goal is assumed
          unreachable and a large penalty cost is used.
       b) Estimate the cost to make <loc> free: compute the shortest-path
          distance from any currently free location to <loc> in the adjacency graph.
          (If <loc> is already free, this cost is 0.)
       c) For each agent, compute:
          - the shortest-path distance from the agent's current position to any
            candidate position P (this models moving the agent to the required
            neighbor position),
          - the cost to ensure the agent holds <res> (0 if already holding it,
            1 if the resource exists in the static resource set and must be changed,
            otherwise treat as impossible / large cost).
       d) Combine the three components:
          cost ≈ agent_move_dist + free_move_dist + resource_change_cost + 1
          (the final +1 is for the operator that produces the cb6fw2xd fact).
       e) Use the minimal such cost across all agents as the cost estimate for that goal.

    4. Sum the per-goal estimates to produce the final heuristic value. The
       heuristic is intentionally approximate (not admissible) but captures
       the primary structural requirements: moving agents, moving the free-marker,
       and changing an agent's held resource.

    Implementation notes
    --------------------
    - Distances are computed with BFS on the undirected adjacency graph.
    - If any required element is unreachable (no path, or resource unavailable),
      a large finite penalty is used so that the heuristic remains finite.
    """

    def __init__(self, task):
        # Goals and static facts
        self.goals = task.goals
        self.static = task.static

        # Build undirected adjacency graph from all relation types that represent
        # connectivity in the domain.
        relation_predicates = {"s4jrpk2b", "vjcqgdgn", "vom4pfig", "ga4l9do0"}
        self.adj = {}  # node -> set(neighbors)
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in relation_predicates:
                a, b = parts[1], parts[2]
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)

        # For goal creation we only accept candidate positions P for which
        # (ga4l9do0 <loc> P) or (s4jrpk2b <loc> P) exists in the static facts.
        self.goal_neighbors = {}  # loc -> set(candidate_positions)
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in ("ga4l9do0", "s4jrpk2b"):
                a, b = parts[1], parts[2]
                self.goal_neighbors.setdefault(a, set()).add(b)

        # Available resources that can be assigned to agents in one action (fezojtj3)
        self.available_resources = {
            get_parts(f)[1] for f in self.static if get_parts(f)[0] == "fezojtj3"
        }

    def bfs(self, start):
        """Return dict of distances from start to all reachable nodes in the adjacency graph."""
        d = {}
        if start not in self.adj:
            return d
        q = deque([start])
        d[start] = 0
        while q:
            u = q.popleft()
            for v in self.adj.get(u, ()):
                if v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        return d

    def __call__(self, node):
        state = node.state

        # 1) If all goals reached, heuristic is 0.
        if self.goals <= state:
            return 0

        # Parse state for agents' positions, agents' resources, and free locations.
        agents_pos = {}   # agent -> position
        agents_res = {}   # agent -> resource
        free_locations = set()
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "ypt9mrgu":
                # (ypt9mrgu ?en18p ?stq0j) => agent, position
                agents_pos[parts[1]] = parts[2]
            elif pred == "qp2t6u4s":
                # (qp2t6u4s ?en18p ?tlfqu) => agent, resource
                agents_res[parts[1]] = parts[2]
            elif pred == "fguw2142":
                # (fguw2142 ?loc)
                free_locations.add(parts[1])

        # Precompute BFS distances from all relevant start nodes (agents and free locations)
        bfs_cache = {}
        start_nodes = set(agents_pos.values()) | free_locations
        for start in start_nodes:
            bfs_cache[start] = self.bfs(start)

        INF = 10**6
        total_estimate = 0

        # Consider each goal that is not yet true
        for g in self.goals:
            if g in state:
                continue
            parts = get_parts(g)
            # Expect predicate cb6fw2xd <loc> <res>
            if parts[0] != "cb6fw2xd" or len(parts) < 3:
                # Unexpected goal format: penalize modestly
                total_estimate += 10
                continue
            loc = parts[1]
            res = parts[2]

            # Candidate neighbor positions from which the required operator can be applied.
            neighbors = self.goal_neighbors.get(loc, set())
            if not neighbors:
                # No static neighbor relation enabling goal creation: large penalty.
                total_estimate += 200
                continue

            # Distance to make loc free: shortest path from any current free location to loc.
            # If loc is already free, distance is 0.
            if loc in free_locations:
                min_free_dist = 0
            else:
                min_free_dist = INF
                for f in free_locations:
                    dist_dict = bfs_cache.get(f)
                    if dist_dict is None:
                        # BFS not computed (free location isolated from adjacency graph)
                        continue
                    d = dist_dict.get(loc)
                    if d is not None and d < min_free_dist:
                        min_free_dist = d
                if min_free_dist == INF:
                    # If no free location can reach loc in the adjacency graph, we still allow
                    # a fallback (treat as moderately expensive).
                    min_free_dist = 50

            # For each agent compute estimated cost, pick best.
            best_agent_cost = INF
            if not agents_pos:
                # No agents in the state (unexpected): use large penalty for this goal.
                total_estimate += 200
                continue

            for agent, pos in agents_pos.items():
                # Resource change cost: 0 if agent already holds desired resource,
                # 1 if resource is available and agent can change to it, INF otherwise.
                cur_res = agents_res.get(agent)
                if cur_res == res:
                    res_cost = 0
                elif res in self.available_resources:
                    # Changing resource requires one operator (wvswfzmn).
                    res_cost = 1
                else:
                    # Desired resource not available statically: treat costly/unavailable.
                    res_cost = INF

                # Distance from agent position to any candidate neighbor position.
                dist_agent_to_neighbor = INF
                agent_distances = bfs_cache.get(pos)
                if agent_distances is None:
                    agent_distances = self.bfs(pos)
                    bfs_cache[pos] = agent_distances
                for stq in neighbors:
                    d = agent_distances.get(stq)
                    if d is not None and d < dist_agent_to_neighbor:
                        dist_agent_to_neighbor = d
                if dist_agent_to_neighbor == INF:
                    # Agent cannot reach any neighbor (disconnected) -> skip this agent
                    continue

                # Combine components:
                # - moving the agent to neighbor: dist_agent_to_neighbor (each move is one action)
                # - moving the free marker to loc: min_free_dist (approx)
                # - changing the agent's resource if needed: res_cost
                # - the final cb6fw2xd-producing operator: +1
                agent_cost = dist_agent_to_neighbor + min_free_dist + res_cost + 1

                if agent_cost < best_agent_cost:
                    best_agent_cost = agent_cost

            if best_agent_cost == INF:
                # No agent could be used to produce this goal; large penalty
                total_estimate += 200
            else:
                total_estimate += best_agent_cost

        # Ensure finite non-negative integer
        if total_estimate >= INF:
            # fallback moderate finite value
            total_estimate = 10**5
        return int(total_estimate)