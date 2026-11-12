from collections import deque
from heuristics.heuristic_base import Heuristic

INF = 10**6


def get_parts(fact):
    """Remove surrounding parentheses and split a fact into its tokens."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (ud3uf11z).

    This heuristic estimates the number of actions required to achieve all
    goal predicates of the form (a7sb56om ?p) for persons/objects ?p of type
    nxbd1hc3. In the domain the agent (represented by a single true fact
    (omggbn9z ?location)) must:
      - move to a pickup location where (kut2ksnk ?p ?loc) holds and apply
        diwlmgg9 to set (f2xo7hb6 ?p) (this consumes the kut2ksnk fact),
      - move to a drop location where (klw11kjf ?p ?loc) holds and apply
        y7nik9q2 to convert (f2xo7hb6 ?p) into (a7sb56om ?p).
    Movement between location facts (t4rdxh32 typed objects) is via edges
    given by static acovodrf facts; the domain provides two move actions
    (one for each traversal direction) but the static relation indicates
    which directed edges exist.

    The heuristic returns 0 exactly when all goal facts are present in the
    state. For states that are solvable, it returns a finite integer estimate.

    Assumptions
    -----------
    - There is a single "agent location" fact of the form (omggbn9z <loc>)
      in the current state (the planner representation uses a single fact).
    - Pickup facts (kut2ksnk) may appear in the current state (they get
      deleted when pickup happens) while drop facts (klw11kjf) may appear
      either in the state or in the static facts. We look for both.
    - The static facts of the task contain acovodrf edges (directed) that
      define movement possibilities between locations (t4rdxh32 objects).
    - Each pickup action (diwlmgg9) and each drop action (y7nik9q2) counts
      as one action. Each movement along one acovodrf (one application of a
      move operator) counts as one action.
    - The heuristic is not required to be admissible; it trades optimality
      for informativeness while remaining efficiently computable.

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
    - self.goals: the goal facts required by the task (task.goals).
    - self.static_facts: the static facts provided by the task.
    - A directed adjacency map self.adj for location nodes based on static
      acovodrf facts. This adjacency is used to compute shortest path
      distances (in number of move actions) between locations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If all goal facts are already true in the current state, return 0.

    2. Identify all problem objects p that appear in the goals of the form
       (a7sb56om p). For each such p:
       - If (a7sb56om p) is already true in the state -> cost 0 for that p.
       - Else check whether (f2xo7hb6 p) (the "processed/picked" flag) is
         true in the state:
           * If yes -> we only need to go to p's drop location (the location
             x where (klw11kjf p x) holds) and apply the drop action
             (y7nik9q2). The local cost contribution is:
               move_cost (distance from agent/route to x) + 1 (drop action).
           * If no -> we need to perform pickup and then drop:
             - pickup at some location k where (kut2ksnk p k) holds: 1 action
               for diwlmgg9 (and we must be at k first),
             - then go to drop location x where (klw11kjf p x) holds and apply
               drop action y7nik9q2: 1 action.
             So local fixed action cost is 2 (pickup + drop) plus movement
             between locations.

    3. Collect the set of concrete location nodes that must be visited:
       - For unprocessed p: both pickup location k and drop location x (if
         present).
       - For processed p: drop location x.

       Add the current agent location to the set of nodes to be connected.

    4. Compute shortest-path distances (in number of move actions) between
       all nodes of interest using BFS on the directed adjacency graph
       built from acovodrf facts (the BFS cost equals number of move
       actions). If two nodes are not mutually reachable in either
       direction, treat the missing link as infinitely costly. To produce
       a symmetric, computationally convenient metric for the MST step we
       use the minimum of the two directed distances between a pair
       (u->v and v->u). This yields a lower-bound metric for connecting
       nodes in an undirected sense.

    5. Compute the Minimum Spanning Tree (MST) weight of the complete
       metric closure over the set {agent_location} ∪ required_locations.
       The MST weight is a lower bound on the number of move actions
       required to connect all required locations starting from the agent
       location (thus it captures an optimistic estimate of route length
       while permitting reuse of movement among multiple tasks).

    6. The final heuristic estimate equals:
          total_action_fixes + MST_move_cost
       where total_action_fixes is the number of diwlmgg9 and y7nik9q2
       actions still required (1 per pickup, 1 per drop), and MST_move_cost
       approximates the movement actions needed to visit the required
       locations.

    This heuristic balances local required operator counts with a compact
    graph-based routing lower bound; it is efficient to compute (BFS from
    each relevant node and a Prim-style MST over the small set of
    required nodes) and typically guides greedy best-first search to
    prioritize states that reduce both the number of remaining pickups/drops
    and the distance to their required locations.
    """

    def __init__(self, task):
        """Extract goal facts and static adjacency (acovodrf edges)."""
        super().__init__(task) if hasattr(super(), "__init__") else None
        self.goals = task.goals  # set/frozenset of goal fact strings
        # Save static facts for runtime lookup (some klw11kjf may be static).
        self.static_facts = set(task.static) if task.static is not None else set()

        # Build directed adjacency from static acovodrf facts.
        # adjacency: node -> set(neighbor_nodes) following direction acovodrf u v
        self.adj = {}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == "acovodrf" and len(parts) == 3:
                u, v = parts[1], parts[2]
                if u not in self.adj:
                    self.adj[u] = set()
                self.adj[u].add(v)
        # Ensure adjacency contains entries for nodes that appear as targets.
        for targets in list(self.adj.values()):
            for v in targets:
                if v not in self.adj:
                    self.adj.setdefault(v, set())

    def _bfs_distances(self, start):
        """
        Compute shortest distances (number of moves) from start to all other
        locations reachable via the directed adjacency. Returns a dict node->dist.
        If start is None or not present in adjacency keys, returns empty dict.
        """
        d = {}
        if start is None or start not in self.adj:
            return d
        queue = deque([start])
        d[start] = 0
        while queue:
            u = queue.popleft()
            for v in self.adj.get(u, ()):
                if v not in d:
                    d[v] = d[u] + 1
                    queue.append(v)
        return d

    def __call__(self, node):
        """Compute and return heuristic estimate (integer)."""
        state = node.state

        # 1) If all goals are satisfied in current state, h = 0.
        if self.goals <= state:
            return 0

        # Combine dynamic (state) and static facts for lookups of klw11kjf and kut2ksnk
        all_facts = set(state) | set(self.static_facts)

        # Find the agent's current location (the single omggbn9z fact in the state).
        agent_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "omggbn9z" and len(parts) == 2:
                agent_loc = parts[1]
                break

        # Build mappings for pickups (kut2ksnk) and drops (klw11kjf).
        pickup_loc = {}  # p -> location where kut2ksnk p loc holds (dynamic)
        drop_loc = {}  # p -> location where klw11kjf p loc holds (static or dynamic)
        processed = set()  # p with f2xo7hb6 p in current state
        # populate pickups and drops from all_facts
        for fact in all_facts:
            parts = get_parts(fact)
            if parts[0] == "kut2ksnk" and len(parts) == 3:
                p, loc = parts[1], parts[2]
                # there may be at most one pickup location per p in normal problems,
                # but if multiple exist we accept the first (deterministic).
                if p not in pickup_loc:
                    pickup_loc[p] = loc
            elif parts[0] == "klw11kjf" and len(parts) == 3:
                p, loc = parts[1], parts[2]
                if p not in drop_loc:
                    drop_loc[p] = loc

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "f2xo7hb6" and len(parts) == 2:
                processed.add(parts[1])

        # Extract the set of p objects that appear in the goals (a7sb56om p).
        goal_ps = []
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "a7sb56om" and len(parts) == 2:
                goal_ps.append(parts[1])

        # Build required locations to visit and count fixed operator actions (pickup/drop).
        required_locations = set()
        pickup_count = 0
        drop_count = 0

        for p in goal_ps:
            # If goal already true in state, skip p.
            if f"(a7sb56om {p})" in state:
                continue

            # If object already processed (picked up), only a drop remains.
            if p in processed:
                # Need to visit the drop location and perform y7nik9q2 once.
                loc = drop_loc.get(p)
                if loc is None:
                    # If we don't know a drop location, we cannot estimate a finite
                    # cost correctly. Return a large heuristic to avoid guiding
                    # search toward unsolvable assumptions.
                    return INF
                required_locations.add(loc)
                drop_count += 1
            else:
                # Need pickup and then drop.
                ploc = pickup_loc.get(p)
                dloc = drop_loc.get(p)
                if ploc is None or dloc is None:
                    # Missing required location facts: unsolvable or incomplete info.
                    # Return a large but finite heuristic.
                    return INF
                required_locations.add(ploc)
                required_locations.add(dloc)
                pickup_count += 1
                drop_count += 1

        # If there are no required locations (e.g., all goals already true), return 0
        if not required_locations:
            return 0

        # Add agent location to the nodes-to-connect if known.
        nodes = set(required_locations)
        if agent_loc is not None:
            nodes.add(agent_loc)

        # For each node compute distances to all other nodes using BFS on directed graph.
        # Note: the directed adjacency map self.adj was built from static facts.
        # If a node isn't present in adjacency keys, BFS returns empty and distances
        # to others are assumed INF unless reverse direction exists.
        dist_from = {}
        for n in nodes:
            dist_from[n] = self._bfs_distances(n)

        # Build symmetric distance metric between nodes as min(dist(u->v), dist(v->u)).
        # If both are missing, treat distance as INF.
        nodes_list = list(nodes)
        ncount = len(nodes_list)
        dist_matrix = {u: {} for u in nodes_list}
        for i, u in enumerate(nodes_list):
            for j, v in enumerate(nodes_list):
                if u == v:
                    dist_matrix[u][v] = 0
                    continue
                d_uv = dist_from[u].get(v, INF)
                d_vu = dist_from[v].get(u, INF)
                d_sym = min(d_uv, d_vu)
                dist_matrix[u][v] = d_sym

        # If any required location is totally disconnected (no finite distance to/from
        # any other node in the set), the task cannot be solved from this state;
        # return a large heuristic.
        for r in required_locations:
            # Check connectivity to any other node
            reachable = any(dist_matrix[r][other] < INF for other in nodes_list if other != r)
            if not reachable:
                return INF

        # Compute MST weight over the nodes set (Prim). This gives a lower bound
        # on the number of move actions required to connect them starting from the
        # agent location.
        # We'll run Prim on the complete graph with weights = dist_matrix[u][v].
        # Start Prim from agent_loc if present, else arbitrary node.
        start = agent_loc if agent_loc is not None else nodes_list[0]
        in_tree = set([start])
        # min edge to tree for each node not in tree
        min_edge = {}
        for v in nodes_list:
            if v == start:
                continue
            min_edge[v] = dist_matrix[start][v]

        mst_weight = 0
        while len(in_tree) < len(nodes_list):
            # pick node with smallest connecting edge
            v, best = None, INF
            for node_candidate, weight in min_edge.items():
                if node_candidate not in in_tree and weight < best:
                    best = weight
                    v = node_candidate
            if v is None or best >= INF:
                # disconnected
                return INF
            # add to MST
            mst_weight += best
            in_tree.add(v)
            # update min_edge values
            for w in nodes_list:
                if w not in in_tree:
                    if dist_matrix[v][w] < min_edge.get(w, INF):
                        min_edge[w] = dist_matrix[v][w]

        total_action_fixes = pickup_count + drop_count
        heuristic_value = total_action_fixes + mst_weight

        # Ensure finite integer
        return int(heuristic_value) if heuristic_value < INF else INF