from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into its components."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Test whether a fact matches a pattern. Patterns may use '*' as a wildcard
    for any token (same semantics as fnmatch).
    """
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (u2gh8ts5). The domain's
    goals in provided instances are of the form (wuqjn5r5 <kukx> <z-node>),
    i.e., they require certain kukxix7x objects to be associated with specific
    z1mgqxkc nodes. The domain provides static connectivity facts of the form
    (k21e3pdq <z1> <z2> <type>) which describe a directed relation between z1
    nodes (the fourth argument is a label that is constant-like). Two actions
    in the domain move the "association" or markers along such connectivity.
    This heuristic treats goal achievement as routing each kukxix7x object's
    current z-node to the desired goal z-node along the graph induced by the
    k21e3pdq facts, and uses shortest-path distances as an estimate of the
    number of required "move" operations.

    Assumptions
    -----------
    - Each goal is a wuqjn5r5 predicate (i.e., a mapping from a kukxix7x object
      to a z1mgqxkc object). The heuristic focuses on these predicates.
    - Moving a kukxix7x mapping from one z-node to an adjacent z-node can be
      done in roughly one "step" (one domain action that changes the mapping).
      This is an approximation: real actions have additional preconditions,
      but this gives a faithful distance-based guide to reachability.
    - The static predicate k21e3pdq defines directed adjacency between z-nodes.
      For safety the heuristic builds the adjacency from the static facts and
      performs directed BFS on that graph.
    - If the current mapping for a kukx is unknown in the state, the heuristic
      uses a reasonable fallback start node (a z-node with ajcpzwcf true if
      available, else any z-node seen in static facts). This ensures the
      heuristic remains finite.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic:
    - Stores the set of goal facts (task.goals).
    - Extracts goal requirements of the form (wuqjn5r5 <kukx> <znode>) into a
      mapping self.goal_map: kukx -> desired_znode.
    - Parses all static facts and collects directed adjacency (self.adj) from
      every k21e3pdq fact: for a static fact (k21e3pdq A B C) an edge A -> B
      is added to the adjacency structure. All z-nodes seen in these facts are
      stored in self.z_nodes for fallback selection.
    """

    def __init__(self, task):
        # Goals (set of strings)
        self.goals = task.goals

        # Extract goal mapping for wuqjn5r5 predicates:
        self.goal_map = {}  # kukx -> target_znode
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "wuqjn5r5" and len(parts) >= 3:
                kukx = parts[1]
                znode = parts[2]
                self.goal_map[kukx] = znode

        # Parse static information: adjacency from k21e3pdq facts.
        self.adj = {}  # directed adjacency: node -> set(neighbor nodes)
        self.z_nodes = set()
        for fact in task.static:
            if match(fact, "k21e3pdq", "*", "*", "*"):
                _, a, b, _ = get_parts(fact)
                self.z_nodes.add(a)
                self.z_nodes.add(b)
                if a not in self.adj:
                    self.adj[a] = set()
                self.adj[a].add(b)
        # Ensure adjacency has entries for all known z_nodes
        for z in list(self.z_nodes):
            self.adj.setdefault(z, set())

        # A fallback infinite distance for unreachable targets (finite but large).
        self._INF = 1000

    def __call__(self, node):
        """
        Step-By-Step Thinking for Computing Heuristic
        ---------------------------------------------
        1. If all goal facts are already true in the current state -> return 0.
        2. Parse the current state's wuqjn5r5 facts to determine where each
           kukxix7x object is currently mapped (kukx -> current_z).
        3. For each goal (kukx -> goal_z):
           a. If the current mapping equals goal_z, no cost is added.
           b. Otherwise, find a start z-node:
              - Prefer the current mapping of that kukx if present.
              - If absent, prefer any z-node that currently has ajcpzwcf true
                (interpreted as a likely useful starting position).
              - If none of the above, pick any z-node that appears in static
                k21e3pdq facts (ensures a finite heuristic).
           c. Run a directed BFS on the adjacency graph (built from static)
              to compute the shortest number of directed edges from start to
              goal. Each directed edge corresponds to an approximate single
              "move" action in the domain, so the distance is used as the
              estimated number of actions required.
           d. If the goal is unreachable in the static graph, add a large but
              finite penalty (self._INF) to signal difficulty.
        4. Sum the estimated costs for all unattained goal mappings and return
           the sum as the heuristic value.
        """
        state = node.state

        # 1) If all goals already satisfied, heuristic is 0.
        if self.goals <= state:
            return 0

        # 2) Extract current mapping of kukx objects from wuqjn5r5 facts.
        current_map = {}  # kukx -> current z-node
        for fact in state:
            if match(fact, "wuqjn5r5", "*", "*"):
                _, kukx, z = get_parts(fact)
                current_map[kukx] = z

        # Also gather current ajcpzwcf nodes for fallback starts.
        ajcp_nodes = {get_parts(f)[1] for f in state if match(f, "ajcpzwcf", "*")}

        total_cost = 0

        # For each required goal mapping, estimate cost to bring kukx to the goal z-node.
        for kukx, goal_z in self.goal_map.items():
            # If current mapping already meets the goal, no cost for this kukx.
            if any(match(f, "wuqjn5r5", kukx, goal_z) for f in state):
                continue

            # Select a start node: prefer current mapping, then ajcp nodes, then any known z-node.
            start = current_map.get(kukx, None)
            if start is None:
                if ajcp_nodes:
                    # pick an arbitrary ajcp node (deterministic by sorted order)
                    start = sorted(ajcp_nodes)[0]
                elif self.z_nodes:
                    start = sorted(self.z_nodes)[0]
                else:
                    # no information about nodes => return a conservative finite cost
                    total_cost += self._INF
                    continue

            # If start equals goal, but the specific wuqjn5r5 goal fact isn't present,
            # still require at least one action to establish the exact mapping in the
            # planner (but this is rare), so we treat distance 0 here (goal absent
            # due to representation mismatch). To be safe, continue to BFS below.
            if start == goal_z:
                # Double-check: if start==goal_z but mapping not in state, give small penalty 1
                total_cost += 1
                continue

            # BFS on directed adjacency graph to find shortest number of edges from start to goal_z.
            dist = self._bfs_distance(start, goal_z)
            total_cost += dist

        return total_cost

    def _bfs_distance(self, start, goal):
        """Return shortest directed edge distance from start to goal in self.adj or INF if unreachable."""
        if start == goal:
            return 0
        if start not in self.adj:
            return self._INF

        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, d = q.popleft()
            for nbr in self.adj.get(node, ()):
                if nbr == goal:
                    # Each directed edge approximated as one domain 'move' action.
                    return d + 1
                if nbr not in visited:
                    visited.add(nbr)
                    q.append((nbr, d + 1))
        return self._INF