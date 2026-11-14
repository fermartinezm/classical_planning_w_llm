from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the parts of a PDDL fact string, e.g. "(p a b)" -> ["p", "a", "b"]."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (u2gh8ts5).
    The domain contains a directed relation `k21e3pdq` that connects z1mgqxkc nodes
    and a predicate `wuqjn5r5 <kukxix7x> <z1mgqxkc>` that denotes the current node of
    a movable object. Goals in instances are of the form (wuqjn5r5 <obj> <target_node>).
    This heuristic estimates the number of actions required to move each goal object
    from its current node to its goal node by computing shortest directed-path
    distances in the static `k21e3pdq` graph and summing these distances.

    Assumptions
    -----------
    - The relevant goal facts are only of the form (wuqjn5r5 <object> <node>).
    - Movement of a `wuqjn5r5` object along a single k21e3pdq edge can be achieved
      in (on average) one domain action (yaz3jy7e), so we count one action per
      traversed edge.
    - Auxiliary mechanics in operators (odqd6scj marker movement and ajcpzwcf flags)
      are abstracted away. The heuristic assumes that those mechanisms can be
      supported in roughly constant extra cost per move; to keep the heuristic
      simple and fast we do not explicitly model them.
    - If a goal object is not present in the state's wuqjn5r5 facts we treat it
      pessimistically (as unreachable) by returning a large but finite penalty
      based on the number of known nodes.

    Heuristic Initialization
    ------------------------
    - Extract the goal wuqjn5r5 targets from the task goals.
    - Parse the static facts to build a directed adjacency graph from `k21e3pdq`
      facts. Each `k21e3pdq A B L` is interpreted as a directed edge A -> B
      (label L is ignored for path distance estimation).
    - Record the set of all known z1mgqxkc nodes (from static facts) to compute
      safe finite fallback costs.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. If the current state already satisfies all goals, return 0.
    2. Build a mapping of current locations for all objects that appear in
       wuqjn5r5 facts in the current state (object -> node).
    3. For each goal fact (wuqjn5r5 object goal_node):
       a. If the object is already at goal_node in the state, its cost contribution is 0.
       b. Otherwise, compute the shortest directed-path distance (in number of edges)
          from the object's current node to the goal_node using the adjacency graph.
       c. If no path exists (disconnected), assign a finite large penalty equal to
          (number_of_nodes + 10) for that object to keep the heuristic finite.
    4. Sum the per-object distances (or penalties) and return that sum as the heuristic value.
    """

    def __init__(self, task):
        """
        Extract static information and goals from `task`.

        Parameters
        ----------
        task : Task
            The planning task object (provides .goals and .static).
        """
        # Goals as given by the task (frozenset of fact strings)
        self.goals = task.goals

        # Static facts (frozenset of fact strings)
        static_facts = task.static

        # Build a directed adjacency list representing the k21e3pdq graph.
        # We interpret facts of the form "(k21e3pdq from to label)" as an edge from -> to.
        self.adjacency = {}  # node -> set(neighbor_nodes)
        self.nodes = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "k21e3pdq" and len(parts) >= 3:
                src, dst = parts[1], parts[2]
                self.nodes.add(src)
                self.nodes.add(dst)
                if src not in self.adjacency:
                    self.adjacency[src] = set()
                self.adjacency[src].add(dst)

        # Ensure every known node has an entry in adjacency (possibly empty)
        for n in list(self.nodes):
            self.adjacency.setdefault(n, set())

        # Finite large penalty for unreachable goals (keeps heuristic finite).
        self.unreachable_penalty = max(len(self.nodes) + 10, 100)

    def _shortest_distance(self, start, target):
        """
        Compute shortest directed-path distance (number of edges) from start to target.
        Returns:
          - Non-negative integer distance if target reachable.
          - None if either node is unknown or no path exists.
        """
        if start == target:
            return 0
        if start not in self.adjacency:
            return None
        # BFS
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for neigh in self.adjacency.get(node, ()):
                if neigh == target:
                    return dist + 1
                if neigh not in visited:
                    visited.add(neigh)
                    queue.append((neigh, dist + 1))
        return None

    def __call__(self, node):
        """
        Compute heuristic value for the given search node.

        Parameters
        ----------
        node : search node
            Must have attribute `state` which is an iterable of PDDL fact strings.

        Returns
        -------
        int
            Estimated number of actions to reach a goal state from the given state.
        """
        state = node.state

        # 1) If goals are already satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # 2) Map current positions of objects that appear in wuqjn5r5 facts:
        #    (wuqjn5r5 <kukxix7x> <z1mgqxkc>)
        current_pos = {}  # object -> node
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "wuqjn5r5" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                current_pos[obj] = loc

        total_estimated = 0

        # 3) For every goal fact of interest, add shortest-path based estimate.
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts:
                continue
            pred = parts[0]
            if pred != "wuqjn5r5" or len(parts) < 3:
                # This heuristic only reasons about wuqjn5r5 goals; ignore other goal types.
                continue
            goal_obj, goal_node = parts[1], parts[2]

            # If goal is already true in the state, cost contribution is zero.
            if goal in state:
                continue

            # Find current node of the object.
            if goal_obj not in current_pos:
                # Object not present in state's wuqjn5r5 facts: treat as unreachable
                total_estimated += self.unreachable_penalty
                continue

            start_node = current_pos[goal_obj]

            # Compute shortest path in static graph from start_node to goal_node.
            dist = self._shortest_distance(start_node, goal_node)
            if dist is None:
                # unreachable according to static graph -> assign finite penalty
                total_estimated += self.unreachable_penalty
            else:
                # We assume one action per edge traversal (yaz3jy7e-like moves).
                total_estimated += dist

        # If we did not account for any wuqjn5r5 goals (e.g., goals of other type),
        # fall back to a small non-zero estimate unless goal already satisfied.
        if total_estimated == 0:
            # There are goals but none were wuqjn5r5 or all are currently satisfied.
            # If goals are not satisfied (we already checked), give a small heuristic.
            total_estimated = 1

        return int(total_estimated)