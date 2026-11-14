from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the components of a PDDL fact, removing surrounding parentheses."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Check whether a fact (string) matches a pattern of tokens (wildcards '*' allowed).
    Example: match("(jft9ynda a b)", "jft9ynda", "*", "loc") -> True/False
    """
    parts = get_parts(fact)
    return all(fnmatch(p, pat) for p, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (mtfkspuf). The heuristic
    focuses on the jft9ynda predicate which (in the provided instances)
    represents an object's location. Most planning goals in this domain are of
    the form (jft9ynda <object> <location>), so the heuristic estimates the
    minimal number of single-edge moves (k12p0czo actions) required to bring
    each goal object from its current location to its goal location. The
    estimate is the sum of shortest-path distances in the static location graph
    defined by z6o4oitz edges.

    Assumptions
    -----------
    - Goals are mostly (or only) jft9ynda facts (object at location).
    - The static relation z6o4oitz defines connectivity between hmhptaxe
      (location) constants. Each k12p0czo operator moves an object along a
      single z6o4oitz edge; therefore shortest-path length equals the number
      of k12p0czo actions needed.
    - Other actions (sw2fg9ql, vyn92k55) may indirectly move objects or agents,
      but they are rarer and typically require additional preconditions. For
      efficiency and robustness the heuristic ignores those compound moves and
      focuses on the direct move operator k12p0czo for distance estimation.
    - If an object's current location is not explicit in the state, or a
      location is not present in the static graph, a finite penalty is used
      (see below) instead of infinite cost.

    Heuristic Initialization
    ------------------------
    - Extract all goal jft9ynda facts and store a mapping goal_object -> goal_location.
      Only jft9ynda goals are used by the heuristic; other goal predicates are
      still respected by returning heuristic 0 when the full goal set is reached.
    - Use the static facts (task.static) to build an adjacency graph of
      locations from z6o4oitz facts. The graph is treated as undirected
      (an explicit reverse edge is also added) for robustness.
    - Precompute the set of known locations (nodes in the static graph). This
      is used to bound unreachable-path penalties.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If all goals are already satisfied in the given state, return 0.
    2. Parse the current state to find the current location of every object
       mentioned in jft9ynda facts (current_location_map).
    3. For each goal of the form (jft9ynda obj goal_loc):
       a. If obj is present in current_location_map, let start = its location.
          - If start == goal_loc: distance = 0.
          - Else compute the shortest path length (number of edges) from start
            to goal_loc using BFS on the z6o4oitz adjacency graph.
       b. If BFS finds a path, distance = path length.
       c. If BFS can't reach goal_loc (or start/goal not in the graph),
          assign a finite penalty:
            penalty = max(1, number_of_locations + 1)
          This keeps heuristic finite while penalizing unreachable situations.
    4. Sum the distances (or penalties) for all jft9ynda goals and return the sum.
       This sum is a simple, efficiently-computable estimate of required moves.
    """

    def __init__(self, task):
        """
        Initialize the heuristic:
         - store the full goal set (for exact zero-check),
         - extract jft9ynda goals as object->goal_location mapping,
         - build adjacency graph of locations from static z6o4oitz facts.
        """
        self.goals = task.goals  # full goal set (frozenset of strings)
        static_facts = task.static

        # Extract jft9ynda goals: mapping object -> goal_location
        self.location_goals = {}
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "jft9ynda" and len(parts) >= 3:
                obj, loc = parts[1], parts[2]
                self.location_goals[obj] = loc

        # Build undirected adjacency graph of locations from z6o4oitz static facts.
        self.adj = {}  # loc -> set(neighbor_locs)
        self.locations = set()
        for f in static_facts:
            if match(f, "z6o4oitz", "*", "*"):
                _, a, b = get_parts(f)
                self.locations.add(a)
                self.locations.add(b)
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)  # treat as undirected for robustness

        # Define a finite penalty to use when a path cannot be found.
        # This is proportional to the number of known locations to remain finite
        # and to reflect that several moves may be required via other operators.
        self.unreachable_penalty = max(1, len(self.locations) + 1)

    def _bfs_distance(self, start, goal):
        """
        BFS over the adjacency graph to find shortest number of edges between
        start and goal. If either node is not in the graph or no path exists,
        return None.
        """
        if start == goal:
            return 0
        if start not in self.adj or goal not in self.adj:
            return None
        visited = {start}
        q = deque([(start, 0)])
        while q:
            node, dist = q.popleft()
            for nb in self.adj.get(node, ()):
                if nb == goal:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, dist + 1))
        return None

    def __call__(self, node):
        """
        Compute the heuristic value for the node.

        Returns the sum over all jft9ynda goals of the shortest-path lengths
        (number of k12p0czo moves) from the current location to the goal
        location. If a path cannot be found for a particular goal, a finite
        penalty is added instead.
        """
        state = node.state

        # If full goal set is satisfied, heuristic is zero.
        if self.goals <= state:
            return 0

        # Map: object -> current location from jft9ynda facts in the current state.
        current_location = {}
        for fact in state:
            if match(fact, "jft9ynda", "*", "*"):
                _, obj, loc = get_parts(fact)
                current_location[obj] = loc

        total = 0
        for obj, goal_loc in self.location_goals.items():
            start_loc = current_location.get(obj)
            if start_loc is None:
                # Object location not explicit in state: penalize but keep finite.
                total += self.unreachable_penalty
                continue

            # Compute BFS distance on location graph.
            dist = self._bfs_distance(start_loc, goal_loc)
            if dist is None:
                # No path known via z6o4oitz; assign finite penalty.
                total += self.unreachable_penalty
            else:
                total += dist

        return total