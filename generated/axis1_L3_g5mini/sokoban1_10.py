from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact into tokens."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Match a fact against a pattern with wildcards.
    Example: match(fact, "served", "*", "*", "floor1")
    """
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the 'miconic' (secret_domain) elevator-style domain.
    The heuristic estimates the number of actions required to make each required
    origin mapping (predicate: (origin <passenger-2> <passenger>)) reach its respective
    goal passenger by moving the origin along the directed "served" chains.

    Assumptions
    -----------
    - The static predicate `served X Y F` defines a directed adjacency edge X -> Y.
      Sequences of these `served` facts define chains along which an `origin` can be
      advanced using the domain's up/depart operators.
    - Each goal is of the form (origin <p2> <target_passenger>). We assume these goal
      origin facts appear in task.goals.
    - The initial state contains an (origin <p2> <current_passenger>) fact for each
      passenger-2 that appears in the goals (this holds for standard instances).
    - There is typically at least one boarded token in the state which the domain
      actions move around; the heuristic only models movement along the served graph,
      and therefore does not attempt to simulate detailed boarded/lift token flows.
    - The heuristic is allowed to be inadmissible; it attempts to produce a useful
      guidance estimate for greedy best-first search (favoring compact, informative
      numbers over guaranteed underestimation).

    Heuristic Initialization
    ------------------------
    In the constructor we extract:
    - self.goal_origins: a mapping from passenger-2 objects to their target passenger
      (extracted from task.goals).
    - self.adjacency: a directed graph (dict of sets) representing edges u -> v for
      every static fact (served u v floor).
    - self.nodes: the set of passengers that appear in the served static facts (used
      for bounding unreachable distances).

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given node/state:
    1. If the state's facts already contain all goal facts (i.e., task.goals subset
       of state), return 0.
    2. For each required origin goal (for a passenger-2 object p2 with goal passenger G):
       a. Find the current origin fact in the state of the form "(origin p2 CUR)".
          - If no origin fact for p2 exists, treat this as a difficult-to-achieve
            situation and assign a sizable finite penalty for that goal (because the
            domain's only operator that modifies origin is `up`, which presupposes an
            existing origin; such instances are generally ill-formed or unsolvable).
       b. Otherwise compute the shortest path distance (number of directed edges)
          from CUR to G in the adjacency graph built from static `served` facts
          using BFS.
       c. Each edge in that path corresponds to advancing the origin by one step.
          Empirically, advancing an origin by one chain step typically requires
          at least one `up` and may require preparatory `depart`/token movement
          — to reflect that, the heuristic multiplies the number of steps by a
          small per-step cost factor (configurable below). This produces a more
          informative (and intentionally slightly optimistic/pessimistic tradeoff)
          estimate for guiding greedy search.
       d. If G is unreachable from CUR in the static graph, assign a finite large
          penalty for that goal proportional to the graph size.
    3. Sum the per-goal estimates to form the final heuristic value. The value is
       returned as an integer.

    Notes on parameter choices
    - per_step_cost: chosen to be 2 to reflect that progress along the served chain
      typically requires more than a single atomic action (this helps steer greedy
      search more decisively toward nodes that reduce chain distance).
    - unreachable_penalty: finite and larger than any plausible solvable path so that
      unsolvable/isolated goals get prioritized as costly but still finite.
    """

    def __init__(self, task):
        # Extract goal origin mappings from task.goals, e.g., "(origin p2 p)".
        self.goal_origins = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "origin" and len(parts) >= 3:
                p2 = parts[1]
                target = parts[2]
                self.goal_origins[p2] = target

        # Build adjacency graph from static served facts: served A B floor => edge A -> B
        self.adjacency = {}
        nodes = set()
        for fact in task.static:
            if match(fact, "served", "*", "*", "*"):
                _, a, b, _ = get_parts(fact)
                nodes.add(a)
                nodes.add(b)
                if a not in self.adjacency:
                    self.adjacency[a] = set()
                self.adjacency[a].add(b)
        self.nodes = nodes

        # Tunable parameters for the heuristic calculation
        self.per_step_cost = 2  # cost assigned per traversed served-edge
        # If unreachable, penalize by a value proportional to the domain size (finite).
        self.unreachable_penalty = max(10, len(self.nodes) * self.per_step_cost + 5)

    def _shortest_distance(self, start, goal):
        """
        Compute the shortest number of directed edges from start to goal in the
        adjacency graph using BFS. Returns integer distance or None if unreachable.
        """
        if start == goal:
            return 0
        if start not in self.nodes or goal not in self.nodes:
            return None
        visited = set([start])
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for succ in self.adjacency.get(node, ()):
                if succ == goal:
                    return dist + 1
                if succ not in visited:
                    visited.add(succ)
                    queue.append((succ, dist + 1))
        return None

    def __call__(self, node):
        """
        Compute the heuristic value for the given search node.

        The heuristic sums, for each goal origin mapping, the estimated cost to
        advance the current origin passenger to the goal passenger along the
        static `served` graph. If the goal is already satisfied return 0.
        """
        state = node.state

        # 2) Return 0 for goal states.
        if self.goal_origins and all(f"(origin {p2} {target})" in state for p2, target in self.goal_origins.items()):
            return 0

        total_cost = 0

        # Precompute origin facts present in the current state: map p2 -> current passenger
        current_origins = {}
        for fact in state:
            if match(fact, "origin", "*", "*"):
                _, p2, curr = get_parts(fact)
                current_origins[p2] = curr

        # If there are no adjacency nodes known (no served facts), fallback:
        if not self.nodes:
            # Finite fallback: if not already at goal, return number of origin goals * small cost
            return len(self.goal_origins) * self.unreachable_penalty

        for p2, target in self.goal_origins.items():
            if f"(origin {p2} {target})" in state:
                # This particular goal already satisfied.
                continue

            if p2 not in current_origins:
                # No origin present for this passenger-2; assign a significant finite penalty.
                total_cost += self.unreachable_penalty
                continue

            curr = current_origins[p2]
            dist = self._shortest_distance(curr, target)
            if dist is None:
                total_cost += self.unreachable_penalty
            else:
                total_cost += dist * self.per_step_cost

        # Ensure returned value is an int and finite.
        return int(total_cost)