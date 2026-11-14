from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract parts of a fact string like '(pred a b)' -> ['pred', 'a', 'b'].
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check whether fact matches pattern args (supports '*' wildcards).
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    This heuristic estimates the number of actions required to reach the goal
    for the secret_domain.  The main observation exploited is that many goals
    are of the form (jft9ynda SUBJECT LOCATION) and that there is an action
    (k12p0czo) which moves a SUBJECT along static adjacency facts
    (z6o4oitz LOC1 LOC2).  Therefore for a goal that requests a subject be at
    some location, the dominant cost is the graph distance (number of moves)
    between the subject's current location and the goal location.  The
    heuristic sums these estimated distances for all unsatisfied goals and
    applies small finite fallbacks for cases where a direct jft9ynda location
    is not currently present in the state.

    Assumptions
    -----------
    - Goals are mostly (jft9ynda SUBJECT LOCATION) facts (this matches the
      provided instances).
    - The action k12p0czo implements movement along edges described by
      z6o4oitz, so each adjacency hop can be achieved with one operator
      application.
    - If a SUBJECT currently has a jft9ynda fact, it represents its current
      location. If it does not, we assume a small fixed cost is required to
      establish a location for it (via other domain actions).
    - Static adjacency (z6o4oitz) facts define an undirected graph for movement.

    Heuristic Initialization
    ------------------------
    - Store the goal facts (task.goals) for quick checking.
    - Extract all static z6o4oitz facts into an adjacency dictionary mapping
      location -> set(neighbor locations).  The adjacency is treated as
      undirected because the static facts in examples appear in both
      directions; if only one direction exists, we still add the reverse edge
      to reflect movement capability along that connection.
    - No other static facts are used for this heuristic, but the constructor
      leaves room to expand the static analysis if needed.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given planner node/state:
    1. If all goal facts are already true in the state, return 0.
    2. For each goal fact:
       a. If the goal fact is already true in the state, it contributes 0.
       b. If the goal is of the form (jft9ynda SUBJECT GOAL_LOC):
          - Search the current state for any fact (jft9ynda SUBJECT CUR_LOC).
            * If such a CUR_LOC exists, compute the shortest path length (BFS)
              between CUR_LOC and GOAL_LOC in the adjacency graph built from
              z6o4oitz. The path length is the estimated cost (each hop is one
              k12p0czo move).
            * If a path exists, use its length as the cost contribution.
            * If no path exists (disconnected), assign a moderate finite
              penalty (e.g., 5) to keep the estimate finite but reflect
              difficulty.
          - If SUBJECT has no jft9ynda in the state (no known current
            location), assign a small finite fallback cost (e.g., 3) to
            represent the need to create/position the subject before moving
            it to the target (this covers use of other domain actions such as
            those that can create jft9ynda facts).
       c. For goals that are not of form jft9ynda (rare in the provided
          instances), if they are not true, assign a unit cost (1) as a
          lightweight finite estimate.
    3. Sum the contributions of all individual unsatisfied goals and return
       that sum as the heuristic value.

    Notes
    -----
    - The heuristic is intentionally optimistic for movement (each hop cost=1)
      and uses small constant fallbacks for cases where the exact sequence of
      actions is unclear. This keeps the heuristic finite and cheap to
      compute while providing useful guidance based on location distances.
    - The heuristic is not guaranteed admissible, which is acceptable for a
      greedy best-first search guiding function.
    """

    def __init__(self, task):
        """Extract goals and static adjacency (z6o4oitz) information."""
        self.goals = set(task.goals)
        static_facts = task.static

        # Build undirected adjacency graph for z6o4oitz facts.
        self.adjacency = {}
        for fact in static_facts:
            if match(fact, "z6o4oitz", "*", "*"):
                _, a, b = get_parts(fact)
                self.adjacency.setdefault(a, set()).add(b)
                self.adjacency.setdefault(b, set()).add(a)

    def _shortest_path_length(self, start, goal):
        """
        BFS on the adjacency graph to find the shortest number of hops from
        start to goal. Returns integer length or None if no path exists.
        """
        if start == goal:
            return 0
        if start not in self.adjacency or goal not in self.adjacency:
            return None
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for nb in self.adjacency.get(node, ()):
                if nb == goal:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return None

    def __call__(self, node):
        """
        Compute the heuristic for the given node/state.

        Returns an integer heuristic value estimating the number of actions to
        reach the goal from node.state.
        """
        state = node.state

        # 1) If goal reached, heuristic is 0.
        if self.goals <= state:
            return 0

        # Precompute current jft9ynda locations: map SUBJECT -> set(locations).
        current_locations = {}
        for fact in state:
            if match(fact, "jft9ynda", "*", "*"):
                _, subj, loc = get_parts(fact)
                current_locations.setdefault(subj, set()).add(loc)

        total_cost = 0

        for goal in self.goals:
            if goal in state:
                continue  # satisfied, contributes 0

            parts = get_parts(goal)
            pred = parts[0]
            args = parts[1:]

            # Handle primary target predicate form: (jft9ynda SUBJECT LOCATION)
            if pred == "jft9ynda" and len(args) >= 2:
                subj, goal_loc = args[0], args[1]

                # If we know current locations for the subject, take best (min) distance.
                if subj in current_locations:
                    best = None
                    for cur_loc in current_locations[subj]:
                        d = self._shortest_path_length(cur_loc, goal_loc)
                        if d is not None:
                            if best is None or d < best:
                                best = d
                    if best is not None:
                        total_cost += best
                    else:
                        # Locations exist but no path in static graph -> moderate penalty
                        total_cost += 5
                else:
                    # Subject has no known jft9ynda -> small finite fallback cost
                    total_cost += 3
            else:
                # Generic fallback for goals not matching jft9ynda pattern.
                total_cost += 1

        # Ensure returned heuristic is finite and integer.
        return int(total_cost)