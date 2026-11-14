from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the components of a PDDL fact as a list of strings (without surrounding parentheses)."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Check whether `fact` matches the given pattern. Patterns may contain '*' wildcards.
    Example: match("(served p1 p2 floor1)", "served", "*", "*", "*") -> True
    """
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the miconic-like "secret_domain".
    The domain's goals are origin facts of the form (origin <passenger-2> <passenger>).
    Actions in the domain propagate the origin pointer along chains of passengers
    connected by served facts; roughly speaking, an "up" action advances an origin
    one step forward along a served-edge. This heuristic estimates how many such
    advances (plus a small overhead) are needed to map each goal's passenger-2
    to its required passenger.

    Assumptions
    -----------
    - The served predicate defines directed adjacency between passengers:
      (served A B floorX) is interpreted as an edge A -> B.
    - Each "up" operation advances the origin by one edge along the served graph.
      Therefore the minimal number of required origin-advances is the length of
      a shortest directed path from the current origin passenger to the goal passenger.
    - Additional necessary actions (to satisfy lift-at or to reposition boarded flags)
      are approximated by small per-step penalties. The heuristic is not required
      to be admissible and uses these penalties to better guide greedy search.
    - If no directed path exists (unreachable by purely following served edges),
      the heuristic returns a large but finite estimate.

    Heuristic Initialization
    ------------------------
    In the constructor we extract and store:
    - self.goals: the set/list of origin goal facts.
    - self.served_adj: a directed adjacency mapping passenger -> list of successor passengers
      inferred from all static (served ...) facts in the task's static facts.
    - self.all_passengers: the set of passenger names that appear in the served facts
      (used for bounds when a path cannot be found).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node/state the heuristic is computed as follows:
    1. If the node state already satisfies all goal origin facts, return 0.
    2. Parse the current state's origin facts to obtain, for every passenger-2 object,
       the passenger to which it currently points.
    3. For each required goal (origin P2 -> targetPassenger):
       a. Look up the current passenger curPassenger for P2 in the state.
       b. Run a BFS on the directed graph defined by served-adjacency to find the
          shortest directed path length d from curPassenger to targetPassenger.
       c. If a path exists, add a step cost proportional to d. To make the heuristic
          more informative for search (and because actions other than the pure "up"
          may be required to enable an up), we weight each step with a small factor
          (cost_per_step). Additionally, for every intermediate step we add a small
          penalty when the successor node currently does not have a (lift-at ...) fact,
          since "up" requires lift-at at the successor in the current domain modeling.
       d. If no path exists, add a large finite penalty (unreachable_penalty) so that
          the heuristic remains finite while indicating a high estimated effort.
    4. The heuristic value is the sum across all origin goals. The heuristic is
       intentionally optimistic in parts (not admissible) but aims to be informative
       and cheap to compute (BFS on the static served graph).
    """

    def __init__(self, task):
        """
        Extract static served relations and goal origin facts.

        - task.goals: set/list of PDDL strings representing goal conditions.
        - task.static: set of static facts (frozenset of strings).
        """
        self.goals = task.goals
        static_facts = task.static

        # Build directed adjacency from served facts: A -> B when (served A B floorX)
        self.served_adj = {}  # passenger -> set(successor passengers)
        self.all_passengers = set()
        for fact in static_facts:
            if match(fact, "served", "*", "*", "*"):
                _, a, b, _ = get_parts(fact)
                self.all_passengers.add(a)
                self.all_passengers.add(b)
                self.served_adj.setdefault(a, set()).add(b)

        # Convert sets to lists for faster iteration later.
        for k in list(self.served_adj.keys()):
            self.served_adj[k] = list(self.served_adj[k])

        # Parameters controlling cost model (tuned to be informative, not admissible):
        self.cost_per_step = 2         # base cost per origin-advance (up action plus overhead)
        self.missing_lift_penalty = 1  # penalty when a needed successor lacks lift-at in current state
        self.unreachable_penalty = 1000  # large finite cost when no directed path exists

    def _shortest_directed_distance(self, start, goal):
        """
        BFS on the static served adjacency to compute shortest directed distance
        from start to goal (number of edges). Returns integer length if reachable,
        otherwise returns None.
        """
        if start == goal:
            return 0
        visited = set()
        q = deque()
        q.append((start, 0))
        visited.add(start)
        while q:
            node, dist = q.popleft()
            for succ in self.served_adj.get(node, []):
                if succ == goal:
                    return dist + 1
                if succ not in visited:
                    visited.add(succ)
                    q.append((succ, dist + 1))
        return None

    def __call__(self, node):
        """
        Compute the heuristic estimate for `node`.

        Steps:
        - Return 0 if goals already satisfied.
        - Build a mapping of passenger-2 -> current passenger from (origin ...) facts.
        - For each goal origin (origin P2 target), compute shortest directed distance
          from current passenger to target using the served graph. Convert that distance
          into an action estimate using cost_per_step and add lift-at penalties for
          intermediate successors that currently do not have the (lift-at ...) fact.
        - If there is no current origin for a given passenger-2 or no path in the
          served graph, add a large but finite unreachable_penalty so the heuristic
          remains finite.
        """
        state = node.state

        # Goal check (heuristic must be 0 for goal states).
        if self.goals <= state:
            return 0

        # Parse current origin mapping in the state: origin_map[P2] = passenger
        origin_map = {}
        for fact in state:
            if match(fact, "origin", "*", "*"):
                _, p2, p = get_parts(fact)
                origin_map[p2] = p

        # Precompute which passengers currently have lift-at.
        lift_at = {get_parts(f)[1] for f in state if match(f, "lift-at", "*")}

        total_cost = 0

        # For each origin goal in the task, estimate cost to move current origin pointer to target.
        for goal_fact in self.goals:
            if not match(goal_fact, "origin", "*", "*"):
                # Skip non-origin goals (defensive; domain's goals are origin facts)
                continue
            _, goal_p2, goal_target = get_parts(goal_fact)

            # Current passenger this passenger-2 points to in the state.
            if goal_p2 not in origin_map:
                # If there is no origin fact for this passenger-2 in the state, treat as costly.
                total_cost += self.unreachable_penalty
                continue

            cur = origin_map[goal_p2]

            if cur == goal_target:
                # Already satisfied for this specific origin requirement.
                continue

            # Find shortest directed path length (number of edges) from cur to goal_target.
            dist = self._shortest_directed_distance(cur, goal_target)
            if dist is None:
                # No directed path in served graph; assign large finite penalty.
                total_cost += self.unreachable_penalty
                continue

            # Base cost: each required origin-advance costs roughly cost_per_step actions.
            cost = dist * self.cost_per_step

            # Add small penalty for missing lift-at on intermediate successors along a
            # representative shortest path. To do this we reconstruct one shortest path
            # by BFS again (this is cheap given the static graph) and count missing lift-ats.
            # If multiple equivalent shortest paths exist, this picks one.
            # This penalty attempts to capture that an up requires (lift-at successor).
            # Reconstruct path using parent pointers.
            if dist > 0:
                # BFS with parents to reconstruct a shortest path
                parent = {}
                q = deque([cur])
                visited = {cur}
                found = False
                while q and not found:
                    nodeq = q.popleft()
                    for succ in self.served_adj.get(nodeq, []):
                        if succ not in visited:
                            visited.add(succ)
                            parent[succ] = nodeq
                            if succ == goal_target:
                                found = True
                                break
                            q.append(succ)
                # Reconstruct path nodes from cur (exclusive) to goal_target (inclusive)
                path_nodes = []
                nodep = goal_target
                while nodep != cur:
                    path_nodes.append(nodep)
                    nodep = parent.get(nodep)
                    if nodep is None:
                        # Should not happen, but break defensively.
                        path_nodes = []
                        break
                path_nodes.reverse()  # now is [first_successor, ..., goal_target]
                # For each successor along the path check presence of lift-at (required by up)
                missing_lifts = sum(1 for p in path_nodes if p not in lift_at)
                cost += missing_lifts * self.missing_lift_penalty

            total_cost += cost

        # Ensure heuristic returns a finite integer.
        try:
            return int(total_cost)
        except Exception:
            return int(self.unreachable_penalty)