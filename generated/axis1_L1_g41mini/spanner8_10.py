from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(u4srx1nb spanner1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary:
    This heuristic estimates the number of actions required to reach the goal state 
    by focusing on the conditions set by the goal predicate (g59iwyxf ?nut). The goal 
    is achieved by transforming certain objects (nuts) to have the predicate (g59iwyxf)
    true. To achieve this, nuts must be at the gate, be 'bu1om0m1' (some kind of state/flag 
    on them), and a spanner (e2zu8y93 object) must lose the (g0brvc06) predicate while 
    setting the nut's (g59iwyxf) predicate through a chain of actions involving moving tools 
    and applying them.

    Assumptions:
    - The goal requires all nuts to have the predicate (g59iwyxf nut).
    - Nuts start at some location (typically gate) and have the predicate (bu1om0m1 nut).
    - Spanners (e2zu8y93 typed objects) start with the predicate (g0brvc06 spanner) and 
      can be moved between locations.
    - The predicate (yud7t3vb bob spanner) assigns a tool (spanner) to user/entity bob. 
    - The preconditions and effects indicate a process of moving a spanner around, applying 
      it to nuts, and changing their states until the nut has (g59iwyxf).
    - Moving spanners along a path is restricted by the adjacency predicate (kxokb5hx loc1 loc2).
    - The heuristic does not try to be admissible but focus on estimating the number of required 
      actions efficiently and reasonably accurately.

    Heuristic Initialization:
    - Extract static connectivity (kxokb5hx) as an undirected graph among locations.
    - Extract which spanners currently have the predicate (g0brvc06) - those are "active" spanners.
    - Extract which nuts are incomplete (do not have g59iwyxf) but have bu1om0m1 predicate.
    - Extract all locations and entities.
    - Extract the location of every object (u4srx1nb predicate).
    - Extract which spanners are currently assigned to bob via (yud7t3vb bob spanner).
    - Precompute the shortest distances between locations to estimate movement costs.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify nuts that are not yet transformed (not g59iwyxf) but are "present" (bu1om0m1).
       Each such nut requires one application of the third action, which requires:
       - The nut to be at a location.
       - A spanner that is both held by bob and at the nut's location (or moved there),
         and the spanner is active (g0brvc06).
    2. To transform each nut:
       a) Ensure the spanner is moved from its current location to the nut's location.
          Cost: the shortest path in number of moves between spanner location and nut location.
          Each move requires an application of action n9sbfmrs.
       b) Then, apply action kjsqfni8 to the nut at that location (one action).
       c) Note: For the spanner to be "held" by bob, bob's location must coincide with or 
          be involved. However, bob's location is not explicitly stated in goals or predicates; 
          we assume bob holds the spanner by (yud7t3vb bob spanner).
       d) It is assumed bob does not move. Heuristic ignores bob's location and focuses on spanner 
          movements and nuts transformations.
    3. Sum up these estimated costs per nut.
    4. If multiple spanners exist, assign nuts to nearest spanners to minimize sum of movement costs.
    5. Lastly, account for the spanners that lose the predicate (g0brvc06) after applying the action 
       (need to "re-activate" spanner by action moi6wnfu?), but since this is complicated and 
       only one action resets it by removing (g0brvc06) and gaining g59iwyxf on nut, 
       assume a constant overhead per nut (e.g. one additional action for resetting).

    This heuristic:
    - Uses shortest path distances on locations (kxokb5hx) to estimate spanner move costs.
    - Assigns nuts to spanners with the minimal transportation cost.
    - Adds one action per nut for the application of the final operator.
    - Adds one action per nut to reset the spanner (moi6wnfu).
    - Returns 0 if all nuts have g59iwyxf predicate (goal state).
    """

    def __init__(self, task):
        # Extract static adjacency information.
        self.adjacency = {}
        self.locations = set()
        for fact in task.static:
            if match(fact, "kxokb5hx", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.adjacency.setdefault(loc1, set()).add(loc2)
                self.adjacency.setdefault(loc2, set()).add(loc1)

        # Precompute all pairs shortest paths between locations using BFS since graph presumably small.
        self.distances = self._compute_all_pairs_shortest_paths()

        self.task = task  # Keep reference to task

    def _compute_all_pairs_shortest_paths(self):
        # Compute shortest path distances between all pairs of locations using BFS.
        distances = {}
        for loc in self.locations:
            distances[loc] = self._bfs_distances(loc)
        return distances

    def _bfs_distances(self, start):
        from collections import deque

        dist = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for nbr in self.adjacency.get(current, ()):
                if nbr not in dist:
                    dist[nbr] = dist[current] + 1
                    queue.append(nbr)
        return dist

    def __call__(self, node):
        state = node.state

        # Check if goal already reached.
        # Goal: all (g59iwyxf ?nut) predicates true
        goal_nuts = set()  # nuts that should have g59iwyxf in goal
        for goal_fact in self.task.goals:
            if match(goal_fact, "g59iwyxf", "*"):
                parts = get_parts(goal_fact)
                nut = parts[1]
                goal_nuts.add(nut)

        # Gather nuts that still need to be transformed (missing g59iwyxf)
        incomplete_nuts = []
        for nut in goal_nuts:
            # If nut has (g59iwyxf) in state, skip
            if any(match(fact, "g59iwyxf", nut) for fact in state):
                continue
            # Only consider nuts that have (bu1om0m1 nut) predicate, i.e. eligible for transformation
            if any(match(fact, "bu1om0m1", nut) for fact in state):
                incomplete_nuts.append(nut)

        if not incomplete_nuts:
            # All nuts have g59iwyxf: goal state
            return 0

        # Extract locations of nuts
        nut_locations = {}
        for fact in state:
            if match(fact, "u4srx1nb", "*", "*"):
                parts = get_parts(fact)
                obj = parts[1]
                loc = parts[2]
                if obj in incomplete_nuts:
                    nut_locations[obj] = loc

        # Extract spanners that are active (have g0brvc06)
        active_spanners = set()
        for fact in state:
            if match(fact, "g0brvc06", "*"):
                parts = get_parts(fact)
                spanner = parts[1]
                active_spanners.add(spanner)

        if not active_spanners:
            # No active spanners => must at least activate one (heuristic cost at least 1 per nut)
            # But since activation requires moving spanner and applying, approximate each nut cost +1
            # to avoid underestimating.
            # However, without spanner locations we cannot do better, so return number of nuts * 2.
            return len(incomplete_nuts) * 2

        # Extract spanner locations
        spanner_locations = {}
        for fact in state:
            if match(fact, "u4srx1nb", "*", "*"):
                parts = get_parts(fact)
                obj = parts[1]
                loc = parts[2]
                if obj in active_spanners:
                    spanner_locations[obj] = loc

        # Extract which spanners bob holds by (yud7t3vb bob spanner)
        bob_spanners = set()
        for fact in state:
            if match(fact, "yud7t3vb", "bob", "*"):
                parts = get_parts(fact)
                spanner = parts[2]
                if spanner in active_spanners:
                    bob_spanners.add(spanner)

        # If bob holds no active spanner, heuristic assumes cost to fix at least once per nut
        if not bob_spanners:
            # bob must get a spanner, cost at least 1 per nut
            # return double number of nuts for safety margin
            return len(incomplete_nuts) * 2

        # For heuristic, consider only spanners held by bob to apply transformations (as per precondition)
        usable_spanners = bob_spanners.intersection(active_spanners)

        if not usable_spanners:
            # As above, if no usable spanners, cost to fix
            return len(incomplete_nuts) * 2

        # Calculate cost of moving spanner to each nut and applying last action
        # For each nut: assign to spanner with minimum cost
        total_cost = 0

        # Precompute distances from each spanner location to each nut location
        # If unknown path (no distance), assign large cost to avoid that match
        import math

        large_cost = 1000

        # Create cost matrix nut x spanner for spanner move cost
        cost_matrix = []
        for nut in incomplete_nuts:
            nut_loc = nut_locations.get(nut, None)
            row = []
            for spanner in usable_spanners:
                spanner_loc = spanner_locations.get(spanner, None)
                if spanner_loc is None or nut_loc is None:
                    dist = large_cost
                else:
                    dist = self.distances.get(spanner_loc, {}).get(nut_loc, large_cost)
                row.append(dist)
            cost_matrix.append(row)

        # Assign nuts to spanners greedily to minimize sum of distances:
        # This is a minimal assignment problem, but greedy suffice here as approximate
        # Convert to list for spanners indexing
        spanner_list = list(usable_spanners)
        assigned_spanners = [None] * len(incomplete_nuts)
        assigned = set()

        for nut_idx in range(len(incomplete_nuts)):
            row = cost_matrix[nut_idx]
            min_dist = large_cost
            min_spanner_idx = None
            for sp_idx, dist in enumerate(row):
                if dist < min_dist and sp_idx not in assigned:
                    min_dist = dist
                    min_spanner_idx = sp_idx
            # If all spanners taken or no better found, just pick min dist ignoring assignment
            if min_spanner_idx is None:
                min_dist = min(row)
                min_spanner_idx = row.index(min_dist)
            assigned_spanners[nut_idx] = min_spanner_idx
            assigned.add(min_spanner_idx)

        # Compute cost summing:
        # Each move from spanner_loc to nut_loc costs dist actions (n9sbfmrs), each assumed 1 action
        # Then, 1 action to apply kjsqfni8 (transform nut and spanner status)
        # Then, 1 action to reset spanner (moi6wnfu), to reactivate (g0brvc06)
        # Total per nut: dist + 2 actions
        for nut_idx, sp_idx in enumerate(assigned_spanners):
            dist = cost_matrix[nut_idx][sp_idx]
            if dist == large_cost:
                # No path found; assign high cost to reflect difficulty
                dist = large_cost
            total_cost += dist + 2  # 1 apply + 1 reset

        return total_cost
