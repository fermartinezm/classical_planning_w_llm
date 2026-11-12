from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(ypt9mrgu pjr7u2i2 j0on2kgr)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the `secret_domain` (hi4l5zmb) domain.

    Summary
    -------
    This heuristic estimates the number of actions required to reach the goal state by
    focusing on satisfying the goal predicate (cb6fw2xd ?stq0j ?tlfqu),
    which encodes that an object ?stq0j (type nh1sy8v9) has the qekv4465-type object ?tlfqu effectively "delivered" or associated.

    The domain manipulates three main predicates relevant to goal satisfaction:
    - ypt9mrgu(?en18p - k9b7lknr, ?stq0j - nh1sy8v9): represents that entity ?en18p is related to location/object ?stq0j.
    - fguw2142(?stq0j - nh1sy8v9): appears to designate a "free" or "available" location or object.
    - cb6fw2xd(?stq0j - nh1sy8v9, ?tlfqu - qekv4465): predicate representing the goal condition, indicating a coupling between ?stq0j and ?tlfqu.

    The heuristic measures for each goal (cb6fw2xd ?stq0j ?tlfqu) whether it has been achieved and, if not, estimates how many actions are needed to achieve it based on:
    - Whether the goal fact already holds,
    - Or whether the required relations involving ypt9mrgu (ownership or presence) and fguw2142 (free status) need to be changed,
    - Factoring in the static adjacency/connection predicates (ga4l9do0, s4jrpk2b, vom4pfig, vjcqgdgn) to estimate minimal steps for moving/relocating objects.

    Assumptions
    -----------
    - Actions mostly move or swap relationships of k9b7lknr objects over nh1sy8v9 objects and qekv4465 objects.
    - Each goal requires one cb6fw2xd fact to be true for some pair (?stq0j, ?tlfqu).
    - Static predicates (ga4l9do0, s4jrpk2b, vom4pfig, vjcqgdgn) describe connections or adjacencies between nh1sy8v9 objects.
    - The "free" predicate fguw2142 indicates availability of a location to be occupied or swapped.
    - The heuristic is not admissible; it aims to provide a quick, informative estimate guiding the search efficiently.
    - State and static are sets of strings encoding facts.
    - The heuristic returns 0 if all goals are satisfied, finite and nonzero otherwise.

    Heuristic Initialization
    ------------------------
    - Extract goals: The set of required cb6fw2xd predicates for completion.
    - Extract static facts describing connectivity:
      * ga4l9do0: "adjacency" arcs between nh1sy8v9 objects.
      * s4jrpk2b, vom4pfig, vjcqgdgn: other directed relations that can be used for movements.
      These are stored as adjacency maps for efficient lookup.
    - Extract sets of all objects by type (k9b7lknr, nh1sy8v9, qekv4465).
    
    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each goal fact (cb6fw2xd ?stq0j ?tlfqu):
      1. Check if the goal fact already holds in the current state. If yes, no cost for this goal.
      2. If missing:
         a. Identify the current location(s) of the k9b7lknr object ?en18p related to this goal via ypt9mrgu.
            We seek an ?en18p such that (ypt9mrgu ?en18p ?some_location).
         b. Identify the current "free" location (fguw2142) indicating where moves can occur.
         c. Use static adjacency relations to estimate how many moves are required to:
            - Move the k9b7lknr object from its current nh1sy8v9 location to the goal ?stq0j location.
            - Manage frees and swaps needed to establish (cb6fw2xd ?stq0j ?tlfqu).
      3. Approximate the minimal number of actions as the sum of:
         - Estimated moves to bring the k9b7lknr object close to the target nh1sy8v9 location.
         - Actions to reassign the qekv4465 object from its current nh1sy8v9 to the goal nh1sy8v9.
      4. Sum these costs over all goals for a total heuristic estimate.
      
    The heuristic uses shortest distance over static adjacency graphs between nh1sy8v9 objects for movement cost estimation.
    """

    def __init__(self, task):
        self.goals = task.goals  # Goal facts to achieve (cb6fw2xd ...)

        # Extract static adjacency relations for nh1sy8v9 nodes to build graphs for shortest path estimation.
        # ga4l9do0, s4jrpk2b, vom4pfig, vjcqgdgn are static and describe directed connectivity.

        # We'll build directed adjacency maps:
        self.adjacency = {}  # mapping: nh1sy8v9_obj -> set of adjacent nh1sy8v9 objects
        for fact in task.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in {"ga4l9do0", "s4jrpk2b", "vom4pfig", "vjcqgdgn"} and len(parts) == 3:
                src, dst = parts[1], parts[2]
                self.adjacency.setdefault(src, set()).add(dst)

        # Extract all facts for static membership
        self.all_nh1sy8v9 = set()
        for fact in task.facts:
            parts = get_parts(fact)
            # nh1sy8v9 are the second argument for some predicates, or can be found in static adjacency
            if len(parts) > 1:
                for arg in parts[1:]:
                    self.all_nh1sy8v9.add(arg)
        # Also add from static adjacency keys/values
        for key in self.adjacency:
            self.all_nh1sy8v9.add(key)
            self.all_nh1sy8v9.update(self.adjacency[key])

        # Precompute shortest paths over the combined adjacency graph between nh1sy8v9 objects using BFS.
        self.shortest_paths_cache = {}

    def shortest_path_length(self, start, goal):
        """
        Compute shortest distance (number of hops) between two nh1sy8v9 objects in the adjacency graph.

        If no path exists, return a large number (e.g., 100).
        Cache results to speed up repeated calls.
        """
        if start == goal:
            return 0
        key = (start, goal)
        if key in self.shortest_paths_cache:
            return self.shortest_paths_cache[key]

        from collections import deque

        visited = {start}
        queue = deque([(start, 0)])

        while queue:
            current, dist = queue.popleft()
            for neighbor in self.adjacency.get(current, ()):
                if neighbor == goal:
                    self.shortest_paths_cache[key] = dist + 1
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # No path found
        self.shortest_paths_cache[key] = 100
        return 100

    def __call__(self, node):
        state = node.state

        # If goal is reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # Extract current facts for quick lookup.
        facts = state  # frozenset of strings

        # Extract all current fguw2142 (free) locations:
        free_locations = {get_parts(f)[1] for f in facts if match(f, "fguw2142", "*")}

        # Extract current cb6fw2xd facts:
        current_cb6fw2xd = {(get_parts(f)[1], get_parts(f)[2]) for f in facts if match(f, "cb6fw2xd", "*", "*")}

        # Extract ypt9mrgu relations: map from k9b7lknr -> nh1sy8v9
        ypt9mrgu_map = {}
        for f in facts:
            if match(f, "ypt9mrgu", "*", "*"):
                parts = get_parts(f)
                en18p, stq0j = parts[1], parts[2]
                ypt9mrgu_map[en18p] = stq0j

        # Extract qp2t6u4s relations: k9b7lknr to qekv4465
        qp2t6u4s_map = {}
        for f in facts:
            if match(f, "qp2t6u4s", "*", "*"):
                parts = get_parts(f)
                en18p, tlfqu = parts[1], parts[2]
                qp2t6u4s_map[en18p] = tlfqu

        # The heuristic estimates per goal fact (cb6fw2xd stq0j tlfqu)
        # Note: The goal is to have cb6fw2xd true for each (stq0j, tlfqu).
        # Our plan: For each goal not yet satisfied, estimate cost to:
        # - Find a k9b7lknr ?en18p that currently relates to some ?loc and has qp2t6u4s ?en18p ?tlfqu.
        # - Move ypt9mrgu ?en18p from current location to goal location ?stq0j.
        # - Free locations management and final assignment of cb6fw2xd.

        # For simplicity, for each goal (stq0j, tlfqu), find any en18p that now owns tlfqu,
        # estimate cost = shortest path from ypt9mrgu loc of en18p to stq0j + 2 (one to free current free location and one action to establish cb6fw2xd).
        # If no en18p owns tlfqu, add a penalty.

        total_estimate = 0

        LARGE_COST = 50  # penalty for unsatisfiable parts, must be finite

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] != "cb6fw2xd":
                continue
            goal_stq0j, goal_tlfqu = parts[1], parts[2]

            if (goal_stq0j, goal_tlfqu) in current_cb6fw2xd:
                # Goal satisfied, no cost.
                continue

            # Find k9b7lknr objects that own goal_tlfqu:
            candidates = [en for en, tlf in qp2t6u4s_map.items() if tlf == goal_tlfqu]

            if not candidates:
                # No owner of tlfqu - high penalty
                total_estimate += LARGE_COST
                continue

            # For each candidate, estimate cost: 
            # cost = move k9b7lknr from current location to goal_stq0j (shortest path)
            # plus 2 for freeing and assigning cb6fw2xd

            min_cost_for_goal = LARGE_COST

            for en18p in candidates:
                current_loc = ypt9mrgu_map.get(en18p)
                if current_loc is None:
                    # No ypt9mrgu for en18p, very costly
                    cost = LARGE_COST
                else:
                    dist = self.shortest_path_length(current_loc, goal_stq0j)
                    # Add fixed costs:
                    cost = dist + 2

                if cost < min_cost_for_goal:
                    min_cost_for_goal = cost

            total_estimate += min_cost_for_goal

        return total_estimate
