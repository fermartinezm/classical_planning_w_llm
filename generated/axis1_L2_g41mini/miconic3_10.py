from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact (remove parentheses, split)."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Check if a fact matches a pattern with wildcards.
    """
    parts = get_parts(fact)
    return all(fnmatch(p, pat) for p, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (unkhrp2l).

    Summary:
    ----------
    This heuristic estimates the number of actions needed to achieve the goal 
    condition(s) of the form (guby8wxu ?x) for some objects of type jxhpkqr5. 
    The main goal is to transform certain predicates on these objects through a 
    sequence of steps involving intermediate predicates and movement of ql6b2h31 
    objects marked by the (v0yy74n7 ?loc) predicate. The heuristic is 
    non-admissible but tries to closely approximate the minimal action count 
    while being fast to compute.

    Assumptions:
    ------------
    - The goal predicates are of the form (guby8wxu ?jxhpkqr5_object).
    - The transformation sequence for each relevant jxhpkqr5-object roughly follows:
        1. (pxch7pi8 ?jxhpkqr5 ?ql6b2h31) must become false to enable (eacj8ftl ?jxhpkqr5).
        2. Then (eacj8ftl ?jxhpkqr5) must become true and then false to reach (guby8wxu ?jxhpkqr5).
    - The predicate (v0yy74n7 ?ql6b2h31) marks the current "active" location.
    - The predicates (pvhp8m0p ?ql6b2h31 ?ql6b2h31) describe a connectivity graph (directional).
    - The "move" actions (jt9wnwgs and v90zoaro) move the active state from one ql6b2h31 location to another along edges.
    - The heuristic treats movement cost between ql6b2h31 objects as one action per edge traversed.
    - It assumes shortest path moves between current and target locations.
    - The goal is reached if all goal jxhpkqr5 objects have (guby8wxu) true.
    
    Heuristic Initialization:
    --------------------------
    - Extract all goal objects that must have (guby8wxu).
    - Extract static connectivity (pvhp8m0p) as a directed graph between ql6b2h31 objects.
    - Precompute adjacency lists from static predicates for quick shortest-path search.
    
    Step-By-Step Thinking for Computing Heuristic:
    -----------------------------------------------
    1. Identify the set of goal jxhpkqr5 objects that need to achieve (guby8wxu).
    2. For each such object:
        a. If (guby8wxu ?obj) true in current state, no cost for this object.
        b. Otherwise, determine whether it currently has (eacj8ftl ?obj) true.
        c. Determine its current associated ql6b2h31 via (pxch7pi8 ?obj ?loc) or (pahbmn6t ?obj ?loc).
           Note that exactly one of (pxch7pi8) or (pahbmn6t) should hold for each object wrt locations.
    3. Determine the current ql6b2h31 location where (v0yy74n7 ?loc) holds (there is exactly one such).
    4. To achieve (guby8wxu ?obj), must perform:
        - If currently (pxch7pi8), must apply makpz2tc to get (eacj8ftl).
        - Then apply jg64qouf to get (guby8wxu).
    5. makpz2tc requires the active location to be the same as the pxch7pi8 location.
       jg64qouf requires the active location to be the same as the pahbmn6t location.
    6. Since active location moves cost 1 per edge, the heuristic estimates:
       - Distance from current active location to pxch7pi8 location (if pxch7pi8 holds) for makpz2tc.
       - Distance from makpz2tc effect's location (which is pxch7pi8 location) to pahbmn6t location for jg64qouf.
       - Each action (move, makpz2tc, jg64qouf) counts as cost 1.
    7. If (eacj8ftl ?obj) is already true, we can skip makpz2tc and directly move to pahbmn6t location for jg64qouf.
    8. Sum costs over all goal objects.
    9. Return 0 if goal achieved.
    10. Since shortest path is unknown at runtime, BFS is used to compute shortest distances between nodes.
    11. If no path exists, heuristic assigns a high cost (but finite).
    """

    def __init__(self, task):
        self.goals = set(task.goals)
        self.static = task.static

        # Extract all goal objects of type jxhpkqr5 that must satisfy (guby8wxu ?obj)
        self.goal_objects = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "guby8wxu":
                self.goal_objects.add(parts[1])

        # Build directed graph from static facts (pvhp8m0p ?from ?to)
        self.adjacency = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "pvhp8m0p":
                src, dst = parts[1], parts[2]
                self.adjacency.setdefault(src, set()).add(dst)

    def shortest_path_distance(self, start, goal):
        """Compute shortest path length from start to goal in adjacency graph; returns None if unreachable."""
        if start == goal:
            return 0
        visited = set()
        queue = [(start, 0)]
        while queue:
            node, dist = queue.pop(0)
            if node == goal:
                return dist
            if node in visited:
                continue
            visited.add(node)
            for neighbor in self.adjacency.get(node, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return None

    def __call__(self, node):
        state = node.state

        # If goal already satisfied, heuristic is zero
        if self.goals <= state:
            return 0

        # Find current active location where (v0yy74n7 ?loc) holds
        active_locs = [get_parts(f)[1] for f in state if match(f, "v0yy74n7", "*")]
        if not active_locs:
            # No active location in state, assign a high cost
            return 1000
        active_loc = active_locs[0]

        total_cost = 0

        for obj in self.goal_objects:
            # If already at goal predicate (guby8wxu), no cost for that object
            if f"(guby8wxu {obj})" in state:
                continue

            # Check if (eacj8ftl ?obj) holds
            has_eacj8ftl = f"(eacj8ftl {obj})" in state

            # Find pxch7pi8 location (should be exactly one or zero)
            pxch7pi8_locs = [get_parts(f)[2] for f in state if match(f, "pxch7pi8", obj, "*")]
            pxch7pi8_loc = pxch7pi8_locs[0] if pxch7pi8_locs else None

            # Find pahbmn6t location (should be exactly one or zero)
            pahbmn6t_locs = [get_parts(f)[2] for f in state if match(f, "pahbmn6t", obj, "*")]
            pahbmn6t_loc = pahbmn6t_locs[0] if pahbmn6t_locs else None

            # If object has no pxch7pi8 nor pahbmn6t, but goal not reached, likely stuck - assign high cost
            if pxch7pi8_loc is None and pahbmn6t_loc is None:
                total_cost += 100
                continue

            # Step 1: move to pxch7pi8 location and apply makpz2tc if needed
            if not has_eacj8ftl:
                # Need to move active location to pxch7pi8 loc if different
                if pxch7pi8_loc is None:
                    # Can't apply makpz2tc, but eacj8ftl not present, penalize heavily
                    total_cost += 100
                    continue
                dist1 = self.shortest_path_distance(active_loc, pxch7pi8_loc)
                if dist1 is None:
                    dist1 = 50  # Assign large finite cost for unreachable

                # makpz2tc action cost = 1
                total_cost += dist1 + 1

                # Update active_loc to pxch7pi8_loc after action for next step
                active_loc_for_step2 = pxch7pi8_loc
            else:
                # Already has eacj8ftl, skip makpz2tc
                active_loc_for_step2 = active_loc

            # Step 2: move to pahbmn6t location and apply jg64qouf if needed
            if pahbmn6t_loc is None:
                # Cannot complete goal for this object without pahbmn6t location - penalize
                total_cost += 100
                continue

            dist2 = self.shortest_path_distance(active_loc_for_step2, pahbmn6t_loc)
            if dist2 is None:
                dist2 = 50  # Large finite penalty

            # jg64qouf action cost = 1
            total_cost += dist2 + 1

        return total_cost
