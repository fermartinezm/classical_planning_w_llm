from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain (unkhrp2l).

    Summary
    -------
    This heuristic estimates the number of actions required to reach the goal state in the secret_domain.
    The domain manipulates objects of two types: jxhpkqr5 and ql6b2h31, with goal conditions requiring
    some jxhpkqr5 objects to satisfy (guby8wxu ?obj). Achieving (guby8wxu ?obj) involves transitioning
    from predicates (eacj8ftl ?obj) and (pxch7pi8 ?obj ?loc) via actions interacting with (v0yy74n7 ?loc)
    and path relations (pvhp8m0p ?from ?to).

    Assumptions
    -----------
    - The goal is composed of facts of the form (guby8wxu ?jxhpkqr5-object).
    - To reach (guby8wxu ?obj), an (eacj8ftl ?obj) fact must be established by performing actions
      that require the object to be at some location with predicate (v0yy74n7 ?loc).
    - The location marked by (v0yy74n7 ?loc) can be changed by two movement actions (jt9wnwgs and v90zoaro)
      that cause the (v0yy74n7 ?loc) fact to move along a path defined by (pvhp8m0p ?from ?to).
    - The domain essentially encodes transporting "something" along a network of connected ql6b2h31 objects.
    - Static facts provide the graph edges (pvhp8m0p) that define connectivity between locations.
    - The heuristic does not need to be admissible, so approximate shortest path lengths from current location(s)
      to required locations are used.

    Heuristic Initialization
    ------------------------
    - Extract from goal facts all jxhpkqr5 objects that must satisfy (guby8wxu ?obj).
    - Extract static facts representing the graph edges (pvhp8m0p ?from ?to).
    - Build an undirected graph of locations (ql6b2h31 objects) using these edges for shortest path computations.
    - Extract (pahbmn6t ?obj ?loc) and (pxch7pi8 ?obj ?loc) static relations linking jxhpkqr5 objects and ql6b2h31 locations.
      These map each jxhpkqr5 object to one or more locations relevant for achieving the goal conditions.
    - Precompute for each jxhpkqr5 object the required location sets used in the heuristic.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. For each goal object (must satisfy (guby8wxu ?obj)):
       a. If (guby8wxu ?obj) is already true in the current state, no cost is needed for that object.
       b. Else, the object must progress through the predicates (eacj8ftl ?obj) and then (guby8wxu ?obj).
    2. To reach (guby8wxu ?obj), the heuristic breaks down needed actions:
       - If the object currently satisfies (eacj8ftl ?obj), one action (jg64qouf) can achieve (guby8wxu ?obj)
         if the location with (v0yy74n7 ?loc) matches the corresponding (pahbmn6t ?obj ?loc).
       - If the object currently satisfies (pxch7pi8 ?obj ?loc), then an action (makpz2tc) can establish (eacj8ftl ?obj),
         but requires the (v0yy74n7 ?loc) to be true for that loc.
       - Otherwise, the object is not yet ready and needs to move the (v0yy74n7 ?loc) marker along the graph edges to the matching location(s).
    3. For the current state, identify the current (v0yy74n7 ?loc) location(s) (only one is true at a time).
    4. For each object that requires (eacj8ftl) or (pxch7pi8) relocation to achieve (guby8wxu), estimate the minimal
       distance along the graph from the current (v0yy74n7 ?loc) to any of the required locations for that object.
       Each step in distance corresponds to one movement action (jt9wnwgs or v90zoaro).
    5. Sum heuristic costs as:
       - Movement steps to reach required locations for (pxch7pi8) or (eacj8ftl).
       - Plus 1 action for makpz2tc (to get from pxch7pi8 to eacj8ftl).
       - Plus 1 action for jg64qouf (to get from eacj8ftl to guby8wxu).
    6. Take max or sum over all required objects to balance efficiency and informativeness; here summing is chosen to guide search well.
    7. If no matching (v0yy74n7) is present or any object lacks reachable locations, fall back to a minimal positive heuristic to avoid zero except at goal.
    8. The heuristic is zero only if all (guby8wxu ?obj) goals are fulfilled.

    This heuristic is efficient: shortest path calculations use BFS on a modest static graph, and predicate checks are simple set lookups.
    The heuristic guides the planner to move (v0yy74n7) locations as needed and promote predicates along the chain until the goal is reached.
    It trades admissibility for being more informed for greedy best-first search.
    """

    def __init__(self, task):
        """Initialize heuristic by extracting static graph edges, object-location links, and goals."""
        self.goals = task.goals

        # Extract all goal objects that need (guby8wxu ?obj)
        self.goal_objects = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts and parts[0] == "guby8wxu":
                self.goal_objects.add(parts[1])

        static = task.static

        # Build undirected graph of location nodes from (pvhp8m0p ?from ?to)
        self.graph = {}
        for fact in static:
            parts = get_parts(fact)
            if parts[0] == "pvhp8m0p":
                src, tgt = parts[1], parts[2]
                self.graph.setdefault(src, set()).add(tgt)
                self.graph.setdefault(tgt, set()).add(src)

        # Map jxhpkqr5 objects -> sets of ql6b2h31 locations for predicates:
        # (pahbmn6t ?obj ?loc) and (pxch7pi8 ?obj ?loc).
        self.obj_to_pahbmn6t_locs = {}
        self.obj_to_pxch7pi8_locs = {}
        for fact in static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "pahbmn6t":
                obj, loc = parts[1], parts[2]
                self.obj_to_pahbmn6t_locs.setdefault(obj, set()).add(loc)
            elif pred == "pxch7pi8":
                obj, loc = parts[1], parts[2]
                self.obj_to_pxch7pi8_locs.setdefault(obj, set()).add(loc)

    def shortest_path_length(self, graph, start, goals):
        """
        Breadth-first search in undirected graph to find shortest distance from start to any goal in goals set.
        Returns the minimal number of edges, or None if no path found.
        """
        if start in goals:
            return 0
        visited = set()
        queue = [(start, 0)]
        visited.add(start)
        while queue:
            current, dist = queue.pop(0)
            for neighbor in graph.get(current, []):
                if neighbor in goals:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return None

    def __call__(self, node):
        """Compute heuristic estimate based on distance and predicate progress for each goal object."""
        state = node.state

        # Quick check: if all goal facts are reached, heuristic is zero
        if self.goals <= state:
            return 0

        # Extract the single current location from (v0yy74n7 ?loc) predicate in current state.
        current_v0yy74n7_locs = [get_parts(fact)[1] for fact in state if fact.startswith("(v0yy74n7 ")]
        if not current_v0yy74n7_locs:
            # No current location set, heuristic positive minimal fallback.
            return 5
        current_location = current_v0yy74n7_locs[0]

        # For each goal object, compute required cost to achieve (guby8wxu ?obj).
        total_cost = 0

        for obj in self.goal_objects:
            # Check if (guby8wxu ?obj) already true
            if f"(guby8wxu {obj})" in state:
                # Object goal satisfied, no cost
                continue

            # Determine if (eacj8ftl ?obj) true -> can do final action if location matches
            has_eacj8ftl = f"(eacj8ftl {obj})" in state

            # Determine if (pxch7pi8 ?obj ?loc) true for some loc(s) in state
            pxch7pi8_locs = {loc for loc in self.obj_to_pxch7pi8_locs.get(obj, set())
                             if f"(pxch7pi8 {obj} {loc})" in state}

            # Relevant locations for "pahbmn6t" predicate for jg64qouf action (final transition)
            pahbmn6t_locs = self.obj_to_pahbmn6t_locs.get(obj, set())

            # Heuristic cost components for this object:
            # - Movement costs for arriving at required loc (via moving (v0yy74n7 ?loc))
            # - Action costs: makpz2tc (pxch7pi8 -> eacj8ftl), jg64qouf (eacj8ftl -> guby8wxu)

            obj_cost = 0

            if has_eacj8ftl:
                # To apply jg64qouf:
                # Need current_location to be in pahbmn6t_locs for object.
                dist = 0 if current_location in pahbmn6t_locs else \
                    self.shortest_path_length(self.graph, current_location, pahbmn6t_locs)
                if dist is None:
                    # Cannot reach needed location: assign a moderate penalty.
                    obj_cost += 10
                else:
                    # Need to move to location + 1 action for jg64qouf
                    obj_cost += dist + 1

            elif pxch7pi8_locs:
                # Need to move (v0yy74n7) to one of the pxch7pi8 locations to enable makpz2tc
                # Then also move to pahbmn6t location to apply jg64qouf
                dist_to_px = self.shortest_path_length(self.graph, current_location, pxch7pi8_locs)
                if dist_to_px is None:
                    obj_cost += 10
                else:
                    # After makpz2tc, we get (eacj8ftl) and have to move to pahbmn6t location
                    dist_to_pah = 0
                    if pahbmn6t_locs:
                        dist_to_pah = self.shortest_path_length(
                            self.graph,
                            next(iter(pxch7pi8_locs)),  # approximate from px location
                            pahbmn6t_locs,
                        )
                        if dist_to_pah is None:
                            obj_cost += 10
                    dist_to_pah = dist_to_pah or 0
                    # Total is moves to pxch7pi8 loc + 1 makpz2tc + moves to pahbmn6t loc + 1 jg64qouf
                    obj_cost += dist_to_px + 1 + dist_to_pah + 1

            else:
                # The object is not in either (eacj8ftl) or (pxch7pi8) predicates in state.
                # Heuristic must estimate cost to first establish pxch7pi8 (unknown how, assume at one of static locs)
                # Conservative heuristic: estimate distance from current location to any pxch7pi8 location for obj,
                # plus moves to pahbmn6t loc, plus 2 actions (makpz2tc + jg64qouf).

                all_pxch7pi8_locs = self.obj_to_pxch7pi8_locs.get(obj, set())
                all_pahbmn6t_locs = self.obj_to_pahbmn6t_locs.get(obj, set())
                if not all_pxch7pi8_locs or not all_pahbmn6t_locs:
                    # Missing info; assign penalty
                    obj_cost += 15
                else:
                    dist_to_px = self.shortest_path_length(self.graph, current_location, all_pxch7pi8_locs)
                    dist_to_pah = self.shortest_path_length(self.graph, next(iter(all_pxch7pi8_locs)), all_pahbmn6t_locs)
                    if dist_to_px is None or dist_to_pah is None:
                        obj_cost += 15
                    else:
                        obj_cost += dist_to_px + 1 + dist_to_pah + 1

            total_cost += obj_cost

        # Ensure heuristic never returns zero except goal states (checked above)
        return max(total_cost, 1)
