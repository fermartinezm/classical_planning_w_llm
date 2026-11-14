from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (unkhrp2l) domain.

    Summary
    -------
    This heuristic estimates the number of actions required to achieve all goal facts of the form (guby8wxu ?obj).
    The domain involves two object types (`jxhpkqr5` and `ql6b2h31`) with predicates relating them and one special predicate indicating a unique "location" fact (v0yy74n7) that moves between ql6b2h31 objects,
    and some agent states captured by predicates `eacj8ftl`, `pxch7pi8`, and `guby8wxu`.

    The main goal predicate is `(guby8wxu ?obj)` for objects of type `jxhpkqr5`.
    Given the domain actions and predicates, to transform an object's state from `pxch7pi8` to `eacj8ftl`, then from `eacj8ftl` to `guby8wxu`, some movement and state change actions are required.

    Assumptions
    -----------
    - Each object of type `jxhpkqr5` must reach the predicate `(guby8wxu ?obj)` to fulfill the goal.
    - To get `(guby8wxu ?obj)`, `eacj8ftl ?obj` must be true first (the second-level state).
    - To get `eacj8ftl ?obj`, this requires `(pxch7pi8 ?obj ?ql6b2h31)` and the moving predicate `(v0yy74n7 ?ql6b2h31)` on the same second argument object.
    - The moving predicate `(v0yy74n7 ?loc)` can only be on one location (`ql6b2h31` object) at a time but can be moved using two symmetric actions `jt9wnwgs` and `v90zoaro`.
    - Static binary relations `(pvhp8m0p ?a ?b)` define connectivity between locations (ql6b2h31 objects), used to move `(v0yy74n7)` between locations.
    - The heuristic only requires counting the minimal estimated number of moves and actions to get all goal objects into `(guby8wxu)` state from the current state.
    - The heuristic is not necessarily admissible but should be efficiently computable and informed.

    Heuristic Initialization
    ------------------------
    - Extract all goal objects of type `jxhpkqr5` that must satisfy `(guby8wxu ?obj)`.
    - Extract from static facts the connectivity relation `(pvhp8m0p ?loc1 ?loc2)` among locations to form a graph for shortest path computations.
    - Extract static `(pahbmn6t ?obj ?loc)` and `(pxch7pi8 ?obj ?loc)` relations to identify fixed associations between `jxhpkqr5` objects and locations for enabling state transitions.
    - Precompute shortest paths between all locations relevant for movement of `(v0yy74n7)` since location moves are the main cost for transitioning objects from `pxch7pi8` to `(guby8wxu)`.

    Step-By-Step Thinking for Computing Heuristic
    ----------------------------------------------
    For each goal object obj of type `jxhpkqr5`:

    1. Check if `(guby8wxu obj)` holds in the current state:
       - If yes, this object requires no further actions.

    2. Else if `(eacj8ftl obj)` holds:
       - This means the object is in the intermediate state.
       - Then one `(jg64qouf)` action is needed to move from `(eacj8ftl)` to `(guby8wxu)`.
       - Preconditions: `(v0yy74n7 ?loc)` and `(pahbmn6t obj ?loc)` must hold.
       - So the heuristic must at least include the cost to have `(v0yy74n7)` at the location matching `(pahbmn6t obj ?loc)` if not already there.

    3. Else if `(pxch7pi8 obj ?loc_pxch)` holds (the initial state for this object):
       - To get `(eacj8ftl obj)` from `(pxch7pi8 obj ?loc_pxch)` requires action `(makpz2tc)`:
         - Preconditions `(v0yy74n7 ?loc_pxch)` and `(pxch7pi8 obj ?loc_pxch)`.
         - Effect: removes `(pxch7pi8 obj ?loc_pxch)` and adds `(eacj8ftl obj)`.
       - This means the marker `(v0yy74n7)` must be at `?loc_pxch` to perform the action.
       - To get `(guby8wxu obj)` finally, as above, `(v0yy74n7)` must later move to another location `?loc_pahb` matching `(pahbmn6t obj ?loc_pahb)`, to perform `(jg64qouf)`.
       - So two location visits by `(v0yy74n7)` may be needed:
         - First to `?loc_pxch` (if not already there), to enable `(makpz2tc)`.
         - Then to `?loc_pahb` (if different), to enable `(jg64qouf)`.

    4. The cost for moving `(v0yy74n7)` is the shortest path distance between locations in the graph formed by `(pvhp8m0p)` predicates.
    5. The heuristic sums for each goal object:
       - Actions needed: 
         - At least one `(makpz2tc)` if object currently has `(pxch7pi8)`.
         - One `(jg64qouf)` to achieve `(guby8wxu)`.
       - Movement cost:
         - Number of moves of `(v0yy74n7)` between involved locations (the current marker location to the needed locations).
    6. If multiple objects share the same needed location, the cost to move `(v0yy74n7)` there can be amortized if the heuristic tries to optimize globally.
       - However, for efficiency, we approximate by summing individual costs, ignoring synergy (to keep heuristic computations light).
    7. The current position of `(v0yy74n7)` must be extracted from current state (only one such fact).
    8. The heuristic value is the sum over all goal objects of minimal estimated cost of the required moves plus actions to transition the object from its current state to `(guby8wxu)`.

    Implementation Detail
    ---------------------
    - To keep computation efficient, a precomputed all-pairs shortest path or BFS from every node can be done on static graph.
    - Since instance sizes can be large, at minimum we implement BFS shortest path queries on-demand from `(v0yy74n7)` current location to target locations.
    - The cost model:
      - Each move of `(v0yy74n7)` between adjacent locations is 1 action.
      - Each domain action that changes object states costs 1 action.
    """

    def __init__(self, task):
        # Store goal objects that must have (guby8wxu ?obj)
        self.goal_objects = set()
        for goal_fact in task.goals:
            parts = get_parts(goal_fact)
            if parts[0] == "guby8wxu":
                self.goal_objects.add(parts[1])

        # Extract static connectivity graph between locations (ql6b2h31) from (pvhp8m0p loc1 loc2)
        self.adjacent = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "pvhp8m0p":
                loc1, loc2 = parts[1], parts[2]
                self.adjacent.setdefault(loc1, set()).add(loc2)
                self.adjacent.setdefault(loc2, set()).add(loc1)

        # Extract mappings for pxch7pi8 and pahbmn6t static facts:
        # (pxch7pi8 jxhpkqr5 ql6b2h31) and (pahbmn6t jxhpkqr5 ql6b2h31)
        # These are static references from each object to 1 or more locations.
        self.obj_to_pxch7pi8_locs = {}
        self.obj_to_pahbmn6t_locs = {}
        for fact in task.static:
            parts = get_parts(fact)
            pred = parts[0]
            obj = parts[1]
            loc = parts[2]
            if pred == "pxch7pi8":
                self.obj_to_pxch7pi8_locs.setdefault(obj, set()).add(loc)
            elif pred == "pahbmn6t":
                self.obj_to_pahbmn6t_locs.setdefault(obj, set()).add(loc)

    def shortest_path(self, start, goal):
        """Compute shortest path length from start to goal in the undirected graph.

        Returns integer distance or a high number if unreachable.
        """
        if start == goal:
            return 0
        if start not in self.adjacent or goal not in self.adjacent:
            return 100000  # Large number for unreachable

        from collections import deque

        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for neighbor in self.adjacent.get(node, []):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return 100000

    def __call__(self, node):
        state = node.state

        # Quick check if goal reached: heuristic zero only if all goal (guby8wxu objs) hold.
        if all(f"(guby8wxu {obj})" in state for obj in self.goal_objects):
            return 0

        # Extract current location of (v0yy74n7) marker - only one such fact possible.
        v0yy_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "v0yy74n7":
                v0yy_location = parts[1]
                break

        if v0yy_location is None:
            # No marker position means impossible or start with large heuristic.
            # But to keep heuristic finite, assume large number.
            v0yy_location = ""

        total_cost = 0
        for obj in self.goal_objects:
            # Check predicates for this obj in current state:
            has_guby8wxu = f"(guby8wxu {obj})" in state
            if has_guby8wxu:
                # This object done, no cost
                continue

            has_eacj8ftl = f"(eacj8ftl {obj})" in state

            # Find locations associated from static info (note: sets can be empty)
            pxch_locs = self.obj_to_pxch7pi8_locs.get(obj, set())
            pahb_locs = self.obj_to_pahbmn6t_locs.get(obj, set())

            # Candidate heuristic cost parts for this object:
            # 1) Movement of (v0yy74n7) to necessary location(s)
            # 2) One or two domain actions per object to reach goal

            # Choose representative locations (if multiple, pick any; choice affects heuristic but for efficiency pick arbitrary)
            # We try to minimize estimates, pick the closest location of these sets from current v0yy marker location.

            # If object only eacj8ftl: just move marker to pahbmn6t and perform jg64qouf
            if has_eacj8ftl:
                if pahb_locs:
                    # minimum movement cost from current marker location to any pahb location + 1 action (jg64qouf)
                    dist = min(self.shortest_path(v0yy_location, loc) for loc in pahb_locs)
                    obj_cost = dist + 1
                else:
                    # No pahb location known — just cost 1 for action (less informed)
                    obj_cost = 1
                total_cost += obj_cost
                continue

            # Else, object is presumably still in pxch7pi8 state or unknown:
            # Need first move (v0yy74n7) to pxch location and apply (makpz2tc)
            # Then move marker to pahb location and apply (jg64qouf)
            # Each action costs 1.
            cost_makpz2tc = 1
            cost_jg64qouf = 1

            # Pick the closest pxch7pi8 location (marker move cost)
            if pxch_locs:
                dist_to_pxch = min(self.shortest_path(v0yy_location, loc) for loc in pxch_locs)
                # Afterwards marker moves from pxch location to pahb location (if different)
                if pahb_locs:
                    # Dist from any pxch to any pahb location:
                    dist_pxch_to_pahb = min(
                        self.shortest_path(pxch_loc, pahb_loc)
                        for pxch_loc in pxch_locs
                        for pahb_loc in pahb_locs
                    )
                else:
                    # No pahb location known: no move needed or cost 0
                    dist_pxch_to_pahb = 0
            else:
                # No pxch location known: assume 0 cost move and 1 action for makpz2tc
                dist_to_pxch = 0
                dist_pxch_to_pahb = 0

            # Total cost for this object:
            obj_cost = dist_to_pxch + cost_makpz2tc + dist_pxch_to_pahb + cost_jg64qouf
            total_cost += obj_cost

        return total_cost
