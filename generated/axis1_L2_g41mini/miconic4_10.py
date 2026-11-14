from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact (without parentheses)."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Wildcards '*' are supported.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary
    -------
    This heuristic estimates the number of actions required to achieve the goal
    conditions specifying `(guby8wxu ?x)` predicates true for all goal objects.
    The domain involves objects of two types: `jxhpkqr5` and `ql6b2h31` with relations
    about positions `(v0yy74n7 ?loc)`, binary connections `(pvhp8m0p ?loc1 ?loc2)`,
    and tracking object states via `(pxch7pi8 ?obj ?loc)`, `(pahbmn6t ?obj ?loc)`,
    `(eacj8ftl ?obj)`, and `(guby8wxu ?obj)`.

    The heuristic focuses on how many actions it minimally takes to transform all
    goal `jxhpkqr5` objects from their current state through intermediate states until
    they satisfy `(guby8wxu ?obj)`.

    Assumptions
    -----------
    - To achieve `(guby8wxu ?obj)`, each `jxhpkqr5` object must:
      1) Start with `(pxch7pi8 ?obj ?loc1)` for some location `?loc1`,
      2) Potentially move its association through locations connected by `(pvhp8m0p)`,
      3) Update its state through intermediate predicates `(eacj8ftl ?obj)`,
      4) Eventually reach `(guby8wxu ?obj)` which indicates goal satisfaction for that object.

    - The location connectivity graph `(pvhp8m0p)` is static and undirected:
      moves between connected locations can be done by actions `jt9wnwgs` or `v90zoaro`
      changing `(v0yy74n7 ?loc)` predicate from one location to a connected one.

    - Each goal requires:
      - Moving the “marker” `(v0yy74n7 ?loc)` through locations along a path to the
        location associated with the `jxhpkqr5` object,
      - Switching `(pxch7pi8)` and `(pahbmn6t)` predicates on the object to enable it
        to reach `(guby8wxu)` via actions `makpz2tc` and `jg64qouf`.

    - The heuristic ignores reordering or concurrency optimizations and assumes:
      - One object marker handled at a time,
      - The shortest path movement cost on location graph suffices as a main cost,
      - One transition of object state per object before achieving goal.

    Heuristic Initialization
    ------------------------
    - Extract the set of all goal objects of type `jxhpkqr5` that must satisfy `(guby8wxu ?obj)`.
    - Extract all static `(pvhp8m0p ?loc1 ?loc2)` facts to build an undirected graph of location connectivity.
    - Extract static predicates associating objects with locations for initial knowledge about the domain.
    - No dynamic knowledge of `(v0yy74n7 ?loc)` in static since it changes in states.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify the set of goal `jxhpkqr5` objects from goal facts `(guby8wxu ?obj)`.
    2. For each such object, check if already satisfied `(guby8wxu ?obj)` in the current state:
       - If yes, cost for this object is zero.
       - Otherwise estimate cost to achieve it.
    3. For each object not yet at `(guby8wxu)` state:
       - Determine the current state of the object among the predicates:
         `(pxch7pi8 ?obj ?loc)`, `(pahbmn6t ?obj ?loc)`, `(eacj8ftl ?obj)`.
         Exactly one of these predicates will hold.
       - Find the relevant location for the object (location parameter of pxch7pi8 or pahbmn6t).
       - Find the current location of the `(v0yy74n7 ?loc)` marker in the state.
       - Compute shortest path length between `(v0yy74n7 ?loc)` location and the object's location in the connectivity graph.
         This estimates how many move actions (`jt9wnwgs` or `v90zoaro`) are needed.
       - Add additional fixed costs for actions needed to switch object states through:
         - From `(pxch7pi8 ?obj ?loc)` to `(eacj8ftl ?obj)` requires 1 action (makpz2tc).
         - From `(eacj8ftl ?obj)` to `(guby8wxu ?obj)` requires 1 action (jg64qouf).
         - If the object is `(pahbmn6t ?obj ?loc)` it needs at least to toggle `(pxch7pi8)` predicate first.
           We approximate by adding both makpz2tc and jg64qouf actions plus the marker movements to object location.
       - Summation of the shortest path moves plus 1 or 2 state-changing actions gives estimation per object.
    4. The heuristic value is the sum of all objects' estimated costs.
    5. If goal is reached, heuristic is zero.

    This heuristic offers a compromise between accuracy and computational efficiency:
    - It carefully computes shortest paths once per call.
    - Uses static connectivity for fast graph traversal.
    - Estimates minimal necessary actions ignoring concurrency and orderings.
    """

    def __init__(self, task):
        """Initialize heuristic, extract goals and static connectivity graph."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal objects that must satisfy (guby8wxu ?obj)
        self.goal_objects = set()
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == "guby8wxu" and len(parts) == 2:
                self.goal_objects.add(parts[1])

        # Build undirected graph of location connectivity from static (pvhp8m0p ?loc1 ?loc2)
        self.connectivity = {}
        for fact in self.static:
            if match(fact, "pvhp8m0p", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.connectivity.setdefault(loc1, set()).add(loc2)
                self.connectivity.setdefault(loc2, set()).add(loc1)

    def shortest_path_length(self, start, goal):
        """Compute shortest path length between two locations in the undirected graph.

        Return integer length, or large number if no path found.
        """
        if start == goal:
            return 0
        if start not in self.connectivity or goal not in self.connectivity:
            return 10000  # Large cost if locations unknown

        from collections import deque

        visited = set([start])
        queue = deque([(start, 0)])

        while queue:
            current, dist = queue.popleft()
            for neighbor in self.connectivity.get(current, []):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return 10000  # No path found: very large cost

    def __call__(self, node):
        """Estimate cost to reach all goal (guby8wxu ?obj) predicates."""
        state = node.state

        # If goal is reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # Find current location of the marker (v0yy74n7 ?loc).
        marker_loc = None
        for fact in state:
            if match(fact, "v0yy74n7", "*"):
                marker_loc = get_parts(fact)[1]
                break
        if marker_loc is None:
            # Marker location unknown - large underestimate, but cannot proceed reasonably.
            return 10000

        total_cost = 0

        for obj in self.goal_objects:
            # If object already at goal, no cost.
            if f"(guby8wxu {obj})" in state:
                continue

            # Determine object's state and location
            # The object can be in exactly one of these predicates:
            # (pxch7pi8 ?obj ?loc)
            # (pahbmn6t ?obj ?loc)
            # (eacj8ftl ?obj)
            obj_pxch = None
            obj_pahbmn6t = None
            obj_eacj = None

            # Search for associated facts in current state
            for fact in state:
                if match(fact, "pxch7pi8", obj, "*"):
                    obj_pxch = get_parts(fact)[2]
                    break
            if obj_pxch is None:
                for fact in state:
                    if match(fact, "pahbmn6t", obj, "*"):
                        obj_pahbmn6t = get_parts(fact)[2]
                        break
            if obj_pxch is None and obj_pahbmn6t is None:
                # Check if (eacj8ftl ?obj)
                if f"(eacj8ftl {obj})" in state:
                    obj_eacj = True

            # Estimate cost to get marker to object's location based on state
            # and the number of actions to make state changes to reach (guby8wxu).
            cost = 0

            # Number of move actions for the marker from marker_loc to object's location:
            # Determine object position for marker moves:
            if obj_pxch is not None:
                # Object at (pxch7pi8 ?obj ?loc)
                path_moves = self.shortest_path_length(marker_loc, obj_pxch)
                # Action makpz2tc to switch pxch7pi8 to eacj8ftl
                # Then jg64qouf to switch eacj8ftl to guby8wxu.
                cost = path_moves + 2
            elif obj_pahbmn6t is not None:
                # Object at (pahbmn6t ?obj ?loc)
                path_moves = self.shortest_path_length(marker_loc, obj_pahbmn6t)
                # Must first do makpz2tc (to get eacj8ftl) and jg64qouf (to get guby8wxu)
                # but pahbmn6t is a different predicate indicating a different state,
                # so assume we first must switch it to pxch7pi8 or eacj8ftl by one action.
                # Approximating cost as 3 state change actions to account extra toggle.
                cost = path_moves + 3
            elif obj_eacj:
                # Object already at eacj8ftl ?obj (no location)
                # Marker must move to some location associated with object to apply jg64qouf
                # However, no location in eacj8ftl, approximate as 0 moves + 1 action to get guby8wxu
                # But marker must be at correct location - find location from static pahbmn6t or pxch7pi8 facts
                obj_loc = None
                # Find location from static facts where this object appears in pxch7pi8 or pahbmn6t to approximate marker location needed.
                for fact in self.static:
                    if match(fact, "pxch7pi8", obj, "*"):
                        obj_loc = get_parts(fact)[2]
                        break
                if obj_loc is None:
                    for fact in self.static:
                        if match(fact, "pahbmn6t", obj, "*"):
                            obj_loc = get_parts(fact)[2]
                            break
                if obj_loc is None:
                    # No known location, assume zero moves
                    path_moves = 0
                else:
                    path_moves = self.shortest_path_length(marker_loc, obj_loc)
                # One jg64qouf action to reach goal
                cost = path_moves + 1
            else:
                # Object not found in expected predicates, assume large cost
                cost = 10000

            total_cost += cost

        return total_cost
