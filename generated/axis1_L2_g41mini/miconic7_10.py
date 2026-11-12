from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(pxch7pi8 ps0y2bwz qklmztf3)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the 'secret_domain' domain.

    Summary
    -------
    This heuristic estimates the number of actions needed to achieve the goal state
    where certain objects of type `jxhpkqr5` have the predicate `(guby8wxu ?obj)` 
    true. The domain involves:
      - Movement of some "ql6b2h31" typed objects maintaining the predicate `(v0yy74n7 ?location)`.
      - Transition of `(eacj8ftl ?obj)` to `(guby8wxu ?obj)` for `jxhpkqr5` objects,
        which requires specific conditions on related objects.
      - Two layered dependencies related to predicates `(pxch7pi8 ?jxhpkqr5 ?ql6b2h31)`,
        `(pahbmn6t ?jxhpkqr5 ?ql6b2h31)`, and a connectivity relation `(pvhp8m0p ?ql6b2h31 ?ql6b2h31)`.
      
    Assumptions
    -----------
    - The main challenge is transitioning `jxhpkqr5` objects from `(eacj8ftl)` to `(guby8wxu)`.
    - To achieve `(guby8wxu ?obj)`, an object must be linked via `(pahbmn6t ?obj ?loc)` with a `loc` where `(v0yy74n7 loc)` holds.
    - To transition from `(pxch7pi8 ?obj ?loc1)` to `(eacj8ftl ?obj)`, the object must be at `loc1`
      and `(v0yy74n7 loc1)` must hold; but since `(pxch7pi8)` facts are deleted and `(eacj8ftl)` added,
      the object must be "moved" accordingly.
    - The domain's `v0yy74n7` predicate moves among `ql6b2h31` objects connected by `(pvhp8m0p)` links, 
      via two kinds of movement actions that transfer `(v0yy74n7)` from one location to another.
    - Each `jxhpkqr5` object "depends" on some `ql6b2h31` locations to satisfy the goal.
    - The static `(pvhp8m0p)` facts define a directed graph of locations; moving the `(v0yy74n7)` predicate is like moving a token along this graph.
    - The heuristic ignores subtle effects of possible orderings and counts minimal estimated steps via shortest path distances and logical state changes.

    Heuristic Initialization
    ------------------------
    - Extract the goal objects that require `(guby8wxu ?obj)`.
    - Build mappings from `jxhpkqr5` objects to their connected `ql6b2h31` locations via `(pahbmn6t)` and `(pxch7pi8)`.
    - Extract all `(pvhp8m0p)` edges and build a directed graph over `ql6b2h31` locations for shortest-path distance calculations.
    - Precompute all pairs shortest paths for the graph (using BFS), or store adjacency for on-demand BFS to estimate shortest moves.
    - Remembers the static `(pahbmn6t)`, `(pxch7pi8)`, `(pvhp8m0p)` facts for efficient heuristic computations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a state:
    1. Check whether the goal `(guby8wxu ?obj)` facts hold for all goal objects; if yes, heuristic=0.
    2. Otherwise, for each goal `jxhpkqr5` object:
       a. If `(guby8wxu obj)` already true -> 0 cost
       b. Else if `(eacj8ftl obj)` true -> estimate cost to get `(guby8wxu obj)` by using one `jg64qouf` action
          (after ensuring `(v0yy74n7)` is at location linked by `(pahbmn6t obj location)`)
       c. Else if `(pxch7pi8 obj loc)` true for some loc:
          i. Need to bring `(v0yy74n7)` from current location to `loc`.
          ii. Estimate shortest path distance from `(v0yy74n7 current_loc)` to `loc` via `(pvhp8m0p)` graph.
          iii. Then perform `makpz2tc` action to convert `(pxch7pi8 obj loc)` -> `(eacj8ftl obj)`.
          iv. Then estimate from (b) step: `eacj8ftl` -> `guby8wxu`.
       d. If no `(pxch7pi8 obj loc)` known, heuristic assumes worst case (some large number).
    3. For the location of `(v0yy74n7)`, find it in the state; there is exactly one location holding `(v0yy74n7 loc)`.
    4. Sum the minimal estimated steps over all goal objects to get heuristic.
       We sum because objects likely need separate processes (domain and actions suggest sequences).
    5. Movement along edges counts one action; each operator contributes roughly one action per step.
    6. The three phases per object roughly:
       - Move token `(v0yy74n7)` along shortest path (distance) from current to required location for `(pxch7pi8)` or `(pahbmn6t)` link.
       - Apply needed action to convert between `(pxch7pi8)` → `(eacj8ftl)` → `(guby8wxu)`.
       - Each such conversion is one action.
    7. Sum these steps, but since the heuristic is not required to be admissible, count direct sums.

    This heuristic should be efficiently computable by:
    - Precomputing adjacency map from static `(pvhp8m0p)`.
    - Caching shortest path distances using BFS per demand.
    - Quickly matching relevant predicates for each `jxhpkqr5` object in the state.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract (pvhp8m0p) as adjacency graph from static facts: directed edges (from -> to)
        self.adjacency = {}
        for fact in self.static:
            if match(fact, "pvhp8m0p", "*", "*"):
                _, from_node, to_node = get_parts(fact)
                if from_node not in self.adjacency:
                    self.adjacency[from_node] = set()
                self.adjacency[from_node].add(to_node)

        # Extract the set of all ql6b2h31 nodes from static and goals
        ql6b2h31_nodes = set()
        # from adjacency keys and values
        for u in self.adjacency:
            ql6b2h31_nodes.add(u)
            for v in self.adjacency[u]:
                ql6b2h31_nodes.add(v)
        self.ql6b2h31_nodes = ql6b2h31_nodes

        # Extract (pahbmn6t ?jxhpkqr5 ?ql6b2h31) mappings from static for fast lookup:
        # maps: object -> set of linked locations
        self.obj_to_pahbmn6t = {}
        for fact in self.static:
            if match(fact, "pahbmn6t", "*", "*"):
                _, obj, loc = get_parts(fact)
                self.obj_to_pahbmn6t.setdefault(obj, set()).add(loc)

        # Extract (pxch7pi8 ?jxhpkqr5 ?ql6b2h31) mappings from static for heuristic reference:
        # This is used as the "current" location predicate for obj, but actually dynamic
        # We will check the state for pxch7pi8 to know current placement.
        self.obj_to_pxch7pi8 = {}
        for fact in self.static:
            if match(fact, "pxch7pi8", "*", "*"):
                _, obj, loc = get_parts(fact)
                self.obj_to_pxch7pi8.setdefault(obj, set()).add(loc)

        # Cache distances computed for pairs: (start, goal) -> distance
        self._distance_cache = {}

    def shortest_path_distance(self, start, goal):
        """
        Compute the shortest path distance from start to goal in the (pvhp8m0p) graph using BFS.
        Returns a large number if no path exists.
        """
        if start == goal:
            return 0
        cache_key = (start, goal)
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]

        if start not in self.ql6b2h31_nodes or goal not in self.ql6b2h31_nodes:
            # Impossible locations, return large cost
            self._distance_cache[cache_key] = 9999
            return 9999

        from collections import deque

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            current, dist = queue.popleft()
            neighbors = self.adjacency.get(current, set())
            for nbr in neighbors:
                if nbr == goal:
                    self._distance_cache[cache_key] = dist + 1
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
        self._distance_cache[cache_key] = 9999
        return 9999

    def __call__(self, node):
        state = node.state

        # If goal already satisfied, heuristic is 0
        if self.goals <= state:
            return 0

        # Find the location where (v0yy74n7 ?) holds: exactly one per state
        v0_loc = None
        for fact in state:
            if match(fact, "v0yy74n7", "*"):
                _, loc = get_parts(fact)
                v0_loc = loc
                break
        if v0_loc is None:
            # No location with (v0yy74n7), possibly invalid state; treat pessimistically
            v0_loc = None

        total_estimated_cost = 0

        # Extract all relevant predicates for faster lookup by object:
        # (guby8wxu ?obj) - goal achieved for that object
        guby8wxu_objs = {get_parts(fact)[1] for fact in state if match(fact, "guby8wxu", "*")}
        # (eacj8ftl ?obj) - intermediate state
        eacj8ftl_objs = {get_parts(fact)[1] for fact in state if match(fact, "eacj8ftl", "*")}
        # (pxch7pi8 ?obj ?loc) - initial location assignments of objects
        pxch7pi8_facts = {}
        for fact in state:
            if match(fact, "pxch7pi8", "*", "*"):
                _, obj, loc = get_parts(fact)
                pxch7pi8_facts[obj] = loc

        # Goal demands (from goals, objects)
        # Relevant objects: those mentioned in goal predicate (guby8wxu ?obj)
        goal_objs = []
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            pred = parts[0]
            if pred == "guby8wxu":
                goal_objs.append(parts[1])

        # Compute heuristic estimates per goal object
        for obj in goal_objs:
            if obj in guby8wxu_objs:
                # This goal object already achieved: no cost
                continue

            # To reach (guby8wxu obj):
            # Needs to have (eacj8ftl obj) true and v0yy74n7 at linked location(pahbmn6t obj ?loc)
            # Then one jg64qouf action can achieve goal.
            if obj in eacj8ftl_objs:
                # Check for linked locations of obj by (pahbmn6t obj ?loc)
                linked_locs = self.obj_to_pahbmn6t.get(obj, set())
                min_move_cost = 9999
                for loc in linked_locs:
                    if v0_loc is None:
                        # No v0yy74n7 location known -> pessimistic estimate
                        dist = 9999
                    else:
                        dist = self.shortest_path_distance(v0_loc, loc)
                    # We add 1 for the final jg64qouf action
                    total_cost = dist + 1
                    if total_cost < min_move_cost:
                        min_move_cost = total_cost
                # Add minimal cost for this object
                total_estimated_cost += min_move_cost
                continue

            # Otherwise, obj must be converted from (pxch7pi8 obj loc) to (eacj8ftl obj),
            # then from eacj8ftl to guby8wxu.
            if obj in pxch7pi8_facts:
                pxch_loc = pxch7pi8_facts[obj]
                # Movement cost to bring v0yy74n7 from current location to pxch_loc
                if v0_loc is None:
                    dist_to_pxch = 9999
                else:
                    dist_to_pxch = self.shortest_path_distance(v0_loc, pxch_loc)

                # One action for makpz2tc (convert pxch7pi8->eacj8ftl)
                # Then estimate from eacj8ftl to guby8wxu as above:
                linked_locs = self.obj_to_pahbmn6t.get(obj, set())
                min_dist_eacj = 9999
                for loc in linked_locs:
                    dist = self.shortest_path_distance(pxch_loc, loc)
                    # +1 for the jg64qouf action
                    total_cost = dist + 1
                    if total_cost < min_dist_eacj:
                        min_dist_eacj = total_cost

                # Total cost includes moving v0yy74n7 from current to pxch loc,
                # one makpz2tc action (pxch->eacj),
                # movement of v0yy74n7 from pxch_loc to pahbmn6t loc,
                # and one jg64qouf action (eacj->guby)
                total_cost_for_obj = dist_to_pxch + 1 + min_dist_eacj
                total_estimated_cost += total_cost_for_obj
                continue

            # If obj is neither in eacj8ftl nor pxch7pi8 in state, underestimate as requiring 
            # 3 steps minimal as a fallback (worst case: move + makpz2tc + jg64qouf)
            total_estimated_cost += 3

        # Return final estimate - heuristic=0 only at goals; positive finite otherwise
        return total_estimated_cost
