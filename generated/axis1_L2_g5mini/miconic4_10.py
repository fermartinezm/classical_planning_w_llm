from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return components of a PDDL fact string by removing parentheses and splitting."""
    return fact.strip()[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (unkhrp2l) PDDL domain.

    The domain contains:
    - A single movable token represented by predicate (v0yy74n7 ?loc).
    - Directed edges between locations represented by (pvhp8m0p ?from ?to).
    - Objects of type jxhpkqr5 that initially may be at some location with (pxch7pi8 ?obj ?loc).
    - A static mapping of desired delivery locations via (pahbmn6t ?obj ?target).
    - Two actions to finish an object's task:
        - makpz2tc: when token at object's pickup location and (pxch7pi8 ?obj ?loc),
                    consumes pxch7pi8 and produces eacj8ftl for the object.
        - jg64qouf: when token at object's target location and eacj8ftl holds,
                    consumes eacj8ftl and produces guby8wxu (the goal fact).
    - Movement actions jt9wnwgs / v90zoaro move the token along directed pvhp8m0p edges.

    The heuristic estimates the number of actions required to achieve all goal facts
    (guby8wxu ?obj) that are not yet true in the current state. The estimate assumes
    the token can sequentially serve objects and computes a simple sum of per-object
    minimal action counts (movement + the required two operator applications). It
    ignores sequencing optimizations between different objects (i.e., it is optimistic
    about parallelism but conservative regarding reusing intermediate states), making it
    informative and efficiently computable while not necessarily admissible.

    Assumptions
    -----------
    - The static facts include pvhp8m0p edges (directed) and pahbmn6t mappings object->target.
    - Each object's pickup fact is represented by dynamic predicate pxch7pi8 in the state.
    - If an object already has eacj8ftl in the current state, the pickup step is already done.
    - Movement along every pvhp8m0p edge costs 1 action; makpz2tc and jg64qouf each cost 1 action.
    - If required facts are missing (e.g., no pickup pxch7pi8 and no eacj8ftl) then the object
      cannot be completed and the heuristic returns a large cost for that object.

    Heuristic Initialization
    ------------------------
    - Extract the directed graph of locations from static pvhp8m0p facts.
    - Precompute single-source shortest-path distances (BFS) between all locations present
      in the static graph (for efficient repeated distance queries).
    - Extract the mapping from object -> target location using static pahbmn6t facts.
    - Store the goal facts (task.goals) for quick membership testing.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state):
    1. If all goal facts are already true in the state's fact set, return 0.
    2. Locate the current token location by searching for a fact with predicate v0yy74n7.
       If no token fact is present, treat the state as (likely) unsolvable and return a
       large finite number.
    3. Build quick-access sets for dynamic facts in the state:
       - pxch_map: mapping from object -> pickup location(s) (pxch7pi8 facts)
       - eacj_set: set of objects that already have eacj8ftl in the state
    4. For each object that is required by the goals but not yet having guby8wxu in the
       current state, compute a minimal estimated cost:
       - If the object already has eacj8ftl:
           cost_obj = dist(token_loc, target_loc) + 1  # move token to target + jg64qouf
       - Else if the object has a pickup location pickup_loc (pxch7pi8 present):
           cost_obj = dist(token_loc, pickup_loc) + 1  # move to pickup and makpz2tc
                      + dist(pickup_loc, target_loc) + 1  # move to target and jg64qouf
           If there are multiple pxch7pi8 facts for same object, choose the minimal cost route.
       - Else (no eacj8ftl and no pxch7pi8): assign a large finite penalty for that object.
       - Distances are taken from the precomputed BFS distances. If a required distance is
         unreachable, assign a large penalty for that object.
    5. Sum the per-object costs and return the sum as the heuristic value.

    The heuristic is designed to be efficient (BFS precomputation once in the constructor,
    constant-time distance lookups during evaluation) and informative for greedy best-first
    search in this domain.
    """

    def __init__(self, task):
        # store goals
        self.goals = task.goals

        # static facts
        static_facts = task.static

        # Build directed adjacency list from static pvhp8m0p facts (edges between locations).
        self.adjacency = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "pvhp8m0p":
                src, dst = parts[1], parts[2]
                locations.add(src)
                locations.add(dst)
                self.adjacency.setdefault(src, set()).add(dst)

        # Ensure adjacency entries exist for isolated nodes found in other static facts.
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "pahbmn6t":
                # object -> target location
                locations.add(parts[2])
                self.adjacency.setdefault(parts[2], set())

        # Precompute BFS distances from every known location to every other known location.
        # distances[src][dst] = integer distance or None if unreachable
        self.distances = {}
        for src in list(self.adjacency.keys()):
            dist = {node: None for node in self.adjacency.keys()}
            # BFS
            dq = deque()
            dist[src] = 0
            dq.append(src)
            while dq:
                u = dq.popleft()
                for v in self.adjacency.get(u, ()):
                    if dist.get(v) is None:
                        dist[v] = dist[u] + 1
                        dq.append(v)
            self.distances[src] = dist

        # Map objects to their static target locations from pahbmn6t
        self.target_for_object = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "pahbmn6t":
                obj, target = parts[1], parts[2]
                self.target_for_object[obj] = target

        # Large finite penalty for unreachable/unsolvable subproblems
        self.PENALTY = 100000

    def __call__(self, node):
        state = node.state

        # If all goals already true -> heuristic is 0
        if self.goals <= state:
            return 0

        # find token location
        token_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "v0yy74n7":
                token_loc = parts[1]
                break
        if token_loc is None:
            # No token in state -> large penalty
            return self.PENALTY

        # Build dynamic maps: pxch7pi8 (pickups) and eacj8ftl (activated)
        pxch_map = {}  # obj -> list of pickup locations (usually one)
        eacj_set = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "pxch7pi8":
                obj, loc = parts[1], parts[2]
                pxch_map.setdefault(obj, []).append(loc)
            elif parts[0] == "eacj8ftl":
                eacj_set.add(parts[1])

        total_cost = 0

        # Extract goal objects that must have guby8wxu true
        goal_objects = []
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "guby8wxu":
                goal_objects.append(parts[1])

        # For each goal object not yet satisfied, estimate cost
        for obj in goal_objects:
            goal_fact = f"(guby8wxu {obj})"
            if goal_fact in state:
                continue  # already achieved

            # Determine the target location for this object
            target_loc = self.target_for_object.get(obj)
            if target_loc is None:
                # No static target -> cannot determine how to satisfy; penalize heavily
                total_cost += self.PENALTY
                continue

            # helper to get distance between two locations
            def dist(a, b):
                if a == b:
                    return 0
                if a not in self.distances:
                    return None
                return self.distances[a].get(b)

            # If object already has eacj8ftl, only need to move token to target and apply jg64qouf
            if obj in eacj_set:
                d = dist(token_loc, target_loc)
                if d is None:
                    total_cost += self.PENALTY
                else:
                    total_cost += d + 1  # moves + jg64qouf
                continue

            # Otherwise we need a pickup (pxch7pi8) followed by delivery
            pickup_locs = pxch_map.get(obj)
            if not pickup_locs:
                # No pickup and no eacj8ftl -> cannot fulfill this goal (penalty)
                total_cost += self.PENALTY
                continue

            # choose the pickup location that yields minimal cost: token->pickup + 1 + pickup->target + 1
            best = None
            for pickup in pickup_locs:
                d1 = dist(token_loc, pickup)
                d2 = dist(pickup, target_loc)
                if d1 is None or d2 is None:
                    continue
                cost_obj = d1 + 1 + d2 + 1  # move to pickup, makpz2tc, move to target, jg64qouf
                if best is None or cost_obj < best:
                    best = cost_obj
            if best is None:
                total_cost += self.PENALTY
            else:
                total_cost += best

        return total_cost