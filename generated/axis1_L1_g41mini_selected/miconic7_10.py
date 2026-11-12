from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing surrounding parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Wildcards '*' are supported in args.

    - fact: A fact string like "(predicate obj1 obj2)".
    - args: Pattern to match, e.g., "kut2ksnk", "*", "f2".
    Returns True if matches, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (ud3uf11z).

    Summary:
    This heuristic estimates the number of actions required to achieve the goal conditions 
    (all objects of type nxbd1hc3 hold the predicate (a7sb56om ?object)) from a given state.
    The domain involves moving "nxbd1hc3" objects between "t4rdxh32" locations marked by (omggbn9z),
    with predicates (kut2ksnk) and (klw11kjf) linking nxbd1hc3 objects to locations. Objects can become
    "f2xo7hb6" (some intermediate state), then "a7sb56om" (final goal state).

    Assumptions:
    - Each nxbd1hc3 object must eventually satisfy (a7sb56om ?object).
    - To achieve (a7sb56om ?object), the object must be first in state (f2xo7hb6 ?object),
      then transitioned by actions requiring specific location predicates.
    - The (omggbn9z ?loc) predicate marks the current "active" location.
    - Actions qs8pdrqj and i0veo5kv move the "omggbn9z" marker along the graph formed by (acovodrf loc1 loc2).
    - At most one location holds (omggbn9z) at a time.
    - The heuristic estimates cost primarily by counting unsatisfied subgoals and approximating
      cost to switch locations and perform object state changes.

    Heuristic Initialization:
    - Extract the set of all nxbd1hc3 objects mentioned in static predicates.
    - Extract (acovodrf loc1 loc2) edges forming a bidirectional graph of t4rdxh32 locations.
    - Extract static predicates (klw11kjf ?obj ?loc) and (kut2ksnk ?obj ?loc).
    - Compute a graph connecting t4rdxh32 locations to approximate shortest path distances between them.
    - Extract goal conditions (a7sb56om ?obj) to know which objects must reach that predicate.
    - Cache data structures for quick lookup.

    Step-By-Step Thinking for Computing Heuristic:
    1. If the state satisfies goals, return 0.

    2. Identify currently "omggbn9z" location (unique).

    3. For each nxbd1hc3 object that is part of the goal (i.e. must have a7sb56om):
        - Check if (a7sb56om obj) achieved in state: no cost.
        - Else if (f2xo7hb6 obj) holds:
            The object is midway; must apply action y7nik9q2 at location L where clause applies:
                Preconditions: (omggbn9z L), (klw11kjf obj L), and (f2xo7hb6 obj).
                Effect: remove (f2xo7hb6 obj), add (a7sb56om obj).
            Estimate cost as:
                - Move from current omggbn9z location to location L using shortest path edges.
                - Then 1 action y7nik9q2 at location L.

        - Else (object neither f2xo7hb6 nor a7sb56om):
            The object must first be moved to (f2xo7hb6) state via action diwlmgg9:
                Preconditions: (omggbn9z L), (kut2ksnk obj L).
                Effect: add (f2xo7hb6 obj).
            Then as above, move to klw11kjf location for y7nik9q2 action.

            For heuristic:
            - Find locations L_kut where (kut2ksnk obj L_kut) holds.
            - Find locations L_klw where (klw11kjf obj L_klw) holds.

            The minimal cost for the object is:
            min over L_kut and L_klw of:
              (distance from current omggbn9z to L_kut) + 1 (diwlmgg9) +
              (distance from L_kut to L_klw) + 1 (y7nik9q2)

    4. Since multiple objects share locations and the omggbn9z marker can only be at one location at a time,
       a greedy summation over the objects' individual minimal costs would guide the search efficiently
       even though it overestimates total cost.

    5. Return sum of minimal estimated costs over all objects.

    Efficiency:
    - Distance computations use cached shortest paths on the static location graph.
    - Matching predicates and lookups use dictionaries for O(1) membership.
    - The heuristic is not admissible but tries to be informative to reduce expanded nodes.
    """

    def __init__(self, task):
        self.goals = task.goals  # Set of goal facts, e.g., (a7sb56om p1)
        self.static = task.static

        # Extract all objects of type nxbd1hc3 and t4rdxh32 from static facts.
        # Since typing info is unavailable here, extract from static predicates:
        # We'll extract objects by scanning relevant predicates:
        self.nxbd1hc3_objects = set()
        self.t4rdxh32_locations = set()
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            args = parts[1:]
            if pred in ("kut2ksnk", "klw11kjf"):
                # Both have signature (?praz5 - nxbd1hc3, ?w3okg - t4rdxh32)
                self.nxbd1hc3_objects.add(args[0])
                self.t4rdxh32_locations.add(args[1])
            elif pred == "acovodrf":
                self.t4rdxh32_locations.update(args)
            elif pred == "omggbn9z":
                self.t4rdxh32_locations.add(args[0])

        # Build a graph (bidirectional) of t4rdxh32 locations from acovodrf predicate
        self.location_graph = {loc: set() for loc in self.t4rdxh32_locations}
        for fact in self.static:
            if match(fact, "acovodrf", "*", "*"):
                loc1, loc2 = get_parts(fact)[1:3]
                self.location_graph[loc1].add(loc2)
                # The domain apparently models edges as directional acovodrf from loc1 to loc2,
                # but we include reverse edges because there are two separate actions qs8pdrqj and i0veo5kv
                # that move the (omggbn9z) marker in both directions
                self.location_graph[loc2].add(loc1)

        # Precompute shortest path distances between locations using Floyd-Warshall or BFS:
        # Use BFS from each location since graph is typically sparse.

        # To speed up, distances stored as: dict: {loc1: {loc2: dist,...}, ...}
        self.distance_cache = {}

        def bfs(start):
            visited = {start: 0}
            queue = [start]
            while queue:
                current = queue.pop(0)
                for neigh in self.location_graph[current]:
                    if neigh not in visited:
                        visited[neigh] = visited[current] + 1
                        queue.append(neigh)
            return visited

        for loc in self.t4rdxh32_locations:
            self.distance_cache[loc] = bfs(loc)

        # Extract for each nxbd1hc3 object their static predicate locations (kut2ksnk, klw11kjf)
        self.obj_kut2ksnk = {}  # obj -> set of locations
        self.obj_klw11kjf = {}  # obj -> set of locations
        for obj in self.nxbd1hc3_objects:
            self.obj_kut2ksnk[obj] = set()
            self.obj_klw11kjf[obj] = set()

        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "kut2ksnk":
                obj, loc = parts[1], parts[2]
                if obj in self.nxbd1hc3_objects:
                    self.obj_kut2ksnk[obj].add(loc)
            elif pred == "klw11kjf":
                obj, loc = parts[1], parts[2]
                if obj in self.nxbd1hc3_objects:
                    self.obj_klw11kjf[obj].add(loc)

    def __call__(self, node):
        state = node.state

        # If goal reached, heuristic is 0.
        if self.goals <= state:
            return 0

        # Find current location holding (omggbn9z loc)
        omggbn9z_locations = [get_parts(fact)[1] for fact in state if match(fact, "omggbn9z", "*")]
        if not omggbn9z_locations:
            # No active location => large cost to recover?
            # But domain presumably always has one; fallback:
            current_loc = None
        else:
            current_loc = omggbn9z_locations[0]

        total_cost = 0

        # Collect all objects needed to have (a7sb56om obj) to satisfy goal:
        goal_objs = set()
        for goal_fact in self.goals:
            if match(goal_fact, "a7sb56om", "*"):
                goal_objs.add(get_parts(goal_fact)[1])

        # Precompute object predicates in current state
        has_f2xo7hb6 = set(get_parts(f)[1] for f in state if match(f, "f2xo7hb6", "*"))
        has_a7sb56om = set(get_parts(f)[1] for f in state if match(f, "a7sb56om", "*"))
        # kut2ksnk and klw11kjf in current state: 
        # These may be changed by diwlmgg9 action and by y7nik9q2 action
        # But for heuristic, trust static knowledge for their locations.
        # So no need to check them dynamically.

        # Define cost for moving (omggbn9z) between locations: 1 action per step
        # Actions that act on object: diwlmgg9 and y7nik9q2 count as 1 action each.

        for obj in goal_objs:
            if obj in has_a7sb56om:
                # Object finished: no cost
                continue

            if obj in has_f2xo7hb6:
                # Need to do y7nik9q2 at some location L where klw11kjf(obj, L) and currently omggbn9z = L
                # Estimate cost: move omggbn9z from current_loc to L + 1 (y7nik9q2)

                target_locs = self.obj_klw11kjf[obj]
                # Defensive: if empty target_locs, fallback cost high
                if not target_locs:
                    # no known klw11kjf location, large arbitrary cost
                    total_cost += 10
                    continue

                min_cost = float("inf")
                for loc in target_locs:
                    dist = self.distance_cache[current_loc].get(loc, float("inf")) if current_loc else float("inf")
                    cost = dist + 1  # 1 for the y7nik9q2 action
                    if cost < min_cost:
                        min_cost = cost
                if min_cost == float("inf"):
                    # no path found: assign high cost
                    min_cost = 10
                total_cost += min_cost

            else:
                # Need first to do diwlmgg9 at some L_kut (kut2ksnk) location,
                # then move to L_klw (klw11kjf) location for y7nik9q2.
                L_kut = self.obj_kut2ksnk[obj]
                L_klw = self.obj_klw11kjf[obj]

                # Defensive checks
                if not L_kut or not L_klw:
                    total_cost += 20  # large cost for incompleteness
                    continue

                min_cost = float("inf")
                for kut_loc in L_kut:
                    dist1 = self.distance_cache[current_loc].get(kut_loc, float("inf")) if current_loc else float("inf")
                    for klw_loc in L_klw:
                        dist2 = self.distance_cache[kut_loc].get(klw_loc, float("inf"))
                        cost = dist1 + 1 + dist2 + 1
                        # dist1: move to kut_loc
                        # +1: diwlmgg9 action
                        # dist2: move to klw_loc
                        # +1: y7nik9q2 action
                        if cost < min_cost:
                            min_cost = cost
                if min_cost == float("inf"):
                    min_cost = 20
                total_cost += min_cost

        return total_cost
