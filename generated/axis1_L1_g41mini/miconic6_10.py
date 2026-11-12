from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(kut2ksnk p1 f6)".
    - `args`: The expected pattern with wildcards `*` allowed.
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain (ud3uf11z).

    Summary
    -------
    This heuristic estimates the number of actions required to achieve the goals,
    specifically reaching the conditions (a7sb56om ?praz5) for all objects ?praz5 of type nxbd1hc3
    as specified in the goal. The heuristic is based on the state of each nxbd1hc3 object in regard
    to the predicates f2xo7hb6 and a7sb56om, and the location (omggbn9z) of t4rdxh32 objects.

    Assumptions
    -----------
    - The goal predicates are always of the form (a7sb56om ?praz5) for nxbd1hc3 objects.
    - Objects of type t4rdxh32 represent locations/places, with some marked by (omggbn9z ?location)
      meaning "occupied" or "current position".
    - Each nxbd1hc3 object can be in one of three main "states":
        * "ready" (a7sb56om true) => goal satisfied for that object.
        * "activated" (f2xo7hb6 true but not a7sb56om) => intermediate state requiring y7nik9q2 action to reach goal.
        * "not activated" (neither a7sb56om nor f2xo7hb6 true) => requires first diwlmgg9 action plus then y7nik9q2.
    - The actions to achieve the goal for each object combine:
        * diwlmgg9: to activate the object, requires one omggbn9z location where kut2ksnk fact holds.
        * y7nik9q2: to finalize the goal, requires the object to be klw11kjf at some omggbn9z location and that the object is f2xo7hb6.
    - Objects not yet activated require first diwlmgg9 and then y7nik9q2, so at least two actions.
    - We consider the "distance" to available omggbn9z locations by the minimal number of transitions following acovodrf facts between locations,
      to estimate how many qs8pdrqj / i0veo5kv moves might be needed to get to the correct location for actions preconditions.
    - The heuristic sums the estimated required actions for all objects, assuming they can be processed roughly independently.

    Heuristic Initialization
    ------------------------
    - Extract goal objects that must satisfy (a7sb56om ?praz5).
    - Extract static facts:
        * kut2ksnk(praz5, location) indicating places where diwlmgg9 can be applied.
        * klw11kjf(praz5, location) indicating places where y7nik9q2 applies.
        * acovodrf(from_loc, to_loc) edges defining a directed graph between t4rdxh32 locations; used to estimate shortest paths.
        * The set of all locations with (omggbn9z location).
    - Build:
        * For each nxbd1hc3 object, sets of kut2ksnk locations and klw11kjf locations.
        * A graph of location connectivity for computing shortest distances between locations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state is a goal state (all a7sb56om for goal objects true), heuristic = 0.
    2. For each goal object (nxbd1hc3) that is not yet (a7sb56om):
        - Check if it is currently (f2xo7hb6), i.e., activated.
        - Identify candidate locations where actions diwlmgg9 or y7nik9q2 can be performed.
        - Identify current locations occupied (omggbn9z) in the state.
        - Estimate minimum steps (moves) needed to reach:
          * For activation: a location L with (omggbn9z L) and (kut2ksnk object L).
            Needed for the precondition of diwlmgg9.
          * For finalization: a location L with (omggbn9z L) and (klw11kjf object L).
            Needed for the precondition of y7nik9q2.
        - To estimate distances, use shortest path over acovodrf edges between current occupied locations and required locations.
        - Estimated cost per object roughly:
          * If not activated (not f2xo7hb6): 
              cost = 
                min_dist_to_kut2ksnk_location + 1 (diwlmgg9 action) +
                min_dist_to_klw11kjf_location + 1 (y7nik9q2 action)
          * If activated:
              cost = 
                min_dist_to_klw11kjf_location + 1 (y7nik9q2 action)
    3. Since moves qs8pdrqj/i0veo5kv toggle (omggbn9z) between adjacent locations,
       add these moves' estimated costs accordingly; we assume each hop cost = 1 action.
    4. Sum these costs over all goal objects not yet achieved.
    5. Return the total sum as the heuristic estimate.

    Efficiency Notes:
    - Precompute shortest paths between all locations once in __init__ using Floyd-Warshall or BFS per location.
    - For large location sets, we use BFS from occupied locations to required locations to find minimal moves.
    - This heuristic trades admissibility for better goal distance approximation, guiding greedy best-first search efficiently.
    """

    def __init__(self, task):
        self.goals = task.goals  # goal facts: expecting (a7sb56om ?praz5)
        self.static = task.static

        # Extract all objects of type nxbd1hc3 and t4rdxh32 from static predicates:
        self.nxbd1hc3_objects = set()
        self.t4rdxh32_locations = set()

        # These used for heuristics
        # Map: object -> set of kut2ksnk locations
        self.obj_kut2ksnk = {}
        # Map: object -> set of klw11kjf locations
        self.obj_klw11kjf = {}
        # Set of all acovodrf edges as adjacency (directed graph)
        self.adjacency = {}

        # Extract objects and build adjacency graph
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == "kut2ksnk":
                obj = parts[1]
                loc = parts[2]
                self.nxbd1hc3_objects.add(obj)
                self.t4rdxh32_locations.add(loc)
                self.obj_kut2ksnk.setdefault(obj, set()).add(loc)

            elif pred == "klw11kjf":
                obj = parts[1]
                loc = parts[2]
                self.nxbd1hc3_objects.add(obj)
                self.t4rdxh32_locations.add(loc)
                self.obj_klw11kjf.setdefault(obj, set()).add(loc)

            elif pred == "acovodrf":
                loc_from = parts[1]
                loc_to = parts[2]
                self.t4rdxh32_locations.add(loc_from)
                self.t4rdxh32_locations.add(loc_to)
                self.adjacency.setdefault(loc_from, set()).add(loc_to)

        # Identify all objects that appear in goals (a7sb56om ?p)
        self.goal_objects = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "a7sb56om":
                self.goal_objects.add(parts[1])

        # Precompute shortest distances between all locations with a BFS dictionary
        # For each location, maps loc->min distance to all other locations
        self.location_distances = {}
        for loc in self.t4rdxh32_locations:
            self.location_distances[loc] = self._bfs_distances(loc)

    def _bfs_distances(self, start):
        """Compute shortest path distances in the directed graph from start to all reachable locations."""
        distances = {start: 0}
        queue = [start]
        while queue:
            current = queue.pop(0)
            dist = distances[current]
            for neighbor in self.adjacency.get(current, ()):
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    queue.append(neighbor)
        return distances

    def _min_distance_between_sets(self, start_locs, target_locs):
        """
        Compute minimal shortest path distance between any location in start_locs to any location in target_locs.
        Returns large number if no path is found.
        """
        min_dist = float("inf")
        for start in start_locs:
            distances = self.location_distances.get(start, {})
            for target in target_locs:
                d = distances.get(target, float("inf"))
                if d < min_dist:
                    min_dist = d
        return min_dist

    def __call__(self, node):
        state = node.state

        # If goal reached, heuristic is 0
        if self.goals <= state:
            return 0

        # Extract facts for predicates of interest
        # - Objects that are f2xo7hb6 (activated)
        activated = set()
        # - Objects that are a7sb56om (goal achieved)
        goal_achieved = set()
        # - Locations currently marked (omggbn9z)
        occupied_locations = set()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "f2xo7hb6":
                activated.add(parts[1])
            elif pred == "a7sb56om":
                goal_achieved.add(parts[1])
            elif pred == "omggbn9z":
                occupied_locations.add(parts[1])

        total_cost = 0

        # For each goal object not yet achieved, estimate required actions
        for obj in self.goal_objects:
            if obj in goal_achieved:
                continue  # already at goal

            is_activated = obj in activated

            # Locations where diwlmgg9 can be applied (kut2ksnk for this object)
            kut2_locations = self.obj_kut2ksnk.get(obj, set())
            # Locations where y7nik9q2 can be applied (klw11kjf for this object)
            klw11_locations = self.obj_klw11kjf.get(obj, set())

            # To apply diwlmgg9: need some location that is omggbn9z and kut2ksnk(obj, loc)
            # To apply y7nik9q2: need some location that is omggbn9z and klw11kjf(obj, loc)

            # Intersect candidate locations with currently occupied locations to check zero-distance cases
            available_diwlmgg9_locs = kut2_locations & occupied_locations
            available_y7nik9q2_locs = klw11_locations & occupied_locations

            # If empty, we must consider travel cost from current occupied locations to proper locations

            if is_activated:
                # Only y7nik9q2 required next
                # Compute distance from occupied locations to any location where y7nik9q2 can be applied
                if available_y7nik9q2_locs:
                    dist_y7 = 0  # can do at current occupied location
                else:
                    dist_y7 = self._min_distance_between_sets(occupied_locations, klw11_locations)
                    if dist_y7 == float("inf"):
                        # No path to required location; assign a high cost
                        dist_y7 = 10 ** 6
                # Total cost: move steps + 1 action y7nik9q2
                cost_obj = dist_y7 + 1

            else:
                # Need both diwlmgg9 and y7nik9q2 in sequence
                # First find distance to do diwlmgg9 at some location with kut2ksnk
                if available_diwlmgg9_locs:
                    dist_diwlmgg9 = 0
                else:
                    dist_diwlmgg9 = self._min_distance_between_sets(occupied_locations, kut2_locations)
                    if dist_diwlmgg9 == float("inf"):
                        dist_diwlmgg9 = 10 ** 6

                # After diwlmgg9, location changes: the effect is that f2xo7hb6 is set, and
                # kut2ksnk condition is removed, but presumably (omggbn9z) location stays at that position,
                # so next target for y7nik9q2 is at locations in klw11_locations.
                # We assume the location after diwlmgg9 is one of the kut2_locations (where diwlmgg9 applied).

                # Therefore, distance from diwlmgg9 loc to y7nik9q2 loc:
                # Find minimal distance between kut2_locations and klw11_locations:
                dist_diwlmgg9_to_y7 = self._min_distance_between_sets(kut2_locations, klw11_locations)
                if dist_diwlmgg9_to_y7 == float("inf"):
                    dist_diwlmgg9_to_y7 = 10 ** 6

                # Total cost:
                # distance from current occupied loc to diwlmgg9 loc + 1 action diwlmgg9 +
                # distance from diwlmgg9 loc to y7nik9q2 loc + 1 action y7nik9q2
                cost_obj = dist_diwlmgg9 + 1 + dist_diwlmgg9_to_y7 + 1

            total_cost += cost_obj

        # If no unsatisfied goal objects, cost is zero
        # We have already handled goal reached case at start

        # Return total heuristic cost (sum over all objects)
        return total_cost
