from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (ud3uf11z) domain.

    Summary:
    This heuristic estimates the number of actions needed to achieve the goal 
    condition (a7sb56om ?praz5) for all specified objects ?praz5 of type nxbd1hc3.
    The domain models a process of enabling objects by first "activating" them via 
    actions involving two specialized relations (kut2ksnk and klw11kjf) and a 
    connectivity relation (acovodrf) between locations (objects t4rdxh32) where 
    objects may be "located" or moved.

    Assumptions:
    - The goal is to have the fact (a7sb56om ?praz5) true for all relevant objects.
    - Objects that have (a7sb56om) are considered finished.
    - Objects with (f2xo7hb6) are activated but not yet finalized.
    - An object can only become activated (f2xo7hb6) at a location where (kut2ksnk) holds.
    - To finalize (a7sb56om), an object must be activated (f2xo7hb6) and satisfy (klw11kjf) with some location where (omggbn9z) holds.
    - Locations can move the "active" marker (omggbn9z) between them based on connectivity (acovodrf).
    - We assume the minimal number of moves and activations are required to reach the goal.
    - Movement of the active marker (omggbn9z) along the acovodrf graph is needed to enable or finalize objects at various locations.

    Heuristic Initialization:
    - Extract from static facts the connectivity graph between locations (acovodrf).
    - Extract the fixed associations kut2ksnk and klw11kjf between objects and locations.
    - Extract the set of goal objects that must have (a7sb56om).
    - Store the initial locations of the active marker (omggbn9z), since only one location holds it at a time.
    - Precompute shortest distances between locations using the connectivity graph (undirected edges).

    Step-By-Step Thinking for Computing Heuristic:
    Given a state:
    1. Identify all objects that still need to reach (a7sb56om). For those objects:
       - If already (a7sb56om), ignore.
       - Else, if (f2xo7hb6) is true, they are activated but not finalized.
       - Else, they are inactive and need to be activated first.

    2. For each object that needs to be activated:
       - Find the location of the active marker (omggbn9z).
       - Determine the locations where (kut2ksnk) for this object is true (from static).
       - The active marker must be moved along acovodrf edges to one of these locations to activate the object.
       - Cost to activate equals: shortest distance from current active marker location to one of the kut2ksnk locations + 1 action (diwlmgg9).

    3. For each object that is activated but not finalized:
       - Find the location where the active marker currently is.
       - Determine the locations connected to the object via (klw11kjf).
       - The active marker must move to one of these klw11kjf locations.
       - Cost to finalize equals: shortest distance to one of these locations + 1 action (y7nik9q2).

    4. The active marker moves between locations, so sequencing matters.
       - To approximate efficiently, independently estimate for each object the cost to activate or finalize ignoring ordering.
       - Sum these minimal costs for all objects not yet finalized.

    5. Additionally, consider that moving the active marker between locations to activate or finalize objects is counted by shortest path lengths.
    
    6. The heuristic value is the sum of:
       - For each unactivated object: distance from active marker to nearest activation location + 1
       - For each activated but unfinished object: distance from active marker to nearest finalization location + 1

    7. Return 0 if all objects satisfy (a7sb56om).

    This heuristic is not guaranteed admissible but aims to guide search efficiently.

    """

    def __init__(self, task):
        # Extract static facts for efficient lookups
        self.kut2ksnk = dict()  # object -> set of locations
        self.klw11kjf = dict()  # object -> set of locations
        self.adjacency = dict()  # Location adjacency graph from acovodrf (undirected)
        self.goal_objects = set()
        self.locations = set()
        self.objects = set()

        # Extract objects and locations from facts
        for fact in task.static:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == "kut2ksnk":
                obj, loc = parts[1], parts[2]
                self.kut2ksnk.setdefault(obj, set()).add(loc)
                self.objects.add(obj)
                self.locations.add(loc)
            elif pred == "klw11kjf":
                obj, loc = parts[1], parts[2]
                self.klw11kjf.setdefault(obj, set()).add(loc)
                self.objects.add(obj)
                self.locations.add(loc)
            elif pred == "acovodrf":
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.adjacency.setdefault(loc1, set()).add(loc2)
                self.adjacency.setdefault(loc2, set()).add(loc1)

        # Goal objects determined from goal facts (a7sb56om ?obj)
        for goal_fact in task.goals:
            parts = get_parts(goal_fact)
            if parts[0] == "a7sb56om":
                self.goal_objects.add(parts[1])
                self.objects.add(parts[1])

        self.locations = frozenset(self.locations)

        # Precompute all-pairs shortest paths between locations using Floyd-Warshall
        loc_list = list(self.locations)
        self.location_index = {loc: i for i, loc in enumerate(loc_list)}
        n = len(loc_list)
        INF = 10 ** 9
        dist = [[INF] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        for loc, neighbors in self.adjacency.items():
            i = self.location_index[loc]
            for nb in neighbors:
                j = self.location_index[nb]
                dist[i][j] = 1
                dist[j][i] = 1
        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                di = dist[i]
                dik = di[k]
                dj = dist[k]
                for j in range(n):
                    d = dik + dj[j]
                    if d < di[j]:
                        di[j] = d
        self.dist = dist
        self.loc_list = loc_list

    def shortest_dist(self, loc_from, loc_to_candidates):
        """
        Compute shortest distance from loc_from to any location in loc_to_candidates using precomputed dist.
        Returns INF (large int) if no path found or inputs invalid.
        """
        if loc_from not in self.location_index:
            return 10 ** 9
        i = self.location_index[loc_from]
        min_dist = 10 ** 9
        for loc_to in loc_to_candidates:
            if loc_to not in self.location_index:
                continue
            j = self.location_index[loc_to]
            d = self.dist[i][j]
            if d < min_dist:
                min_dist = d
        return min_dist

    def __call__(self, node):
        """
        Estimate number of actions required to reach a goal state from the current node.

        We analyze the state to:
        - Identify objects not yet finalized (lack a7sb56om)
        - Identify which are activated (f2xo7hb6) and which not
        - Find current active marker location (omggbn9z), assume exactly one exists
        - For each object not finalized, estimate cost to activate or finalize,
          incorporating movement cost of active marker plus 1 activation/finalizing action

        The heuristic sums these minimal costs over all relevant objects.
        """

        state = node.state

        # Quickly check if goal already achieved
        if self.goal_objects <= state:
            return 0

        # Identify the location that currently has (omggbn9z)
        active_marker_locs = [fact for fact in state if fact.startswith("(omggbn9z ")]
        if not active_marker_locs:
            # No active marker location found, assume high cost
            return 10 ** 9
        # There should be exactly one such fact (omggbn9z ?loc)
        active_marker_fact = active_marker_locs[0]
        parts = get_parts(active_marker_fact)
        active_marker_loc = parts[1]

        heuristic_cost = 0

        # For each goal object not finalized, determine cost to activate or finalize
        for obj in self.goal_objects:
            obj_final_fact = f"(a7sb56om {obj})"
            if obj_final_fact in state:
                # Already finalized
                continue
            obj_activated_fact = f"(f2xo7hb6 {obj})"
            if obj_activated_fact in state:
                # Activated but not finalized: move active marker to a location that satisfies klw11kjf and apply y7nik9q2 (1 action)
                # If no klw11kjf locations, assume cannot finalize -> high cost
                final_locs = self.klw11kjf.get(obj, None)
                if not final_locs:
                    # No known location to finalize, very high cost
                    heuristic_cost += 10 ** 9
                    continue
                dist_to_finalize = self.shortest_dist(active_marker_loc, final_locs)
                if dist_to_finalize == 10 ** 9:
                    # No path: very high cost
                    heuristic_cost += 10 ** 9
                else:
                    heuristic_cost += dist_to_finalize + 1  # +1 for the y7nik9q2 action
            else:
                # Not activated: first activate by moving active marker to a kut2ksnk location and applying diwlmgg9 (1 action)
                activate_locs = self.kut2ksnk.get(obj, None)
                if not activate_locs:
                    # No known location to activate -> high cost
                    heuristic_cost += 10 ** 9
                    continue
                dist_to_activate = self.shortest_dist(active_marker_loc, activate_locs)
                if dist_to_activate == 10 ** 9:
                    heuristic_cost += 10 ** 9
                    continue
                # After activation, assume immediately finalize (optimistic)
                # So also add minimal distance from activate location to a klw11kjf location + 1
                final_locs = self.klw11kjf.get(obj, None)
                if not final_locs:
                    heuristic_cost += 10 ** 9
                    continue
                # Try all combinations of activate loc and finalize loc to find minimal sum distance
                min_finalize_dist = 10 ** 9
                for activate_loc in activate_locs:
                    for finalize_loc in final_locs:
                        if activate_loc not in self.location_index or finalize_loc not in self.location_index:
                            continue
                        al = self.location_index[activate_loc]
                        fl = self.location_index[finalize_loc]
                        d = self.dist[al][fl]
                        if d < min_finalize_dist:
                            min_finalize_dist = d
                if min_finalize_dist == 10 ** 9:
                    heuristic_cost += 10 ** 9
                    continue

                # Total cost: distance to activate location + 1 action (diwlmgg9) + 
                # distance to finalize location + 1 action (y7nik9q2)
                heuristic_cost += dist_to_activate + 1 + min_finalize_dist + 1

        return heuristic_cost
