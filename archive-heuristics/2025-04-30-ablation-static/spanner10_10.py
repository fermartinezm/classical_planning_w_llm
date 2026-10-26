# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match facts
from fnmatch import fnmatch
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts is at least the number of args (to handle trailing wildcards)
    if len(parts) < len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS for shortest paths
def bfs(graph, start_node):
    """Computes shortest path distances from start_node to all other nodes in the graph."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph: # Handle case where start_node is not in the graph (e.g., isolated location)
        return distances
    distances[start_node] = 0
    queue = [start_node]
    while queue:
        u = queue.pop(0)
        for v in graph.get(u, []):
            if distances[v] == float('inf'):
                distances[v] = distances[u] + 1
                queue.append(v)
    return distances

# Dummy Heuristic base class if not provided
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions (tighten, pickup, walk) required
    to tighten all goal nuts. It considers the number of nuts remaining, the travel
    cost to reach the closest nut, the number of spanners that need to be picked up,
    and the travel cost to reach those spanners.

    # Assumptions:
    - Nuts are static and remain at their initial locations.
    - Links between locations are bidirectional for walking.
    - A spanner becomes unusable after tightening one nut.
    - The man can carry multiple spanners (although only one usable spanner is needed per tighten action).
    - Sufficient usable spanners exist in the initial state (either carried or on the ground)
      to tighten all goal nuts, if the problem is solvable.
    - The man object can be identified (e.g., by name like 'bob').

    # Heuristic Initialization
    - Identify all locations from initial state and static facts.
    - Build a graph of locations based on 'link' predicates and compute all-pairs shortest paths using BFS.
    - Identify the man object.
    - Identify all nut objects and store their static initial locations.
    - Identify all spanner objects.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the set of loose nuts that are also goal conditions (`GoalLooseNuts`).
       If this set is empty, the goal is reached, and the heuristic is 0.

    2. Initialize the heuristic value (`h`) with the number of untightened goal nuts.
       This accounts for the minimum number of `tighten_nut` actions required.

    3. Find the man's current location (`L_man`).

    4. Find the locations of all nuts in `GoalLooseNuts`.

    5. Calculate the minimum shortest path distance from the man's current location
       to any of the locations of the untightened goal nuts. Add this distance to `h`.
       This estimates the travel cost to reach the first nut.

    6. Count the number of usable spanners the man is currently carrying (`N_usable_carried`).

    7. Count the number of usable spanners currently on the ground (`N_usable_on_ground`).

    8. Determine how many *additional* usable spanners the man needs to pick up from the ground.
       This is `N_spanners_to_pickup = max(0, |GoalLooseNuts| - N_usable_carried)`.

    9. Check if the total number of currently usable spanners (carried + on ground)
       is less than the number of nuts that need tightening. If so, the problem is
       unsolvable with the currently available usable spanners, and we return a large value (1000000).

    10. Add `N_spanners_to_pickup` to `h`. This accounts for the minimum number of `pickup_spanner` actions required.

    11. If `N_spanners_to_pickup > 0`:
        a. Find all usable spanners currently on the ground and their locations.
        b. Calculate the shortest path distance from the man's current location to the location of each of these spanners.
        c. Sort these usable spanners on the ground by their distance from the man.
        d. Sum the distances to the locations of the first `N_spanners_to_pickup` spanners in the sorted list. Add this sum to `h`.
           This estimates the travel cost specifically for picking up the necessary spanners.

    12. Return the calculated heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and computing distances."""
        self.goals = task.goals
        initial_state = task.initial_state
        static_facts = task.static

        # 1. Identify objects and locations
        self.man = None
        self.all_locations = set()
        self.all_nuts = set()
        self.all_spanners = set()

        all_facts = set(initial_state) | set(static_facts) | set(self.goals)

        for fact in all_facts:
            parts = get_parts(fact)
            if parts[0] == 'at':
                # Object and Location
                obj, loc = parts[1], parts[2]
                self.all_locations.add(loc)
                # Attempt to identify man based on common name pattern
                if 'bob' in obj.lower() or 'man' in obj.lower():
                     self.man = obj
            elif parts[0] == 'link':
                # Locations from links
                self.all_locations.add(parts[1])
                self.all_locations.add(parts[2])
            elif parts[0] == 'loose' or parts[0] == 'tightened':
                 # Nuts
                 self.all_nuts.add(parts[1])
            elif parts[0] == 'usable':
                 # Spanners
                 self.all_spanners.add(parts[1])
            elif parts[0] == 'carrying':
                 # Man and Spanner
                 self.man = parts[1] # Man is the carrier
                 self.all_spanners.add(parts[2]) # Spanner is carried

        # Fallback/ensure man is found if not by carrying
        if self.man is None:
             # This case is less likely if 'at' facts are processed first
             for fact in initial_state:
                 if match(fact, "at", "*", "*"):
                     obj_name = get_parts(fact)[1]
                     if 'bob' in obj_name.lower() or 'man' in obj_name.lower():
                          self.man = obj_name
                          break

        # Ensure all identified nuts and spanners are added to their sets
        for fact in all_facts:
             parts = get_parts(fact)
             if parts[0] == 'loose' or parts[0] == 'tightened':
                 self.all_nuts.add(parts[1])
             elif parts[0] == 'usable':
                 self.all_spanners.add(parts[1])
             elif parts[0] == 'carrying':
                 self.all_spanners.add(parts[2])


        # 2. Build location graph and compute distances
        self.location_graph = {loc: [] for loc in self.all_locations}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                if l1 in self.location_graph and l2 in self.location_graph: # Ensure locations are known
                    self.location_graph[l1].append(l2)
                    self.location_graph[l2].append(l1) # Links are bidirectional

        self.distances = {}
        for loc in self.all_locations:
            self.distances[loc] = bfs(self.location_graph, loc)

        # 3. Store static nut locations
        self.nut_locations = {}
        for nut in self.all_nuts:
            # Nut locations are assumed static, find them in initial state
            found_loc = False
            for fact in initial_state:
                if match(fact, "at", nut, "*"):
                    self.nut_locations[nut] = get_parts(fact)[2]
                    found_loc = True
                    break
            # If a nut exists (e.g., in goals) but has no initial @at fact, it's problematic.
            # We only store locations for nuts found with an initial @at fact.
            if not found_loc and nut in {get_parts(g)[1] for g in self.goals if match(g, "tightened", "*")}:
                 # This goal nut has no initial location. Problematic instance.
                 # print(f"Warning: Goal nut {nut} has no initial location.")
                 pass # Heuristic might be inaccurate or return large value later if location is needed.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify loose nuts that are goal conditions
        goal_nuts_to_tighten = set()
        for goal_fact in self.goals:
            if match(goal_fact, "tightened", "*"):
                nut = get_parts(goal_fact)[1]
                # Check if this nut is one we know about and is currently loose
                if nut in self.all_nuts and any(match(fact, "loose", nut) for fact in state):
                     goal_nuts_to_tighten.add(nut)

        if not goal_nuts_to_tighten:
            return 0 # Goal reached

        # 2. Calculate base cost (tighten actions)
        h = len(goal_nuts_to_tighten)

        # 3. Find man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", self.man, "*"):
                man_location = get_parts(fact)[2]
                break
        if man_location is None:
             # Man's location must be known in a valid state
             return 1000000 # Should not happen

        # 4. Find locations of all loose goal nuts
        nut_locations = set()
        for nut in goal_nuts_to_tighten:
             if nut in self.nut_locations:
                 nut_locations.add(self.nut_locations[nut])
             else:
                 # Location of a goal nut is unknown (should not happen based on init logic)
                 return 1000000

        if not nut_locations:
             # Should not happen if goal_nuts_to_tighten is not empty and nut_locations were found in init
             return 1000000

        # 5. Add minimum distance from man's location to any nut location
        min_dist_to_nut = float('inf')
        if man_location in self.distances:
            for loc in nut_locations:
                 if loc in self.distances[man_location]:
                     min_dist_to_nut = min(min_dist_to_nut, self.distances[man_location][loc])
                 else:
                     # Path from man to nut location unknown
                     return 1000000
        else:
             # Man's location is not in the graph (should not happen)
             return 1000000

        if min_dist_to_nut == float('inf'):
             # No path to any nut location (should not happen if graph is connected)
             return 1000000

        h += min_dist_to_nut

        # 6. Count usable spanners currently carried by the man
        usable_carried = set()
        for s in self.all_spanners:
            is_carrying = any(match(fact, "carrying", self.man, s) for fact in state)
            is_usable = any(match(fact, "usable", s) for fact in state)
            if is_carrying and is_usable:
                usable_carried.add(s)
        n_usable_carried = len(usable_carried)

        # 7. Count usable spanners currently on the ground
        usable_on_ground = set()
        for s in self.all_spanners:
            is_usable = any(match(fact, "usable", s) for fact in state)
            is_on_ground = any(match(fact, "at", s, "*") for fact in state)
            if is_usable and is_on_ground:
                usable_on_ground.add(s)
        n_usable_on_ground = len(usable_on_ground)

        # 8. Calculate number of additional spanners needed from the ground
        n_spanners_to_pickup = max(0, len(goal_nuts_to_tighten) - n_usable_carried)

        # 9. Check if total usable spanners are sufficient
        if n_usable_carried + n_usable_on_ground < len(goal_nuts_to_tighten):
             return 1000000 # Unsolvable with currently usable spanners

        # 10. Add cost for pickup actions needed
        h += n_spanners_to_pickup

        # 11. Add travel cost for picking up spanners if needed
        if n_spanners_to_pickup > 0:
            # Find usable spanners on ground and their locations
            usable_spanners_on_ground_with_loc = []
            for s in usable_on_ground: # Iterate through the set found in step 7
                for fact in state:
                    if match(fact, "at", s, "*"):
                        loc = get_parts(fact)[2]
                        usable_spanners_on_ground_with_loc.append((s, loc))
                        break # Found location for this spanner

            # This list should not be empty if step 9 passed and n_spanners_to_pickup > 0

            # Calculate distances from man's location to each usable spanner location on ground
            spanner_loc_dist_pairs = []
            for s, loc in usable_spanners_on_ground_with_loc:
                 if man_location in self.distances and loc in self.distances[man_location]:
                     spanner_loc_dist_pairs.append((s, loc, self.distances[man_location][loc]))
                 else:
                     # Path from man to spanner location unknown
                     return 1000000

            # Sort by distance and take the first N_spanners_to_pickup
            spanner_loc_dist_pairs.sort(key=lambda item: item[2])

            # Sum the distances to the locations of the first N_spanners_to_pickup spanners
            # Ensure we don't try to sum more distances than available spanners on ground locations
            # Note: This sums distances to distinct spanners, not distinct locations.
            num_to_sum = min(n_spanners_to_pickup, len(spanner_loc_dist_pairs))
            travel_cost_for_pickups = sum(item[2] for item in spanner_loc_dist_pairs[:num_to_sum])
            h += travel_cost_for_pickups

        return h
