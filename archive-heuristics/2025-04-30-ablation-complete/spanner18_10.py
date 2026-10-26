# Need to define a dummy Heuristic base class if not provided externally
# This is just for self-contained testing or if the base class isn't given.
# In a real environment, this would be imported.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            pass

from fnmatch import fnmatch
from collections import deque

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It uses a greedy approach: repeatedly find the nearest usable spanner (if needed),
    go pick it up, then find the nearest loose goal nut, go to its location, and tighten it.
    The cost is the sum of travel distances, pickup actions, and tighten actions.

    # Assumptions:
    - The man can carry multiple spanners simultaneously (based on example state).
    - Spanners become unusable after one use.
    - Locations form a connected graph (or heuristic returns infinity if needed locations are unreachable).
    - All goal nuts are initially loose and their locations are static.

    # Heuristic Initialization
    - Extract goal nuts from the task definition.
    - Extract static nut locations from the initial state (nuts don't move).
    - Build the location graph from 'link' static facts.
    - Precompute all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal nuts that are currently loose in the state.
    2. Identify all usable spanners (on the ground or carried by the man).
    3. If there are no loose goal nuts, the heuristic is 0.
    4. If the number of available usable spanners is less than the number of loose goal nuts, the problem is likely unsolvable (return infinity).
    5. Get the man's current location.
    6. Get the set of usable spanners currently carried by the man.
    7. Get the set of usable spanners currently on the ground and their locations.
    8. Initialize heuristic cost `h = 0`.
    9. Initialize the man's current location for calculation purposes (`current_loc`).
    10. Initialize the set of available usable spanners on the ground (`available_spanners_on_ground_list`) and the count of usable spanners carried by the man (`carried_usable_spanners_count`).
    11. While there are still loose goal nuts:
        a. If the man is not currently carrying a usable spanner (`carried_usable_spanners_count == 0`):
           i. Find the nearest usable spanner on the ground from `current_loc`.
           ii. Add the distance to this spanner plus the pickup action cost (1) to `h`.
           iii. Update `current_loc` to the spanner's location.
           iv. Mark the spanner as carried (increment `carried_usable_spanners_count`) and remove it from `available_spanners_on_ground_list`.
           v. If no usable spanners are available on the ground but needed (should be caught by initial check, but defensive), return infinity.
        b. Now the man is carrying a usable spanner (`carried_usable_spanners_count > 0`). Find the nearest loose goal nut from `current_loc`.
        c. Add the distance to this nut plus the tighten action cost (1) to `h`.
        d. Update `current_loc` to the nut's location.
        e. Mark the nut as tightened (remove from `loose_goal_nuts`).
        f. Use up one carried spanner (decrement `carried_usable_spanners_count`).
    12. Return the total cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal nuts, nut locations,
        and precomputing distances.
        """
        # Extract goal nuts
        self.goal_nuts = set()
        for goal in task.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened":
                nut = args[0]
                self.goal_nuts.add(nut)

        # Extract static nut locations (nuts don't move)
        self.static_nut_locations = {}
        # Nut locations are typically in the initial state
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 if obj.startswith('nut'):
                     self.static_nut_locations[obj] = loc

        # Build location graph from 'link' facts
        self.location_graph = {}
        self.all_locations = set()
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, []).append(loc2)
                self.location_graph.setdefault(loc2, []).append(loc1)
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Add any locations mentioned in initial state but not linked
        initial_locations = set()
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 initial_locations.add(loc)
        self.all_locations.update(initial_locations)

        # Ensure all locations have an entry in the graph, even if isolated
        for loc in self.all_locations:
             self.location_graph.setdefault(loc, [])

        # Precompute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.all_locations:
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_loc):
        """Perform BFS from start_loc to find distances to all other locations."""
        dist = {loc: float('inf') for loc in self.all_locations}
        dist[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            u = queue.popleft()
            # u is guaranteed to be in self.all_locations, and self.location_graph
            # has an entry for every location in self.all_locations (even if empty list)
            for v in self.location_graph[u]:
                # v is also guaranteed to be in self.all_locations
                if dist[v] == float('inf'):
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Parse state
        loose_nuts_in_state = set()
        usable_spanners_in_state = set()
        spanner_locations_on_ground = {} # Spanners on the ground
        man_location = None
        man_carrying_spanners = set() # Set of spanner names carried by man

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "at":
                obj, loc = parts[1], parts[2]
                if obj == 'bob': # Assuming 'bob' is the man object name
                     man_location = loc
                elif obj.startswith('spanner'):
                     # This spanner is on the ground
                     spanner_locations_on_ground[obj] = loc

            elif predicate == "loose":
                nut = parts[1]
                loose_nuts_in_state.add(nut)

            elif predicate == "usable":
                spanner = parts[1]
                usable_spanners_in_state.add(spanner)

            elif predicate == "carrying":
                 # Assuming the predicate is (carrying man spanner)
                 # We don't strictly need the man name here if there's only one man
                 spanner = parts[2]
                 man_carrying_spanners.add(spanner)

        # 2. Identify relevant objects based on goals and state
        # Only consider loose nuts that are goal nuts
        loose_goal_nuts = list(loose_nuts_in_state.intersection(self.goal_nuts)) # Use list for easy removal
        usable_spanners_on_ground = {s for s in usable_spanners_in_state if s in spanner_locations_on_ground}
        carried_usable_spanners = {s for s in man_carrying_spanners if s in usable_spanners_in_state}

        # 3. Check base cases
        K = len(loose_goal_nuts)
        M_ground = len(usable_spanners_on_ground)
        M_carried = len(carried_usable_spanners)

        if K == 0:
            return 0

        if M_ground + M_carried < K:
            # Not enough usable spanners available in the state to tighten all goal nuts
            return float('inf') # Return a large value

        # 4. Compute heuristic using sequential greedy approach
        h = 0
        current_loc = man_location
        available_spanners_on_ground_list = list(usable_spanners_on_ground) # Use list for easy removal
        carried_usable_spanners_count = M_carried

        # The order of processing loose nuts and available spanners affects the greedy choice.
        # We iterate until all loose goal nuts are conceptually tightened.

        while loose_goal_nuts:
            # Do we need a spanner?
            if carried_usable_spanners_count == 0:
                # Need to get a spanner from the ground
                # Find the nearest usable spanner on the ground
                min_dist_to_spanner = float('inf')
                best_spanner = None
                best_spanner_loc = None

                # This check should be redundant due to the M_ground + M_carried < K check before the loop,
                # but defensive programming doesn't hurt.
                if not available_spanners_on_ground_list:
                     # Should not happen if initial spanner count was sufficient
                     return float('inf')

                for spanner in available_spanners_on_ground_list:
                    loc = spanner_locations_on_ground[spanner] # Use the map for ground spanners
                    dist = self.distances[current_loc][loc]

                    if dist < min_dist_to_spanner:
                        min_dist_to_spanner = dist
                        best_spanner = spanner
                        best_spanner_loc = loc

                # If the nearest spanner is unreachable, the state is unsolvable
                if min_dist_to_spanner == float('inf'):
                     return float('inf')

                # Add cost to get the spanner
                h += min_dist_to_spanner + 1 # Travel + Pickup
                current_loc = best_spanner_loc
                carried_usable_spanners_count += 1
                available_spanners_on_ground_list.remove(best_spanner)

            # Now man is carrying a usable spanner. Find the nearest loose goal nut.
            min_dist_to_nut = float('inf')
            best_nut = None
            best_nut_loc = None

            for nut in loose_goal_nuts:
                loc = self.static_nut_locations[nut] # Get nut location from static info
                dist = self.distances[current_loc][loc]

                if dist < min_dist_to_nut:
                    min_dist_to_nut = dist
                    best_nut = nut
                    best_nut_loc = loc

            # If the nearest nut is unreachable, the state is unsolvable
            if min_dist_to_nut == float('inf'):
                 return float('inf')

            # Add cost to tighten the nut
            h += min_dist_to_nut + 1 # Travel + Tighten
            current_loc = best_nut_loc
            loose_goal_nuts.remove(best_nut) # Remove the tightened nut
            carried_usable_spanners_count -= 1 # Use up one spanner

        return h
