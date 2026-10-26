import fnmatch
from collections import deque
import math # Import math for infinity

# Helper functions
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
    return len(parts) == len(args) and all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))

# Assuming Heuristic base class is available
from heuristics.heuristic_base import Heuristic

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the cost to tighten all loose nuts specified in the goal.
    The cost is estimated by summing:
    1. The number of loose goal nuts (representing the tighten actions).
    2. The cost to acquire necessary usable spanners from the ground (including pickup actions and simplified travel between ground spanners if multiple are needed).
    3. The cost to travel to the necessary locations (including travel to the first required location - either a spanner if needed, or a nut - and simplified travel between subsequent required locations).

    # Assumptions
    - The man can carry multiple spanners (based on PDDL structure, although typically only one is needed at a time for pickup/tighten).
    - Each tighten action consumes one usable spanner.
    - Shortest path distances between locations are precomputed.
    - If the number of loose goal nuts exceeds the total number of usable spanners available (carried + on ground), the problem is considered unsolvable.

    # Heuristic Initialization
    - Extract the goal conditions to identify which nuts need tightening.
    - Identify the man object's name.
    - Extract static facts (`link` predicates) and initial state facts (`at`) to collect all relevant locations and build a graph of locations.
    - Compute all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify all nuts that are in the goal and are currently loose. Get their locations.
    3. Count the number of usable spanners the man is currently carrying.
    4. Identify usable spanners on the ground and their locations.
    5. Calculate the total number of usable spanners available (carried + on ground).
    6. If the number of loose goal nuts is greater than the total usable spanners, return infinity (unsolvable).
    7. Initialize the heuristic cost with the number of loose goal nuts (cost of tighten actions).
    8. Calculate the number of spanners that need to be acquired from the ground (`spanners_to_acquire_from_ground`). This is the number of loose goal nuts minus the number of usable spanners already carried, minimum 0.
    9. If `spanners_to_acquire_from_ground > 0`:
       - Add `spanners_to_acquire_from_ground` to the cost (pickup actions).
       - If `spanners_to_acquire_from_ground > 1`, add `spanners_to_acquire_from_ground - 1` to the cost for simplified travel between ground spanners needed for pickup.
    10. Calculate the minimum distance from the man's current location to any required location. A location is required if it's a usable spanner on the ground (and spanners are needed from the ground) or if it's a loose goal nut location. Add this minimum distance to the cost (travel to the first required location).
    11. Calculate the total number of items to visit: `spanners_to_acquire_from_ground` (representing pickup locations) + `num_loose_goal_nuts` (representing nut locations). If this total is greater than 1, add `total_items_to_visit - 1` to the cost for simplified travel between subsequent items.
    12. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and precomputing shortest path distances between locations.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Identify goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Identify the man object name
        self.man_name = None
        # Look for an object of type 'man' in initial state facts (assuming type facts are present)
        for fact in initial_state:
            parts = get_parts(fact)
            if len(parts) == 2 and parts[0] == 'man':
                self.man_name = parts[1]
                break

        if self.man_name is None:
             # Fallback: Look for an object in a 'carrying' predicate in initial state
             carried_facts = [f for f in initial_state if match(f, "carrying", "*", "*")]
             if carried_facts:
                 self.man_name = get_parts(carried_facts[0])[1]

        if self.man_name is None:
             # Fallback: Assume the first object at a location that isn't a spanner/nut is the man (fragile)
             for fact_at in initial_state:
                 if match(fact_at, "at", "*", "*"):
                     obj_at, loc_at = get_parts(fact_at)[1:]
                     if not obj_at.startswith("spanner") and not obj_at.startswith("nut"):
                          self.man_name = obj_at
                          break

        if self.man_name is None:
             # Final Fallback: Assume 'bob' if no other man object found (very fragile)
             self.man_name = 'bob' # Default assumption based on example


        # Collect all relevant locations
        all_locations_set = set()
        # Locations from link facts
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1:]
                all_locations_set.add(loc1)
                all_locations_set.add(loc2)
        # Locations from initial state 'at' facts
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 all_locations_set.add(loc)
        # Locations of goal nuts (from initial state 'at' facts) are needed.
        # These are covered by collecting 'at' facts from initial state.

        self.locations = list(all_locations_set) # Store list of locations

        # Build location graph from link facts using the collected locations
        self.location_graph = {loc: set() for loc in self.locations} # Initialize graph with all locations
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1:]
                if loc1 in self.location_graph and loc2 in self.location_graph: # Only add links between known locations
                    self.location_graph[loc1].add(loc2)
                    self.location_graph[loc2].add(loc1) # Links are bidirectional


        # Compute all-pairs shortest paths
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_loc):
        """Performs BFS from a start location to find distances to all other locations."""
        distances = {loc: math.inf for loc in self.locations}
        if start_loc not in distances:
             # This start_loc is not in our collected locations. Treat as isolated.
             return distances # All distances remain infinity

        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            current_loc = queue.popleft()

            if current_loc in self.location_graph:
                for neighbor in self.location_graph[current_loc]:
                    if neighbor in distances and distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)
        return distances

    def dist(self, loc1, loc2):
        """Returns the shortest distance between two locations."""
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
             # If locations are not in the precomputed distances, they might be isolated.
             # Distance is infinity unless loc1 == loc2 (distance 0).
             return 0 if loc1 == loc2 else math.inf
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify man's current location.
        man_loc = None
        man_name = self.man_name # Get man's name identified during initialization
        for fact in state:
             if match(fact, "at", man_name, "*"):
                 man_loc = get_parts(fact)[2]
                 break

        if man_loc is None:
             # Man's location is unknown? Should not happen in valid states.
             # Return infinity or a large value.
             return math.inf


        # 2. Identify loose nuts that are goal conditions and their locations.
        loose_goal_nuts = {} # {nut_name: location}
        for nut in self.goal_nuts:
            if f'(loose {nut})' in state:
                # Find the location of this nut
                nut_loc = None
                for fact in state:
                    if match(fact, "at", nut, "*"):
                        nut_loc = get_parts(fact)[2]
                        break
                if nut_loc:
                    loose_goal_nuts[nut] = nut_loc
                # else: nut location unknown? Problematic state.

        # If all goal nuts are tightened, heuristic is 0.
        if not loose_goal_nuts:
            return 0

        num_loose_goal_nuts = len(loose_goal_nuts)

        # 3. Count usable spanners the man is carrying.
        usable_carried_count = 0
        carried_spanners = {get_parts(f)[2] for f in state if match(f, "carrying", man_name, "*")}
        for s in carried_spanners:
            if f'(usable {s})' in state:
                usable_carried_count += 1

        # 4. Identify usable spanners on the ground and their locations.
        usable_ground_locs = {} # {spanner_name: location}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj.startswith("spanner") and f'(usable {obj})' in state and obj not in carried_spanners:
                     usable_ground_locs[obj] = loc

        # 5. Calculate total usable spanners available.
        usable_spanners_total = usable_carried_count + len(usable_ground_locs)

        # 6. Check solvability based on spanner count.
        if num_loose_goal_nuts > usable_spanners_total:
            return math.inf # Unsolvable

        # 7. Initialize heuristic cost.
        cost = num_loose_goal_nuts # Cost for tighten actions

        # 8. Calculate spanners to acquire from ground.
        spanners_to_acquire_from_ground = max(0, num_loose_goal_nuts - usable_carried_count)

        # 9. Cost for spanner acquisition (pickups and simplified travel between ground spanners).
        spanner_acquisition_cost = 0
        if spanners_to_acquire_from_ground > 0:
            spanner_acquisition_cost += spanners_to_acquire_from_ground # Pickup actions
            # Add simplified travel between subsequent spanner pickups if multiple needed from ground
            if spanners_to_acquire_from_ground > 1:
                 spanner_acquisition_cost += (spanners_to_acquire_from_ground - 1) * 1 # Simplified travel
        cost += spanner_acquisition_cost

        # 10. & 11. Cost for movement (travel to first required location + simplified travel between subsequent items).
        travel_cost = 0
        if num_loose_goal_nuts > 0:
            min_dist_to_any_target = math.inf

            # Option 1: Closest usable spanner on ground (if needed)
            if usable_carried_count == 0: # Need a spanner for the first nut
                if usable_ground_locs:
                    min_dist_to_usable_spanner = math.inf
                    for L_S in usable_ground_locs.values():
                        min_dist_to_usable_spanner = min(min_dist_to_usable_spanner, self.dist(man_loc, L_S))
                    min_dist_to_any_target = min(min_dist_to_any_target, min_dist_to_usable_spanner)
                # else: No usable spanners on ground, but total check passed means man must be carrying enough.

            # Option 2: Closest loose goal nut
            min_dist_to_nut = math.inf
            for L_N in loose_goal_nuts.values():
                min_dist_to_nut = min(min_dist_to_nut, self.dist(man_loc, L_N))
            min_dist_to_any_target = min(min_dist_to_any_target, min_dist_to_nut)

            # If min_dist_to_any_target is still infinity, it means the man is isolated from all targets.
            if min_dist_to_any_target == math.inf:
                 return math.inf

            travel_cost += min_dist_to_any_target # Travel to the first required location

            # Add simplified travel for subsequent items (spanners or nuts)
            # Total items to visit = spanners_to_acquire_from_ground (pickup locations) + num_loose_goal_nuts (nut locations)
            total_items_to_visit = spanners_to_acquire_from_ground + num_loose_goal_nuts
            if total_items_to_visit > 1:
                travel_cost += (total_items_to_visit - 1) * 1 # Simplified travel between items

        cost += travel_cost

        # 12. Return the total calculated cost.
        return cost
