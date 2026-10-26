from fnmatch import fnmatch
from collections import deque

# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing if needed
# This part should be removed when integrating into a planner environment
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Added for man name inference

    def __call__(self, node):
        raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential errors or unexpected fact formats
        # print(f"Warning: Unexpected fact format: {fact}")
        return [] # Return empty list for malformed facts

    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts is at least the number of args
    if len(parts) < len(args):
         return False
    # Check if the provided args match the beginning of the parts
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the cost to tighten all goal nuts by summing
    the estimated costs for three main components for each loose goal nut:
    1. The cost of the 'tighten_nut' action itself.
    2. The cost for the man to travel from his current location to the nut's location.
    3. The cost for the man to acquire a usable spanner (including travel and pickup).

    The heuristic sums these costs over all loose goal nuts, with spanner acquisition
    cost estimated based on the number of spanners needed from the ground and the
    distance to the closest available usable spanners.

    # Assumptions
    - Each goal nut requires a separate 'tighten_nut' action.
    - Each 'tighten_nut' action consumes one usable spanner.
    - Spanners cannot be made usable again.
    - The problem is solvable only if there are enough usable spanners (including the one the man might be carrying) as there are loose goal nuts.
    - The cost of actions (walk, pickup, tighten) is 1.
    - Distances between locations are shortest path distances based on 'link' facts.
    - There is exactly one man object in the domain.

    # Heuristic Initialization
    - Extracts the set of goal nuts from the task's goal conditions.
    - Builds a graph of locations based on 'link' static facts.
    - Computes all-pairs shortest paths between locations using BFS to determine travel costs.
    - Identifies the name of the man object by looking for objects involved in 'at' or 'carrying' predicates that are not nuts or spanners (heuristic inference based on domain structure).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all nuts that are part of the goal and are currently in a 'loose' state. Let this set be `LooseGoalNuts`.
    2. If `LooseGoalNuts` is empty, the goal is reached, and the heuristic value is 0.
    3. If `LooseGoalNuts` is not empty, determine the number of loose goal nuts, `K = |LooseGoalNuts|`.
    4. Find the man's current location (`L_man`).
    5. Find the location (`L_N`) for each nut `N` in `LooseGoalNuts`.
    6. Identify all usable spanners currently on the ground (`UsableSpannersOnGround`) and their locations (`L_S`).
    7. Check if the man is currently carrying a usable spanner (`ManCarryingUsable`).
    8. Check if the total number of available usable spanners (`|UsableSpannersOnGround| + (1 if ManCarryingUsable else 0)`) is less than `K`. If so, the problem is likely unsolvable with the available spanners, return a large heuristic value (e.g., K * 100).
    9. Initialize the total heuristic cost `h = 0`.
    10. Add the cost for the 'tighten_nut' actions: `h += K * 1`.
    11. Add the estimated travel cost for the man to reach all nut locations. A simple estimate is the sum of shortest path distances from `L_man` to each `L_N` for `N` in `LooseGoalNuts`. `h += sum(self.dist(L_man, L_N) for N in LooseGoalNuts)`.
    12. Calculate the number of additional usable spanners the man needs to pick up from the ground: `spanners_to_pickup = K - (1 if ManCarryingUsable else 0)`. This value is at least 0.
    13. If `spanners_to_pickup > 0`:
        a. Add the cost for the 'pickup_spanner' actions: `h += spanners_to_pickup * 1`.
        b. Estimate the travel cost to reach the spanners. Find the `spanners_to_pickup` usable spanners on the ground that are closest to `L_man`. Add the sum of the shortest path distances from `L_man` to the locations of these closest spanners.
        c. To do this, calculate `dist(L_man, L_S)` for all `S` in `UsableSpannersOnGround`, sort them by distance, take the top `spanners_to_pickup` entries, and sum their distances. `h += sum(dist for dist, s, loc in sorted_closest_spanners_info)`.
    14. Return the total calculated heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal nuts and precomputing distances."""
        super().__init__(task)

        # Extract goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Build location graph from static 'link' facts
        self.location_graph = {}
        locations = set()
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                locations.add(l1)
                locations.add(l2)
                self.location_graph.setdefault(l1, set()).add(l2)
                self.location_graph.setdefault(l2, set()).add(l1)

        self.locations = list(locations) # Store locations list

        # Precompute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self._bfs(start_loc)

        # Identify the man object name (assuming only one man)
        self.man_name = None
        # Infer man name from initial state facts
        initial_state_facts = task.initial_state
        for fact in initial_state_facts:
             parts = get_parts(fact)
             if parts and parts[0] == "at" and len(parts) == 3:
                  obj_name, loc_name = parts[1:]
                  # Heuristically check if it's the man based on typical object naming
                  if not obj_name.startswith("spanner") and not obj_name.startswith("nut"):
                       self.man_name = obj_name
                       break
             elif parts and parts[0] == "carrying" and len(parts) == 3:
                  carrier_name, spanner_name = parts[1:]
                  self.man_name = carrier_name # 'carrying' predicate involves the man
                  break

        if self.man_name is None:
             # Fallback if inference fails (e.g., empty initial state or unusual naming)
             # This might indicate an issue with the task definition or state.
             # For robustness, could try to find any object not a spanner/nut in goal/static,
             # or assume a default name like 'bob'. Let's assume 'bob' as a fallback.
             self.man_name = "bob" # Assume 'bob' if inference fails


    def _bfs(self, start_loc):
        """Performs BFS from start_loc to compute distances to all reachable locations."""
        q = deque([(start_loc, 0)])
        visited = {start_loc}
        self.distances[(start_loc, start_loc)] = 0

        while q:
            curr_loc, dist = q.popleft()

            if curr_loc in self.location_graph: # Ensure location has links
                for neighbor in self.location_graph[curr_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_loc, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

    def dist(self, l1, l2):
        """Returns the shortest distance between two locations, or float('inf') if unreachable."""
        return self.distances.get((l1, l2), float('inf'))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract current state information
        loose_nuts_in_state = set()
        all_nut_locations = {}
        man_location = None
        spanner_locations_on_ground = {}
        usable_spanners_in_state = set()
        man_carrying_spanner = None

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == "loose":
                if len(parts) == 2: loose_nuts_in_state.add(parts[1])
            elif predicate == "at":
                if len(parts) == 3:
                    obj_name, loc_name = parts[1:]
                    if obj_name.startswith("nut"):
                        all_nut_locations[obj_name] = loc_name
                    elif obj_name == self.man_name:
                        man_location = loc_name
                    elif obj_name.startswith("spanner"):
                        spanner_locations_on_ground[obj_name] = loc_name
            elif predicate == "usable":
                 if len(parts) == 2: usable_spanners_in_state.add(parts[1])
            elif predicate == "carrying":
                 if len(parts) == 3:
                     carrier_name, spanner_name = parts[1:]
                     if carrier_name == self.man_name:
                         man_carrying_spanner = spanner_name

        # 1. Identify loose goal nuts
        loose_goal_nuts = {nut for nut in self.goal_nuts if nut in loose_nuts_in_state}

        # 2. Check if goal is reached
        K = len(loose_goal_nuts)
        if K == 0:
            return 0

        # 3. Get man's location
        if man_location is None:
             # Man's location is unknown - state is likely invalid
             return float('inf')

        # Get locations for the loose goal nuts
        loose_goal_nut_locations = {}
        for nut in loose_goal_nuts:
             loc = all_nut_locations.get(nut)
             if loc is None:
                  # Location of a loose goal nut is unknown - state is likely invalid
                  return float('inf')
             loose_goal_nut_locations[nut] = loc


        # 4. Identify usable spanners on ground and their locations
        usable_spanners_on_ground = {
            s: loc for s, loc in spanner_locations_on_ground.items() if s in usable_spanners_in_state
        }

        # 5. Check if man is carrying a usable spanner
        man_carrying_usable = (man_carrying_spanner is not None) and (man_carrying_spanner in usable_spanners_in_state)

        # 6. Check solvability based on spanner count
        total_usable_spanners_available = len(usable_spanners_on_ground) + (1 if man_carrying_usable else 0)
        if total_usable_spanners_available < K:
            # Not enough usable spanners to tighten all nuts
            # Return a large value indicating likely unsolvable
            return K * 100 # Using a large finite value for greedy BFS

        # 9. Initialize heuristic cost
        h = 0

        # 10. Add cost for tighten actions
        h += K # Each tighten action costs 1

        # 11. Add estimated travel cost for man to reach nut locations
        # Sum of distances from man's current location to each loose goal nut's location
        travel_to_nuts_cost = 0
        for nut, nut_loc in loose_goal_nut_locations.items():
            dist_to_nut = self.dist(man_location, nut_loc)
            if dist_to_nut == float('inf'):
                 # Nut location is unreachable
                 return float('inf')
            travel_to_nuts_cost += dist_to_nut
        h += travel_to_nuts_cost

        # 12. Calculate spanner acquisition cost
        spanners_to_pickup = K - (1 if man_carrying_usable else 0)
        spanners_to_pickup = max(0, spanners_to_pickup) # Ensure non-negative

        if spanners_to_pickup > 0:
            # 13a. Add cost for pickup actions
            h += spanners_to_pickup # Each pickup action costs 1

            # 13b. Estimate travel cost to reach spanners
            usable_spanners_on_ground_with_dist = []
            for s, loc in usable_spanners_on_ground.items():
                dist_to_spanner = self.dist(man_location, loc)
                if dist_to_spanner != float('inf'):
                    usable_spanners_on_ground_with_dist.append((dist_to_spanner, s, loc))

            # If after filtering unreachable spanners, we don't have enough, return inf
            # This check is slightly redundant given the earlier total_usable_spanners_available check,
            # but accounts for reachability from the *current* man location.
            if len(usable_spanners_on_ground_with_dist) < spanners_to_pickup:
                 return float('inf') # Not enough *reachable* usable spanners on ground

            # Sort by distance and take the closest ones
            needed_spanners_info = sorted(usable_spanners_on_ground_with_dist)[:spanners_to_pickup]

            # Add travel cost to these spanners
            travel_to_spanners_cost = sum(dist for dist, s, loc in needed_spanners_info)
            h += travel_to_spanners_cost

        # 14. Return total heuristic value
        return h
