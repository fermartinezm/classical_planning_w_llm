import math
from collections import deque
from fnmatch import fnmatch

# Helper functions (assuming they are not globally available)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS helper
def bfs(graph, start):
    distances = {node: float('inf') for node in graph}
    if start not in graph:
         # Start node is not in the graph nodes (e.g., an object location not linked)
         # It's an isolated node. Distances to others are infinite unless they are the start node itself.
         if start in distances: distances[start] = 0
         return distances

    distances[start] = 0
    queue = deque([start])
    while queue:
        current = queue.popleft()
        # Ensure current node is still in the graph keys (should be if it was added)
        if current not in graph: continue
        for neighbor in graph[current]:
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances

# All-pairs shortest path using BFS
def compute_all_pairs_shortest_paths(graph, all_nodes):
    all_distances = {}
    for start_node in all_nodes:
         # Ensure start_node is in the graph keys for BFS to work correctly
         # If a location is isolated, it might not be a key in adj if it has no links.
         # BFS handles this if the node is passed as 'start'.
         all_distances[start_node] = bfs(graph, start_node)
    return all_distances


# Dummy Heuristic base class for standalone testing if needed
# In the actual environment, this would be imported.
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        pass


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the cost to tighten all goal nuts that are currently loose.
    It sums the number of tighten actions needed, the estimated travel cost for the man
    to reach the location of the goal nuts, and the estimated cost to acquire enough
    usable spanners.

    # Assumptions
    - All goal nuts are located at the same location in the initial state.
    - There is only one man object. The heuristic attempts to identify the man object
      from the initial state based on predicate usage ('at' and 'carrying'). If not found,
      it assumes the man object is named 'bob' (based on examples).
    - Objects starting with 'spanner' are spanners, 'nut' are nuts, others are locations.
      This is a simplification based on example naming conventions.
    - The location graph defined by 'link' predicates is connected, or at least the
      relevant locations (man's start, nut locations, spanner locations) are reachable
      from each other.
    - A spanner becomes unusable after one tighten action.

    # Heuristic Initialization
    - Parses static facts (`link`) to build the location graph and computes all-pairs shortest paths.
    - Identifies goal nuts and their common location from the initial state (`at` facts for goal nuts).
    - Identifies the man object.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. **Identify Man's State:** Determine the man's current location and whether he is currently carrying a usable spanner.
    2. **Identify Loose Goal Nuts:** Find all goal nuts that are currently in a `(loose ?n)` state.
    3. **Goal Check:** If there are no loose goal nuts, the goal is achieved for these nuts, and the heuristic is 0.
    4. **Determine Goal Location:** Identify the single location where all goal nuts are located (precomputed during initialization based on the initial state). If this location wasn't uniquely determined (violating assumption), return infinity.
    5. **Count Needed Actions:** The minimum number of `tighten_nut` actions required is equal to the number of loose goal nuts (`NumLooseGoalNuts`). This contributes `NumLooseGoalNuts` to the heuristic.
    6. **Estimate Travel Cost:** The man needs to reach the goal nut location. Estimate this cost as the shortest distance from the man's current location to the goal nut location using the precomputed distances. If the location is unreachable, the heuristic is infinity.
    7. **Estimate Spanner Acquisition Cost:**
       - Each `tighten_nut` action consumes one usable spanner. Thus, `NumLooseGoalNuts` usable spanners are needed in total throughout the plan.
       - If the man is currently carrying a usable spanner, he can use it for the first tightening, so he needs `NumLooseGoalNuts - 1` *additional* usable spanners.
       - If he is not carrying a usable spanner, he needs `NumLooseGoalNuts` additional usable spanners.
       - Calculate the number of spanners that need to be acquired via a `pickup_spanner` action (`NumSpannersToAcquire`).
       - Find the minimum cost to acquire *one* usable spanner: This involves finding a usable spanner on the ground, traveling to its location, and picking it up. Calculate the minimum of (distance from man's current location to a spanner location + 1 for pickup) over all locations where usable spanners are currently on the ground. If no usable spanners are on the ground, this minimum cost is infinity.
       - The total estimated spanner acquisition cost is `NumSpannersToAcquire` multiplied by the minimum cost to acquire one spanner.
       - Check if enough usable spanners exist in total (carried + on ground) to satisfy `NumLooseGoalNuts`. If not, the heuristic is infinity.
       - If additional spanners are needed (`NumSpannersToAcquire > 0`) but no usable spanners are available on the ground or reachable, the heuristic is infinity.
    9. **Sum Costs:** The total heuristic value is the sum of the tighten action count, the estimated travel cost to the goal nut location, and the estimated total spanner acquisition cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # --- Identify Man Object ---
        # Attempt to identify the man object from initial state facts
        self.man_name = None
        potential_locatables = set()
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == "at" and len(parts) == 3:
                 potential_locatables.add(parts[1])
             if parts[0] == "carrying" and len(parts) == 3:
                 self.man_name = parts[1] # Object doing the carrying is the man
                 break # Found the man

        if not self.man_name:
             # Fallback: If not found via carrying, assume it's the only object at a location
             # that isn't a spanner or nut (based on name pattern assumption)
             for obj in potential_locatables:
                  if not (obj.startswith('spanner') or obj.startswith('nut')):
                       self.man_name = obj
                       break

        if not self.man_name:
             # Final fallback: Assume 'bob' as per examples if no man found
             self.man_name = 'bob'
             # print(f"Warning: Man object not clearly identified. Assuming '{self.man_name}'.")


        # --- Build Location Graph and Compute Distances ---
        self.adj = {}
        all_locations = set()

        # Collect locations from link facts
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                self.adj.setdefault(l1, set()).add(l2)
                self.adj.setdefault(l2, set()).add(l1)
                all_locations.add(l1)
                all_locations.add(l2)

        # Collect locations from initial state and goals (at predicates)
        all_facts = initial_state | self.goals
        for fact in all_facts:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3:
                 obj, loc = parts[1:]
                 # The second argument of 'at' is always a location in this domain
                 all_locations.add(loc)
                 if loc not in self.adj: self.adj[loc] = set() # Add isolated locations to graph nodes


        # Compute distances for all collected locations
        self.distances = compute_all_pairs_shortest_paths(self.adj, all_locations)


        # --- Identify Goal Nuts and their Location ---
        self.goal_nuts = set()
        self.goal_nut_location = None
        goal_nut_locations_in_init = {}

        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nut = get_parts(goal)[1]
                self.goal_nuts.add(nut)
                # Find the location of this nut in the initial state
                for fact in initial_state:
                    if match(fact, "at", nut, "*"):
                        loc = get_parts(fact)[2]
                        goal_nut_locations_in_init[nut] = loc
                        break # Found location for this nut

        # Assume all goal nuts are at the same location in the initial state
        if self.goal_nuts:
             unique_locations = set(goal_nut_locations_in_init.values())
             if len(unique_locations) == 1:
                 self.goal_nut_location = unique_locations.pop()
             elif len(unique_locations) > 1:
                 # Heuristic assumption violated. Cannot proceed with this heuristic reliably.
                 # Return infinity if goal nuts are at multiple locations.
                 self.goal_nut_location = "MULTIPLE_LOCATIONS" # Sentinel value


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Get Man's Current State ---
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_loc = get_parts(fact)[2]
                break

        if man_loc is None:
             # Man's location should always be known in a valid state
             return float('inf')

        man_carrying_spanner = None
        for fact in state:
             if match(fact, "carrying", self.man_name, "*"):
                 man_carrying_spanner = get_parts(fact)[2]
                 break

        man_carrying_usable_spanner = False
        if man_carrying_spanner:
             if f"(usable {man_carrying_spanner})" in state:
                 man_carrying_usable_spanner = True

        # --- Identify Loose Goal Nuts in Current State ---
        current_loose_goal_nuts = {nut for nut in self.goal_nuts if f"(loose {nut})" in state}
        num_loose_goal_nuts = len(current_loose_goal_nuts)

        # --- Heuristic Calculation ---
        if num_loose_goal_nuts == 0:
            return 0 # Goal reached

        # Check if goal location was identified uniquely in init
        if self.goal_nut_location is None or self.goal_nut_location == "MULTIPLE_LOCATIONS":
             # Cannot apply heuristic if goal nuts are at multiple locations (violates assumption)
             return float('inf')

        # 1. Cost for tighten actions
        tighten_cost = num_loose_goal_nuts

        # 2. Travel cost to goal nut location
        # Ensure man_loc and goal_nut_location are in the computed distances graph
        if man_loc not in self.distances or self.goal_nut_location not in self.distances.get(man_loc, {}):
             # This means the man's current location or the goal location is isolated
             # or wasn't included in the graph building, or is unreachable.
             travel_cost = float('inf')
        else:
             travel_cost = self.distances[man_loc][self.goal_nut_location]

        if travel_cost == float('inf'):
            return float('inf') # Goal location unreachable

        # 3. Spanner acquisition cost
        # Find usable spanners on the ground and their locations
        usable_spanners_on_ground_locs = []
        carried_spanner_name = None
        for fact in state:
            if match(fact, "carrying", self.man_name, "*"):
                carried_spanner_name = get_parts(fact)[2]
                break

        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                # Check if this object is a spanner (by name convention) and is usable and is on the ground
                if obj.startswith('spanner') and f"(usable {obj})" in state and obj != carried_spanner_name:
                     usable_spanners_on_ground_locs.append(loc)

        # Calculate min travel to a usable spanner on the ground
        min_travel_to_spanner = float('inf')
        for loc_s in usable_spanners_on_ground_locs:
             # Ensure loc_s is in the computed distances graph from man_loc
             if man_loc in self.distances and loc_s in self.distances[man_loc]:
                 dist = self.distances[man_loc][loc_s]
                 min_travel_to_spanner = min(min_travel_to_spanner, dist)

        # Cost to acquire one spanner (travel + pickup)
        spanner_acquisition_cost_per_pickup = min_travel_to_spanner + 1 if min_travel_to_spanner != float('inf') else float('inf')

        # Number of spanners that need to be acquired via pickup action
        # This is the total number of tightenings needed minus the one the man might be carrying (if usable).
        num_spanners_to_acquire = max(0, num_loose_goal_nuts - (1 if man_carrying_usable_spanner else 0))

        # Check if enough usable spanners exist in total (carried + on ground)
        total_usable_spanners_available = sum(1 for fact in state if match(fact, "usable", "*"))
        if num_loose_goal_nuts > total_usable_spanners_available:
             return float('inf') # Not enough spanners in the world to tighten all nuts

        # Total spanner acquisition cost
        total_spanner_acquisition_cost = 0
        if num_spanners_to_acquire > 0:
             if spanner_acquisition_cost_per_pickup == float('inf'):
                  # This case means spanners exist but are unreachable from the man's current location.
                  return float('inf')
             total_spanner_acquisition_cost = num_spanners_to_acquire * spanner_acquisition_cost_per_pickup


        # Total heuristic value
        total_heuristic = tighten_cost + travel_cost + total_spanner_acquisition_cost

        return total_heuristic
