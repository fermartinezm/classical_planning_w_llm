import math
from collections import deque
# Assuming the Heuristic base class is available in this path.
# Adjust the import if necessary based on the project structure.
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL fact strings like '(predicate arg1 arg2)'
def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string."""
    # Basic validation for expected format: (predicate ... )
    if not fact or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

class SpannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the PDDL Spanner domain.

    # Summary
    Estimates the cost to tighten all goal nuts that are currently loose.
    It simulates a greedy strategy where the man repeatedly performs the
    cheapest action sequence (walk, pickup, walk, tighten) to tighten one
    of the remaining loose nuts, consuming a usable spanner each time. The cost
    is the sum of actions in this simulated greedy plan. This heuristic is
    designed for Greedy Best-First Search and is not necessarily admissible.

    # Assumptions
    - There is exactly one 'man' agent in the problem instance.
    - Nuts do not change location throughout the plan (their locations are static).
    - Each 'tighten_nut' action requires and consumes exactly one usable spanner.
    - The 'link' predicates define symmetric connectivity between locations.
    - Object naming conventions (e.g., 'nut' in nut names, 'spanner' in spanner names)
      are consistent with examples, aiding object identification.

    # Heuristic Initialization
    - Identifies the man agent's name by checking 'at' or 'carrying' predicates
      in the initial state. Falls back to checking for 'bob' if needed.
    - Stores the locations of all nuts, assuming locations are static and defined
      by 'at' predicates in the initial state.
    - Stores the set of nuts that need to be 'tightened' according to the goal definition.
    - Builds an undirected graph representation of the locations based on static 'link' predicates.
    - Precomputes all-pairs shortest path distances between locations using Breadth-First
      Search (BFS). Unreachable locations will have infinite distance.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state (`node.state`) to determine:
       - The man's current location (`man_loc`).
       - Whether the man is carrying a spanner (`carried_spanner`).
       - Whether the carried spanner (if any) is currently 'usable' (`carried_spanner_usable`).
       - The set of goal nuts that are currently 'loose' (`nuts_to_tighten`).
       - The set of 'usable' spanners currently available on the ground (not carried),
         and their locations (`available_usable_spanners`).
    2. If `nuts_to_tighten` is empty, all goal nuts are tightened. Return heuristic value 0.
    3. Initialize `total_heuristic_cost = 0`.
    4. Create mutable copies of the relevant state information for simulation:
       `sim_man_loc`, `sim_carrying_usable`, `sim_available_spanners`, `sim_nuts_to_tighten`.
    5. Start a loop that continues as long as `sim_nuts_to_tighten` is not empty:
       a. Inside the loop, find the minimum cost action sequence to tighten *any single*
          nut remaining in `sim_nuts_to_tighten`. This involves checking each remaining nut:
       b. For a specific nut `n` at location `ln`:
          i. Calculate the cost if the man uses the currently carried usable spanner
             (if applicable): `cost_carry = distance(sim_man_loc, ln) + 1` (walk + tighten).
             Set to infinity if not carrying a usable spanner or if unreachable.
          ii. Calculate the cost if the man needs to pick up a spanner: Find the "best"
              available spanner `s` in `sim_available_spanners` at location `ls`. "Best"
              minimizes `distance(sim_man_loc, ls) + 1 (pickup) + distance(ls, ln) + 1 (tighten)`.
              Let this minimum cost be `cost_pickup`. Set to infinity if no usable spanners
              are available on the ground or if paths are invalid.
          iii. The cost to tighten nut `n` in this step is `min(cost_carry, cost_pickup)`.
               Keep track of whether the pickup option was chosen and which spanner (`s`) was used.
       c. After calculating the minimum cost for each remaining nut, select the nut `n_best`
          (and potentially the spanner `s_best` to be picked up) that resulted in the
          overall minimum cost (`min_cost_for_step`) for this iteration of the loop.
       d. If `min_cost_for_step` is infinity (meaning no remaining nut could be tightened,
          possibly due to lack of spanners or reachability issues), return `math.inf`.
          This suggests the goal might be unreachable from the current state.
       e. Add `min_cost_for_step` to `total_heuristic_cost`.
       f. Update the simulation state based on the chosen action sequence for `n_best`:
          - Update `sim_man_loc` to `ln` (the location of the tightened nut `n_best`).
          - If the man *was* carrying a usable spanner and used it (`cost_carry <= cost_pickup`),
            set `sim_carrying_usable = False`.
          - If the man *picked up* a spanner `s_best` (`cost_pickup < cost_carry`), remove
            `s_best` from `sim_available_spanners`. The man remains without a usable
            spanner (`sim_carrying_usable` remains false).
          - Remove `n_best` from `sim_nuts_to_tighten`.
    6. Once the loop finishes (all nuts in the initial `nuts_to_tighten` set have been
       processed), return the accumulated `total_heuristic_cost`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing static info, goals, and precomputing distances.
        """
        static_facts = task.static
        initial_state = task.initial_state
        self.goals = task.goals

        # 1. Find Man Name (heuristic identification)
        self.man_name = None
        for fact in initial_state:
            parts = get_parts(fact)
            if not parts: continue
            # Assume man is the agent in 'at' or 'carrying'
            if parts[0] == 'at' or parts[0] == 'carrying':
                 self.man_name = parts[1]
                 break
        if self.man_name is None:
             # Fallback check for common name 'bob'
             if any('(at bob ' in fact or '(carrying bob ' in fact for fact in initial_state):
                 self.man_name = 'bob'
             else:
                 raise ValueError("SpannerHeuristic: Could not determine the man's name.")

        # 2. Find Goal Nuts and Nut Locations (assuming static locations from init state)
        self.nut_locations = {}
        for fact in initial_state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'at':
                 obj, loc = parts[1], parts[2]
                 # Basic check if object name suggests it's a nut
                 # A more robust parser would use type information.
                 if 'nut' in obj:
                     self.nut_locations[obj] = loc

        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            if parts[0] == 'tightened':
                nut = parts[1]
                self.goal_nuts.add(nut)
                # Ensure location is known for goal nuts that might be initially loose.
                # If a nut starts tightened, its location might not be strictly needed by the heuristic.


        # 3. Build Location Graph and Compute All-Pairs Shortest Paths using BFS
        self.locations = set()
        adj = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'link':
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                # Assume links are bidirectional as per typical PDDL semantics
                adj.setdefault(loc1, []).append(loc2)
                adj.setdefault(loc2, []).append(loc1)

        self.distances = {}
        all_locs_list = list(self.locations) # Use a fixed list for iteration

        for start_node in all_locs_list:
            # Initialize distances for this BFS run
            dist_from_start = {loc: math.inf for loc in all_locs_list}

            # Check if start_node is actually part of the connected graph components
            if start_node in dist_from_start:
                dist_from_start[start_node] = 0
                queue = deque([start_node])
                # Keep track of nodes added to the queue to avoid cycles and redundant processing
                processed_nodes = {start_node}

                while queue:
                    current_node = queue.popleft()
                    # Explore neighbors
                    for neighbor in adj.get(current_node, []):
                        # Process neighbor only if it's a known location and not yet added to queue
                        if neighbor in dist_from_start and neighbor not in processed_nodes:
                             dist_from_start[neighbor] = dist_from_start[current_node] + 1
                             processed_nodes.add(neighbor)
                             queue.append(neighbor)

            # Store computed distances from start_node to all other nodes
            for end_node in all_locs_list:
                # Store the distance, will be inf if end_node was unreachable
                self.distances[(start_node, end_node)] = dist_from_start[end_node]


    def _get_dist(self, loc1, loc2):
        """ Safely retrieves the precomputed distance between two locations. """
        if loc1 is None or loc2 is None:
            # Cannot compute distance if a location is unknown
            return math.inf
        if loc1 == loc2:
            return 0
        # Return precomputed distance, default to infinity if pair not found or unreachable
        return self.distances.get((loc1, loc2), math.inf)

    def __call__(self, node):
        """ Calculates the heuristic value for the given state node. """
        state = node.state

        # 1. Parse current state efficiently
        obj_locations = {} # Tracks location of objects currently 'at' somewhere
        carried_spanner = None
        usable_spanners_state = set() # Set of spanner names that are 'usable'
        loose_nuts_state = set() # Set of nut names that are 'loose'
        man_loc = None

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            pred = parts[0]
            if pred == 'at':
                obj, loc = parts[1], parts[2]
                obj_locations[obj] = loc
                if obj == self.man_name:
                    man_loc = loc
            elif pred == 'carrying':
                # Assuming man carries only one item based on domain actions
                # parts[1] should be self.man_name
                carried_spanner = parts[2]
            elif pred == 'usable':
                usable_spanners_state.add(parts[1])
            elif pred == 'loose':
                loose_nuts_state.add(parts[1])

        # Crucial check: Man must have a location
        if man_loc is None:
             # This indicates a potentially invalid state or issue finding the man
             # print(f"Warning: Man '{self.man_name}' location not found in state.")
             return math.inf # Cannot compute heuristic without man's location

        # Determine status of carried spanner
        carried_spanner_usable = (carried_spanner is not None) and (carried_spanner in usable_spanners_state)

        # Identify goal nuts that are currently loose
        nuts_to_tighten = {nut for nut in self.goal_nuts if nut in loose_nuts_state}

        # Identify usable spanners available on the ground
        available_usable_spanners = {} # Map: spanner_name -> location
        for spanner in usable_spanners_state:
            if spanner != carried_spanner:
                loc = obj_locations.get(spanner)
                if loc is not None: # Ensure the spanner is actually 'at' a location
                    available_usable_spanners[spanner] = loc
                # Else: A usable spanner not carried should be 'at' somewhere. Log warning?

        # 2. Goal Check: If no goal nuts are loose, heuristic is 0
        if not nuts_to_tighten:
            return 0

        # 3. Initialize Simulation State
        total_heuristic_cost = 0
        sim_man_loc = man_loc
        sim_carrying_usable = carried_spanner_usable
        # Create copies for simulation as these will be modified
        sim_available_spanners = available_usable_spanners.copy()
        sim_nuts_to_tighten = nuts_to_tighten.copy()

        # 4. Greedy Simulation Loop: Process nuts one by one
        while sim_nuts_to_tighten:
            min_cost_for_step = math.inf
            best_nut_to_tighten = None
            # Track if the best action involves picking up a spanner, and which one
            pickup_spanner_for_best_step = None

            # Find the cheapest nut to tighten in the current simulation state
            for nut in sim_nuts_to_tighten:
                nut_loc = self.nut_locations.get(nut)
                if nut_loc is None:
                    # This should not happen if initialization was correct and nut exists
                    # print(f"Warning: Location unknown for nut '{nut}'. Skipping.")
                    continue # Cannot plan for this nut without location

                # Calculate cost using carried spanner (if possible)
                cost_if_carrying = math.inf
                if sim_carrying_usable:
                    walk_cost = self._get_dist(sim_man_loc, nut_loc)
                    if walk_cost != math.inf:
                        cost_if_carrying = walk_cost + 1 # walk + tighten

                # Calculate cost by picking up the best available spanner (if possible)
                cost_if_pickup = math.inf
                best_pickup_spanner_for_this_nut = None
                if sim_available_spanners: # Only possible if spanners are on the ground
                    current_min_pickup_cost = math.inf
                    for spanner, spanner_loc in sim_available_spanners.items():
                        cost_walk1 = self._get_dist(sim_man_loc, spanner_loc)
                        cost_walk2 = self._get_dist(spanner_loc, nut_loc)
                        if cost_walk1 != math.inf and cost_walk2 != math.inf:
                            # Total cost: walk to spanner, pickup, walk to nut, tighten
                            total_pickup_seq_cost = cost_walk1 + 1 + cost_walk2 + 1
                            if total_pickup_seq_cost < current_min_pickup_cost:
                                current_min_pickup_cost = total_pickup_seq_cost
                                best_pickup_spanner_for_this_nut = spanner
                    cost_if_pickup = current_min_pickup_cost # Will be inf if no path found

                # Determine the minimum cost for *this* nut in this step
                current_min_cost_for_nut = min(cost_if_carrying, cost_if_pickup)

                # Update the overall best choice for *this iteration* of the outer loop
                if current_min_cost_for_nut < min_cost_for_step:
                    min_cost_for_step = current_min_cost_for_nut
                    best_nut_to_tighten = nut
                    # Record if the best option involved pickup
                    if cost_if_pickup < cost_if_carrying:
                        pickup_spanner_for_best_step = best_pickup_spanner_for_this_nut
                    else:
                        pickup_spanner_for_best_step = None # Used carried spanner or costs were equal

            # After checking all nuts, if no action is possible:
            if min_cost_for_step == math.inf:
                # Cannot tighten any remaining nut (no path, no spanners?)
                return math.inf # Indicate potential dead end

            # 5. Add cost and update simulation state based on the chosen best action
            total_heuristic_cost += min_cost_for_step
            # Man moves to the location of the nut that was just tightened
            sim_man_loc = self.nut_locations[best_nut_to_tighten]

            if pickup_spanner_for_best_step is None:
                # Implies the carried spanner was used (or costs were equal and carry was preferred)
                sim_carrying_usable = False # Spanner is consumed
            else:
                # Implies a spanner was picked up from the ground
                # Remove the used spanner from available ones in simulation
                if pickup_spanner_for_best_step in sim_available_spanners:
                     del sim_available_spanners[pickup_spanner_for_best_step]
                # Man remains without a usable spanner after picking up and using one
                sim_carrying_usable = False

            # Remove the tightened nut from the set needing processing
            sim_nuts_to_tighten.remove(best_nut_to_tighten)

        # Loop finished, all initially loose goal nuts have been processed
        return total_heuristic_cost
