import sys
from collections import deque
# Ensure Heuristic base class is correctly imported or defined
try:
    # Assuming the platform provides this base class
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the import fails (e.g., for standalone testing)
    # In the target environment, this import should work.
    print("Warning: heuristics.heuristic_base not found. Using dummy Heuristic class.", file=sys.stderr)
    class Heuristic:
        def __init__(self, task):
            """Initialize the heuristic with the planning task."""
            self.task = task
            # Example: Extract goals and static facts if needed by all heuristics
            self.goals = task.goals
            self.static = task.static

        def __call__(self, node):
            """
            Evaluate the heuristic function for a given search node.
            Must be implemented by subclasses.
            """
            raise NotImplementedError("Heuristic evaluation not implemented.")

# Helper function to parse PDDL facts "(pred obj1 obj2 ...)" -> ["pred", "obj1", "obj2", ...]
def get_parts(fact):
    """Extract the components of a PDDL fact string."""
    # Example: "(at bob shed)" -> ["at", "bob", "shed"]
    return fact[1:-1].split()

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL spanner domain.

    # Summary
    This heuristic estimates the number of actions required to reach a goal state
    where all specified nuts are tightened. It simulates a greedy strategy where
    the single man repeatedly travels to the nearest loose nut, acquiring a
    usable spanner along the way if necessary. The cost includes tighten actions,
    pickup actions, and estimated travel distance based on precomputed shortest paths.

    # Assumptions
    - There is exactly one 'man' agent in the problem instance.
    - Nuts do not change their location throughout the plan.
    - Location links (`link` predicates) are static and represent an undirected graph.
    - The graph of locations is connected for relevant parts of the problem. If locations
      are disconnected, distances are treated as infinite (represented by a large penalty value).
    - There are enough usable spanners available (either carried or on the ground)
      to tighten all the required loose nuts. If the simulation finds insufficient
      or unreachable spanners, it adds a large penalty, potentially overestimating the cost.
    - The heuristic uses a greedy approach for selecting the next nut and spanner
      based on proximity (shortest path distance), which may not reflect the optimal plan's choices.

    # Heuristic Initialization
    - Stores the goal conditions (specifically `tightened` predicates).
    - Identifies all location, man, nut, and spanner objects by parsing the initial state
      and static facts. Makes assumptions about object types based on predicate usage
      (e.g., object in `(carrying ?m ?s)` is man or spanner).
    - Parses the `link` predicates from static facts to build an adjacency list
      representation of the location graph.
    - Precomputes all-pairs shortest path distances between locations using Breadth-First Search (BFS).
      Handles disconnected locations by assigning infinite distance.
    - Precomputes the fixed locations of all 'nut' objects based on the initial state.
    - Stores the set of all 'nut' objects mentioned in the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Unmet Goals:** Determine the set of goal nuts that are currently `loose`
        in the given state (`node.state`). If this set is empty and all goal nuts are `tightened`,
        the goal is reached, and the heuristic value is 0.
    2.  **Base Cost:** Initialize the heuristic estimate `h` to the number of loose goal nuts.
        This accounts for the `tighten_nut` action required for each.
    3.  **Get Current State Info:** Parse the current state to find:
        - The man's current location (`current_man_loc`). If not found, return infinity.
        - Whether the man is currently carrying a spanner and if that spanner is `usable`
          (`current_has_usable_spanner`).
        - The set of all `usable` spanners currently on the ground and their locations
          (`available_ground_spanners`).
    4.  **Greedy Simulation Loop:** While there are still loose goal nuts remaining in the simulation:
        a.  **Select Nut:** Find the remaining loose goal nut (`closest_nut`) whose location
            (`closest_nut_loc`) has the minimum shortest path distance from `current_man_loc`.
            If no remaining nuts are reachable (all have infinite distance), add a large penalty
            for each remaining nut and break the loop.
        b.  **Check Spanner:**
            i.  **If `current_has_usable_spanner` is True:**
                - Add the travel distance (`distance(current_man_loc, closest_nut_loc)`) to `h`.
                - Update the simulated man's location `current_man_loc` to `closest_nut_loc`.
                - Set `current_has_usable_spanner` to `False` (simulating spanner use).
            ii. **Else (man needs a spanner):**
                - **Select Spanner:** Find the available usable ground spanner (`closest_spanner`)
                  whose location (`closest_spanner_loc`) is nearest (minimum distance) to `current_man_loc`.
                - **If no usable ground spanner is found:** Add a large penalty for each
                  remaining nut (as they cannot be tightened in this simulation) and break the loop.
                - **If no usable ground spanner is reachable:** Add a large penalty and break.
                - **Calculate Path & Costs:**
                    - Add distance to spanner (`distance(current_man_loc, closest_spanner_loc)`) to `h`.
                    - Add 1 to `h` (for `pickup_spanner` action).
                    - Remove `closest_spanner` from the simulated `available_ground_spanners`.
                    - Calculate distance from spanner to nut (`distance(closest_spanner_loc, closest_nut_loc)`).
                    - If nut is unreachable from spanner (infinite distance), add large penalty for remaining nuts and break.
                    - Add this distance to `h`.
                - **Update Location:** Update `current_man_loc` to `closest_nut_loc`.
                - `current_has_usable_spanner` remains `False` (spanner consumed by tighten).
        c.  **Update Goal:** Remove `closest_nut` from the set of simulated remaining loose goal nuts.
    5.  **Return Value:** Return the final computed value of `h`.
    """
    # Define a large number to represent infinite distance or unreachable penalty
    INFINITE_DISTANCE_PENALTY = 9999

    def __init__(self, task):
        """Initialize the heuristic by precomputing distances and object info."""
        super().__init__(task)
        initial_state = task.initial_state

        # --- Identify Objects ---
        self.man = None
        self.nuts = set()
        self.spanners = set()
        self.locations = set()

        # Infer objects and types from initial state and static facts
        all_objects_in_init = set()
        obj_locations_init = {} # Track initial locations {obj: loc}
        obj_carrying_init = {} # Track initial carrying {man: spanner}
        obj_usable_init = set() # Track initial usable spanners

        for fact in initial_state:
             parts = get_parts(fact)
             pred = parts[0]
             if len(parts) < 2: continue # Skip malformed facts
             arg1 = parts[1]
             all_objects_in_init.add(arg1)
             if len(parts) > 2:
                 arg2 = parts[2]
                 all_objects_in_init.add(arg2)

             if pred == 'at':
                 obj, loc = arg1, arg2
                 obj_locations_init[obj] = loc
             elif pred == 'carrying':
                 man_obj, spanner_obj = arg1, arg2
                 obj_carrying_init[man_obj] = spanner_obj
             elif pred == 'usable':
                 spanner_obj = arg1
                 obj_usable_init.add(spanner_obj)
             elif pred == 'loose' or pred == 'tightened':
                 nut_obj = arg1
                 self.nuts.add(nut_obj) # Identify nuts

        objects_in_static = set()
        for fact in self.static:
             parts = get_parts(fact)
             pred = parts[0]
             if len(parts) < 2: continue
             arg1 = parts[1]
             objects_in_static.add(arg1)
             if len(parts) > 2:
                 arg2 = parts[2]
                 objects_in_static.add(arg2)

             if pred == 'link':
                 loc1, loc2 = arg1, arg2
                 self.locations.add(loc1) # Identify locations
                 self.locations.add(loc2)

        all_objects = all_objects_in_init.union(objects_in_static)

        # Refine object types
        potential_men = set(obj_carrying_init.keys())
        self.spanners = set(obj_carrying_init.values()).union(obj_usable_init)
        # Assume anything 'at' a location that isn't a nut or spanner is the man or location
        for obj, loc in obj_locations_init.items():
            self.locations.add(loc) # Ensure location is known
            if obj not in self.nuts and obj not in self.spanners:
                # Could be man or another object type if domain expands
                potential_men.add(obj)

        # Finalize Man identification (assuming exactly one)
        if len(potential_men) == 1:
            self.man = list(potential_men)[0]
        elif len(potential_men) > 1:
             men_who_carry = set(obj_carrying_init.keys())
             if len(men_who_carry) == 1:
                 self.man = list(men_who_carry)[0]
             else: # Fallback guess
                 guessed_man = next((p for p in potential_men if 'man' in p or 'bob' in p), None)
                 self.man = guessed_man if guessed_man else list(potential_men)[0]
                 print(f"Warning: Multiple potential 'man' objects ({potential_men}). Assuming '{self.man}'.", file=sys.stderr)
        else:
             raise ValueError("Could not identify the 'man' object in the task.")

        # Ensure all objects involved are categorized (basic typing)
        for obj in all_objects:
             is_man = obj == self.man
             is_nut = obj in self.nuts
             is_spanner = obj in self.spanners
             is_location = obj in self.locations
             if not (is_man or is_nut or is_spanner or is_location):
                 # If untyped, try to guess or assume location as default
                 if 'spanner' in obj: self.spanners.add(obj)
                 elif 'nut' in obj: self.nuts.add(obj)
                 else: self.locations.add(obj) # Default assumption

        # --- Goal Nuts ---
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'tightened':
                if parts[1] in self.nuts:
                    self.goal_nuts.add(parts[1])
                else:
                    print(f"Warning: Goal refers to tightening '{parts[1]}', which was not identified as a nut.", file=sys.stderr)

        # --- Nut Locations (from initial state) ---
        self.nut_locations = {}
        for nut in self.nuts:
            if nut in obj_locations_init:
                self.nut_locations[nut] = obj_locations_init[nut]
            else:
                print(f"Warning: Initial location for nut '{nut}' not found.", file=sys.stderr)


        # --- Location Graph and Distances ---
        adj = {loc: [] for loc in self.locations} # Initialize all known locations
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                if l1 in self.locations and l2 in self.locations:
                    adj[l1].append(l2)
                    adj[l2].append(l1)
                else:
                    print(f"Warning: Link between '{l1}' and '{l2}' involves unknown location(s). Ignoring link.", file=sys.stderr)

        self.distances = {}
        for loc in self.locations:
            self.distances[loc] = self._bfs(adj, loc, self.locations)


    def _bfs(self, graph, start_node, all_nodes):
        """Performs BFS to find shortest path distances from start_node."""
        distances = {node: float('inf') for node in all_nodes}
        if start_node not in distances:
             print(f"Error: BFS start node '{start_node}' not in known locations.", file=sys.stderr)
             return distances # Should not happen if graph keys match all_nodes
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            # Use graph.get for safety, though neighbors should be locations
            for neighbor in graph.get(current_node, []):
                if neighbor in distances and distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def _get_distance(self, loc1, loc2):
        """Gets the precomputed distance between two locations."""
        if loc1 is None or loc2 is None:
             # print(f"Warning: Trying to get distance for None location ({loc1}, {loc2}).", file=sys.stderr)
             return self.INFINITE_DISTANCE_PENALTY

        if loc1 == loc2:
            return 0

        # Check if locations are known before dictionary access
        if loc1 not in self.distances or loc2 not in self.distances.get(loc1, {}):
             # This implies loc1 or loc2 was not identified as a location during init, or graph is incomplete.
             # print(f"Warning: Distance query for unknown locations: {loc1}, {loc2}.", file=sys.stderr)
             return self.INFINITE_DISTANCE_PENALTY

        dist = self.distances.get(loc1, {}).get(loc2, float('inf'))

        return dist if dist != float('inf') else self.INFINITE_DISTANCE_PENALTY

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the current node."""
        state = node.state

        # --- Find loose goal nuts ---
        loose_goal_nuts = {} # Map nut -> location
        goal_met_count = 0
        for nut in self.goal_nuts:
            loose_pred = f'(loose {nut})'
            tightened_pred = f'(tightened {nut})'

            if loose_pred in state:
                nut_loc = self.nut_locations.get(nut)
                if nut_loc:
                    loose_goal_nuts[nut] = nut_loc
                else: # Fallback: find location in current state (should be static)
                    found_loc = next((get_parts(f)[2] for f in state if get_parts(f)[0] == 'at' and get_parts(f)[1] == nut), None)
                    if found_loc:
                        loose_goal_nuts[nut] = found_loc
                    else:
                        print(f"Error: Location for loose goal nut {nut} not found.", file=sys.stderr)
                        return float('inf') # Cannot proceed reliably
            elif tightened_pred in state:
                 goal_met_count += 1
            # else: nut is neither loose nor tightened (initial state error?)

        # Check if all goals are met
        if not loose_goal_nuts and goal_met_count == len(self.goal_nuts):
             return 0

        # --- Find current state: man location, carried spanner, usable spanners ---
        man_loc = None
        carried_spanner = None
        usable_status = {} # Map spanner -> bool (is usable?)
        spanner_locations = {} # Map spanner -> location (if not carried)

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if len(parts) < 2: continue
            arg1 = parts[1]
            if pred == 'at':
                loc = parts[2]
                if arg1 == self.man:
                    man_loc = loc
                elif arg1 in self.spanners:
                    spanner_locations[arg1] = loc
            elif pred == 'carrying':
                spanner_obj = parts[2]
                if arg1 == self.man and spanner_obj in self.spanners:
                    carried_spanner = spanner_obj
            elif pred == 'usable':
                if arg1 in self.spanners:
                    usable_status[arg1] = True

        if man_loc is None:
             print(f"Error: Man '{self.man}' location not found in state: {state}", file=sys.stderr)
             return float('inf') # Cannot calculate heuristic

        usable_carried_spanner = None
        if carried_spanner and usable_status.get(carried_spanner, False):
            usable_carried_spanner = carried_spanner

        available_ground_spanners = {} # Map usable spanner -> location
        for spanner, loc in spanner_locations.items():
            if usable_status.get(spanner, False):
                available_ground_spanners[spanner] = loc

        # --- Heuristic Calculation ---
        h = 0
        current_man_loc = man_loc
        current_has_usable_spanner = (usable_carried_spanner is not None)
        # Make copies to modify during simulation
        sim_available_ground_spanners = available_ground_spanners.copy()
        sim_remaining_loose_nuts = loose_goal_nuts.copy()

        # Add base cost for tighten actions
        h += len(sim_remaining_loose_nuts)

        while sim_remaining_loose_nuts:
            # Find closest reachable remaining loose nut
            closest_nut = None
            closest_nut_loc = None
            min_dist_to_nut = float('inf')

            if current_man_loc is None: # Should not happen if initial check passed
                 print("Error: current_man_loc became None during simulation.", file=sys.stderr)
                 h += len(sim_remaining_loose_nuts) * self.INFINITE_DISTANCE_PENALTY
                 break

            found_reachable_nut = False
            for nut, loc in sim_remaining_loose_nuts.items():
                dist = self._get_distance(current_man_loc, loc)
                if dist < min_dist_to_nut:
                    min_dist_to_nut = dist
                    closest_nut = nut
                    closest_nut_loc = loc
                if dist != self.INFINITE_DISTANCE_PENALTY:
                    found_reachable_nut = True # At least one nut is reachable

            # If no nuts are reachable from current location, penalize and stop
            if not found_reachable_nut and sim_remaining_loose_nuts:
                 # print(f"Warning: No remaining loose nuts are reachable from {current_man_loc}.", file=sys.stderr)
                 h += len(sim_remaining_loose_nuts) * self.INFINITE_DISTANCE_PENALTY
                 break

            # If the closest nut itself is unreachable (min_dist is penalty), stop.
            if min_dist_to_nut == self.INFINITE_DISTANCE_PENALTY:
                 # print(f"Warning: Closest nut {closest_nut} is unreachable from {current_man_loc}.", file=sys.stderr)
                 h += len(sim_remaining_loose_nuts) * self.INFINITE_DISTANCE_PENALTY
                 break

            if closest_nut is None: # Should not happen if loop condition met
                break

            if current_has_usable_spanner:
                # Move to nut
                h += min_dist_to_nut # Add the calculated distance
                current_man_loc = closest_nut_loc
                current_has_usable_spanner = False # Spanner used
                del sim_remaining_loose_nuts[closest_nut]
            else:
                # Need to get a spanner
                # Find closest reachable available ground spanner
                closest_spanner = None
                closest_spanner_loc = None
                min_dist_to_spanner = float('inf')
                found_reachable_spanner = False

                if not sim_available_ground_spanners:
                     # print("Warning: No usable ground spanners left, but needed.", file=sys.stderr)
                     h += len(sim_remaining_loose_nuts) * self.INFINITE_DISTANCE_PENALTY
                     break

                for spanner, loc in sim_available_ground_spanners.items():
                    dist = self._get_distance(current_man_loc, loc)
                    if dist < min_dist_to_spanner:
                        min_dist_to_spanner = dist
                        closest_spanner = spanner
                        closest_spanner_loc = loc
                    if dist != self.INFINITE_DISTANCE_PENALTY:
                        found_reachable_spanner = True # At least one spanner is reachable

                # If no spanners are reachable, penalize and stop
                if not found_reachable_spanner and sim_available_ground_spanners:
                     # print(f"Warning: No usable ground spanners are reachable from {current_man_loc}.", file=sys.stderr)
                     h += len(sim_remaining_loose_nuts) * self.INFINITE_DISTANCE_PENALTY
                     break

                # If the closest spanner is unreachable, stop
                if min_dist_to_spanner == self.INFINITE_DISTANCE_PENALTY:
                     # print(f"Warning: Closest spanner {closest_spanner} is unreachable from {current_man_loc}.", file=sys.stderr)
                     h += len(sim_remaining_loose_nuts) * self.INFINITE_DISTANCE_PENALTY
                     break

                if closest_spanner is None: # Should not happen
                     break

                # Move to spanner
                h += min_dist_to_spanner
                # Pickup spanner
                h += 1
                # Move from spanner to nut
                move_to_nut_cost = self._get_distance(closest_spanner_loc, closest_nut_loc)
                if move_to_nut_cost == self.INFINITE_DISTANCE_PENALTY:
                     # print(f"Warning: Cannot reach nut {closest_nut} from spanner {closest_spanner} at {closest_spanner_loc}.", file=sys.stderr)
                     # This path failed. Penalize remaining nuts and stop.
                     h += len(sim_remaining_loose_nuts) * self.INFINITE_DISTANCE_PENALTY
                     break

                h += move_to_nut_cost

                current_man_loc = closest_nut_loc # Man is now at the nut location
                current_has_usable_spanner = False # Spanner used immediately
                del sim_available_ground_spanners[closest_spanner] # Spanner is picked up
                del sim_remaining_loose_nuts[closest_nut] # Nut is tightened

        return h
