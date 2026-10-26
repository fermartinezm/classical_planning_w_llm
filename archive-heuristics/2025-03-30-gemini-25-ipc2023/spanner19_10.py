import heapq
from collections import deque
from fnmatch import fnmatch
import math # Import math for infinity

# Try to import the standard Heuristic base class.
# If running standalone or in a different environment, define a dummy class.
try:
    # This assumes the heuristic is used within a framework where this path is valid.
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the import fails (e.g., for standalone testing)
    print("Warning: heuristics.heuristic_base not found. Using dummy Heuristic class.")
    class Heuristic:
        """Dummy base class for Heuristic if standard import fails."""
        def __init__(self, task):
            """Initializes the heuristic with the planning task."""
            self.task = task
        def __call__(self, node):
            """Calculates the heuristic value for a given node."""
            raise NotImplementedError("Heuristic calculation not implemented.")

# Utility functions for parsing PDDL facts
def get_parts(fact):
    """
    Extracts the predicate and arguments from a PDDL fact string.
    Example: "(at bob shed)" -> ["at", "bob", "shed"]
    """
    # Removes surrounding parentheses and splits by space
    return fact[1:-1].split()

def match(fact, *args):
    """
    Checks if a PDDL fact matches a given pattern.
    Supports '*' wildcards in the pattern arguments using fnmatch.
    Example: match("(at bob shed)", "at", "*", "shed") -> True
             match("(at bob shed)", "at", "alice", "*") -> False
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the pattern length
    if len(parts) != len(args):
        return False
    # Check each part against the corresponding pattern argument using fnmatch
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts
    that are currently loose. It calculates the cost by greedily simulating the
    process of tightening nuts one by one. In each step, it selects the nut that
    can be tightened with the minimum estimated cost from the man's current
    simulated location. The cost calculation considers travel distance (number of
    'walk' actions computed via BFS) and the necessary actions (pickup, tighten).
    It correctly accounts for the fact that each tighten action consumes a unique
    usable spanner.

    # Assumptions
    - There is only one 'man' agent in the problem instance.
    - 'link' predicates define traversable paths between locations and are
      assumed to allow bidirectional travel for the 'walk' action.
    - The planning goal is solely defined by a conjunction of `(tightened nut)` predicates.
    - Each `tighten_nut` action requires the man to carry a `usable` spanner,
      and applying the action consumes the `usable` status of that spanner.

    # Heuristic Initialization
    - Parses the static `link` facts provided in `task.static` to build an
      adjacency list representation of the location graph.
    - Computes all-pairs shortest path distances (in terms of number of 'walk'
      actions) between all connected locations using Breadth-First Search (BFS).
      These distances are stored for efficient lookup during heuristic calculation.
    - Identifies and stores the set of nuts that must be tightened to satisfy
      the goal conditions (`self.goal_nuts`).
    - Attempts to identify the man object, all spanner objects, and all nut objects
      from the initial state predicates for more robust state parsing later.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Extract current facts from the input `node.state`.
        Identify:
        - The man's current location (`man_loc`).
        - The locations of all nuts (`nut_locs`) and spanners (`spanner_locs`).
        - The set of nuts that are currently `loose`.
        - The set of spanners that are currently `usable`.
        - Whether the man is `carrying` a spanner, and if so, which one (`carrying_spanner`).
    2.  **Identify Pending Goals:** Determine the subset of `self.goal_nuts` that
        are also in the set of `loose_nuts` found in the current state. This gives
        the set `nuts_to_tighten`. If this set is empty, the goal is satisfied,
        and the heuristic value is 0.
    3.  **Check Solvability (Spanner Availability):** Count the total number of
        currently usable spanners (those `usable` and located on the ground, plus
        the one `usable` and `carrying`, if any). If this count is less than the
        number of `nuts_to_tighten`, the goal is definitely unreachable from this
        state; return infinity (`math.inf`).
    4.  **Greedy Sequential Tightening Simulation:** Simulate the process step-by-step:
        a. Initialize `total_cost = 0`.
        b. Keep track of the simulated man location (`current_man_loc`), initialized
           to the actual `man_loc` from the state.
        c. Maintain the set of usable spanners still available on the ground
           (`remaining_ground_spanners`).
        d. Track whether the initially carried spanner (if any and usable) is still
           available for use (`carried_spanner_available`).
        e. Maintain the set of nuts still needing to be tightened (`nuts_to_process`).
        f. **Loop** as long as `nuts_to_process` is not empty:
            i.   **Find Best Next Step:** Iterate through each `nut` in `nuts_to_process`.
                 For each nut, calculate the cost to tighten it *in this step*,
                 considering two possibilities:
                 - **Option 1 (Use Carried Spanner):** If `carried_spanner_available`,
                   the cost is `distance(current_man_loc, nut_loc) + 1 (tighten)`.
                   Calculate the distance using the precomputed BFS results.
                 - **Option 2 (Use Ground Spanner):** If `remaining_ground_spanners`
                   is not empty, find the ground spanner `s*` that minimizes the
                   combined travel distance:
                   `distance(current_man_loc, spanner_loc(s*)) + distance(spanner_loc(s*), nut_loc)`.
                   The total cost for this option is
                   `min_travel_dist + 1 (pickup) + 1 (tighten)`.
            ii.  **Select Minimum Cost Action:** Compare the minimum costs calculated
                 for all nuts in `nuts_to_process`. Select the action (tightening a
                 specific `nut` using either the 'carried' or a 'ground' spanner)
                 that has the absolute lowest cost among all possibilities in this step.
                 Store this as `(chosen_nut, chosen_type, step_cost, chosen_spanner)`.
            iii. **Handle Unreachability:** If no nut can be tightened in this step
                 (e.g., all remaining nuts or the required spanners are in locations
                 unreachable from `current_man_loc`), return infinity (`math.inf`).
            iv.  **Update State for Next Iteration:**
                 - Add the `step_cost` of the chosen action to `total_cost`.
                 - Update `current_man_loc` to the location of the `chosen_nut` (as
                   the man must be there to tighten it).
                 - Remove `chosen_nut` from `nuts_to_process`.
                 - Update spanner availability based on `chosen_type`: if 'carried',
                   set `carried_spanner_available = False`. If 'ground', remove the
                   `chosen_spanner` from `remaining_ground_spanners`.
        g. **Return `total_cost`** after the loop finishes (all nuts processed).
    """
    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # --- Initialization ---
        locations = set()
        links = set()
        # Extract locations and links from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                links.add(fact)
                locations.add(parts[1])
                locations.add(parts[2])
            # Add any locations mentioned in 'at' predicates in the initial state
            # This ensures disconnected locations are included in distance calculations (as inf)
            elif parts[0] == 'at':
                 locations.add(parts[2])


        # Compute all-pairs shortest paths using BFS
        self.distances = self._compute_distances(locations, links)

        # Store goal nuts for quick lookup
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Identify all spanner, nut, and man objects from the initial state if possible
        self.all_spanners = set()
        self.all_nuts = set()
        self.man_object = None
        # Use initial state facts for identification
        for fact in task.initial_state:
             parts = get_parts(fact)
             pred = parts[0]
             args = parts[1:]
             if pred == 'usable':
                 self.all_spanners.add(args[0])
             elif pred == 'at':
                 obj, loc = args
                 # Basic type inference based on common PDDL structures/names
                 # This helps if an object isn't mentioned elsewhere initially
                 if obj.startswith('spanner'): self.all_spanners.add(obj)
                 elif obj.startswith('nut'): self.all_nuts.add(obj)
                 # Assume the first object 'at' somewhere that isn't a nut/spanner is the man
                 elif self.man_object is None and not obj.startswith('spanner') and not obj.startswith('nut'):
                     self.man_object = obj
                 # Add location to the set if not seen before
                 locations.add(loc)
             elif pred == 'loose':
                 self.all_nuts.add(args[0])
             elif pred == 'carrying':
                 # 'carrying' predicate strongly identifies the man and a spanner
                 self.man_object = args[0]
                 self.all_spanners.add(args[1])

        # Ensure goal nuts are included in the set of all nuts
        self.all_nuts.update(self.goal_nuts)

        # If the man object wasn't identified, log a warning. The heuristic might fail later.
        if self.man_object is None:
            print("WARNING: spannerHeuristic could not reliably identify the man object during initialization.")


    def _compute_distances(self, locations, links):
        """
        Computes shortest path distances (number of links/walk actions) between
        all pairs of locations using BFS.
        Handles disconnected graphs (distance will be infinity).
        """
        distances = {}
        adj = {loc: [] for loc in locations}

        # Build adjacency list from link facts
        for fact in links:
            parts = get_parts(fact)
            l1, l2 = parts[1], parts[2]
            # Ensure locations exist in adj before adding edges (should exist from parsing)
            if l1 in adj: adj[l1].append(l2)
            if l2 in adj: adj[l2].append(l1) # Assume links are bidirectional

        # Run BFS from each location to find shortest paths to all others
        for start_node in locations:
            # Initialize distances from start_node to infinity
            for loc in locations:
                distances[(start_node, loc)] = math.inf
            distances[(start_node, start_node)] = 0 # Distance to self is 0

            queue = deque([(start_node, 0)]) # Queue stores (location, distance)
            visited_dist = {start_node: 0} # Track shortest distance found so far

            while queue:
                current_node, dist = queue.popleft()

                # Explore neighbors
                for neighbor in adj.get(current_node, []):
                    # If neighbor not visited or found a shorter path
                    if neighbor not in visited_dist or visited_dist[neighbor] > dist + 1:
                        visited_dist[neighbor] = dist + 1
                        distances[(start_node, neighbor)] = dist + 1
                        queue.append((neighbor, dist + 1))
        return distances

    def _get_dist(self, loc1, loc2):
        """
        Gets the precomputed shortest distance between two locations.
        Returns math.inf if locations are unreachable or not in the graph.
        """
        if loc1 == loc2:
            return 0 # Distance from a location to itself is 0
        return self.distances.get((loc1, loc2), math.inf)

    def __call__(self, node):
        """Calculates the heuristic value for the given state node."""
        state = node.state

        # --- 1. Parse Current State ---
        man_loc = None
        current_man_obj = self.man_object # Use pre-identified man if available
        nut_locs = {}
        spanner_locs = {}
        loose_nuts = set()
        usable_spanners = set()
        carrying_spanner = None
        carried_by = None # Store who is carrying, should match current_man_obj

        # First pass: Collect facts about loose nuts, usable spanners, and carrying
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            args = parts[1:]
            if pred == "loose":
                loose_nuts.add(args[0])
            elif pred == "usable":
                usable_spanners.add(args[0])
            elif pred == "carrying":
                carried_by, carrying_spanner = args
                # If man wasn't identified in init, try to identify now
                if current_man_obj is None:
                    current_man_obj = carried_by
                # Sanity check: ensure only one man is carrying
                elif current_man_obj != carried_by:
                     print(f"WARNING: Multiple agents carrying? Expected {current_man_obj}, found {carried_by}")


        # Second pass: Find locations using 'at' predicates
        for fact in state:
             parts = get_parts(fact)
             pred = parts[0]
             args = parts[1:]
             if pred == "at":
                 obj, loc = args
                 if obj == current_man_obj:
                     man_loc = loc
                 # Use pre-identified sets for nuts/spanners if possible
                 elif self.all_nuts and obj in self.all_nuts:
                     nut_locs[obj] = loc
                 elif self.all_spanners and obj in self.all_spanners:
                     spanner_locs[obj] = loc
                 # Fallback: Guess type by name if sets were incomplete
                 elif obj.startswith("nut"):
                      if obj not in nut_locs: nut_locs[obj] = loc
                 elif obj.startswith("spanner"):
                      if obj not in spanner_locs: spanner_locs[obj] = loc
                 # If man wasn't identified yet and this object isn't nut/spanner
                 elif current_man_obj is None and carried_by is None and not obj.startswith("nut") and not obj.startswith("spanner"):
                      current_man_obj = obj
                      man_loc = loc

        # Final check: Ensure man's location was found
        if man_loc is None:
             # If the man object itself wasn't identified, we can't proceed
             if current_man_obj is None:
                  print("FATAL ERROR: spannerHeuristic could not identify the man object in the current state.")
                  return math.inf
             # If the man object is known but has no 'at' fact, state is likely inconsistent
             # print(f"Warning: Could not find 'at' predicate for man '{current_man_obj}' in state {state}")
             return math.inf # Cannot calculate heuristic without knowing man's location

        # --- 2. Identify Pending Goals ---
        nuts_to_tighten = self.goal_nuts.intersection(loose_nuts)
        # If no nuts need tightening, the goal is reached (or already satisfied)
        if not nuts_to_tighten:
            return 0

        # --- 3. Check Solvability (Spanner Availability) ---
        # Find usable spanners currently on the ground
        available_usable_spanners_ground = {s for s in usable_spanners if s in spanner_locs}
        # Check if the carried spanner (if one exists) is usable
        carried_usable_spanner = carrying_spanner if carrying_spanner and carrying_spanner in usable_spanners else None
        # Count total usable spanners
        total_usable_spanners_count = len(available_usable_spanners_ground) + (1 if carried_usable_spanner else 0)

        # If not enough usable spanners for the remaining loose goal nuts, return infinity
        if len(nuts_to_tighten) > total_usable_spanners_count:
            return math.inf

        # --- 4. Greedy Sequential Tightening Simulation ---
        current_man_loc = man_loc
        # Copy the set of usable ground spanners to modify during simulation
        remaining_ground_spanners = set(available_usable_spanners_ground)
        # Track if the initially carried usable spanner is still available
        carried_spanner_available = carried_usable_spanner is not None
        total_cost = 0
        # Copy the set of nuts to tighten to modify during simulation
        nuts_to_process = set(nuts_to_tighten)

        # Loop until all required nuts have been processed in the simulation
        while nuts_to_process:
            best_choice = None # Stores (nut, type, step_cost, spanner_obj) for the best action this iteration
            min_total_step_cost = math.inf # Min cost found across all nuts this iteration

            # Iterate through each nut that still needs tightening
            for nut in nuts_to_process:
                # Ensure the location of this nut is known from the current state
                if nut not in nut_locs:
                    # Should not happen if state is consistent and parsing worked
                    # print(f"Warning: Location for nut '{nut}' not found in state. Skipping for heuristic.")
                    continue # Skip this nut if location unknown

                nut_loc = nut_locs[nut]

                cost_step_carried = math.inf
                cost_step_ground = math.inf
                best_ground_spanner_for_nut = None # Specific spanner object name

                # --- Calculate cost for Option 1: Use carried spanner ---
                if carried_spanner_available:
                    walk_cost = self._get_dist(current_man_loc, nut_loc)
                    # Only possible if the nut location is reachable
                    if walk_cost != math.inf:
                        # Cost = walk actions + 1 tighten action
                        cost_step_carried = walk_cost + 1

                # --- Calculate cost for Option 2: Use a ground spanner ---
                current_best_ground_spanner = None # Track best spanner *for this nut*
                min_ground_walk_pickup_walk_cost = math.inf # Track min travel *for this nut*

                # Only possible if there are usable spanners left on the ground
                if remaining_ground_spanners:
                    # Check each available ground spanner
                    for s in remaining_ground_spanners:
                        # Ensure spanner location is known
                        if s not in spanner_locs: continue
                        spanner_loc = spanner_locs[s]

                        # Calculate travel: man -> spanner -> nut
                        walk1 = self._get_dist(current_man_loc, spanner_loc)
                        walk2 = self._get_dist(spanner_loc, nut_loc)

                        # Only consider if both parts of the path are possible
                        if walk1 != math.inf and walk2 != math.inf:
                            current_walk_cost = walk1 + walk2
                            # If this spanner offers a shorter path than others for this nut
                            if current_walk_cost < min_ground_walk_pickup_walk_cost:
                                min_ground_walk_pickup_walk_cost = current_walk_cost
                                current_best_ground_spanner = s

                    # If a reachable ground spanner was found for this nut
                    if current_best_ground_spanner is not None:
                        best_ground_spanner_for_nut = current_best_ground_spanner
                        # Cost = walk_to_spanner + pickup + walk_to_nut + tighten
                        cost_step_ground = min_ground_walk_pickup_walk_cost + 1 + 1

                # --- Compare options for this nut and update overall best choice ---
                current_nut_min_cost = math.inf
                chosen_type_for_nut = None
                chosen_spanner_for_nut = None

                # Determine the cheaper way to tighten *this* specific nut
                if cost_step_carried <= cost_step_ground: # Favor carried if costs are equal
                    current_nut_min_cost = cost_step_carried
                    chosen_type_for_nut = 'carried'
                    chosen_spanner_for_nut = carried_usable_spanner # Name of the carried spanner
                elif cost_step_ground != math.inf: # Check if ground option was possible
                    current_nut_min_cost = cost_step_ground
                    chosen_type_for_nut = 'ground'
                    chosen_spanner_for_nut = best_ground_spanner_for_nut # Name of the best ground spanner
                else:
                    # Neither option was possible for this nut (e.g., unreachable)
                     continue # Skip to the next nut in nuts_to_process

                # Check if tightening this nut (via its best option) is the cheapest
                # action overall found *in this iteration* across all nuts
                if current_nut_min_cost < min_total_step_cost:
                    min_total_step_cost = current_nut_min_cost
                    # Store the details of this best action found so far
                    best_choice = (nut, chosen_type_for_nut, current_nut_min_cost, chosen_spanner_for_nut)


            # --- End of loop checking all nuts for the current step ---

            # If no choice was found after checking all remaining nuts, it means none are reachable
            if best_choice is None:
                # This implies remaining nuts/spanners are in unreachable parts of the map
                return math.inf

            # --- Apply the best choice found in this iteration ---
            chosen_nut, chosen_type, step_cost, chosen_spanner = best_choice

            # Accumulate the estimated cost for this step
            total_cost += step_cost

            # --- Update simulated state for the next iteration ---
            # Man moves to the nut's location to perform the tighten action
            current_man_loc = nut_locs[chosen_nut]
            # Remove the processed nut from the set
            nuts_to_process.remove(chosen_nut)
            # Update spanner availability based on which one was used
            if chosen_type == 'carried':
                carried_spanner_available = False # Carried spanner is now considered used
            else: # 'ground' type was used
                # Remove the specific ground spanner that was chosen
                if chosen_spanner in remaining_ground_spanners:
                     remaining_ground_spanners.remove(chosen_spanner)

        # End of while loop (all nuts processed)
        return int(round(total_cost)) # Return the total estimated cost as an integer

