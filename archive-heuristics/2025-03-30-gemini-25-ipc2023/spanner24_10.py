import heapq
from collections import deque
from fnmatch import fnmatch
# Assuming the heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic
import copy
import math # For infinity

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handles facts like '(at bob shed)' -> ['at', 'bob', 'shed']
    # Handles facts like '(tightened nut1)' -> ['tightened', 'nut1']
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern. Wildcards (*) are allowed.
    Example: match('(at bob shed)', 'at', '*', 'shed') -> True
             match('(tightened nut1)', 'tightened', 'nut*') -> True
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It simulates the process of the 'man' sequentially tightening the remaining loose goal nuts,
    always choosing the nut that seems cheapest to tighten next based on the current state (man's location,
    carried spanner, available usable spanners). The cost includes walking, picking up spanners, and tightening nuts.
    It is designed for Greedy Best-First Search and is likely non-admissible but aims to be informative.

    # Assumptions
    - There is exactly one 'man' agent in the problem instance.
    - Links between locations ('link' predicate) are static and represent bidirectional connections with a uniform cost of 1 per traversal.
    - Nuts do not move from their initial locations specified in the problem's initial state.
    - Each 'tighten_nut' action requires one 'usable' spanner and consumes its usability (makes it not usable).
    - The goal is solely defined by a conjunction of '(tightened ?nut)' predicates for a subset of nuts.

    # Heuristic Initialization
    - The constructor (`__init__`) parses the task definition (`task`) once.
    - It identifies the single 'man' agent, all 'nuts', 'spanners', and 'locations' present in the problem. It uses predicate information and naming conventions (like 'nut*', 'spanner*') as heuristics if explicit type information isn't readily available.
    - It determines and stores the static location of each nut (`self.nut_locations`).
    - It identifies the set of nuts that must be tightened to satisfy the goal conditions (`self.goal_nuts`).
    - It pre-computes all-pairs shortest path distances between all known locations using Breadth-First Search (BFS) based on the static 'link' predicates. These distances are stored in `self.distances`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Remaining Work:** In the `__call__` method, given the current `state`, determine the set of goal nuts (`self.goal_nuts`) that are *not* currently 'tightened'. If this set is empty, the goal state is reached, and the heuristic value is 0.
    2.  **Snapshot Current State:** Parse the current `state` (a set of facts) to find:
        - The man's current location (`man_loc`).
        - The spanner the man is currently carrying (`carried_spanner`), if any.
        - The set of all spanners that are currently `usable` (`usable_spanners_state`).
        - The locations of all spanners currently on the ground (`spanner_locations`). Check if these ground spanners are also in `usable_spanners_state`.
    3.  **Prepare Simulation:**
        - Initialize `total_cost` to 0.0.
        - Set the simulation's current man location (`current_man_loc`) to the actual `man_loc`.
        - Create a mutable copy of the set of loose goal nuts identified in step 1 (`sim_loose_goal_nuts`).
        - Determine if the initially carried spanner (if any) is currently usable (`sim_carried_spanner_is_usable`).
        - Create a list of usable spanners currently on the ground (`sim_usable_ground_spanners`), storing them as (spanner_name, location) tuples.
    4.  **Simulate Tightening Sequentially (Greedy Loop):**
        a. Start a loop that continues as long as `sim_loose_goal_nuts` is not empty.
        b. Inside the loop, find the "best" nut to tighten next:
            i.   Initialize `min_cost_for_step` to infinity and `best_nut_to_tighten` to None.
            ii.  Iterate through each nut `n` in `sim_loose_goal_nuts`. For each `n`, calculate the minimum estimated cost to tighten it *from the current simulated state* (`current_man_loc`, `sim_carried_spanner_is_usable`, `sim_usable_ground_spanners`):
                 - *Option A (Use Carried Spanner):* If `sim_carried_spanner_is_usable` is true, calculate the cost: `distance(current_man_loc, nut_loc) + 1` (for walk + tighten action). If the path is impossible (distance is infinity), this option's cost is infinity.
                 - *Option B (Pick Up Ground Spanner):* Iterate through each available usable ground spanner `(s, sloc)` in `sim_usable_ground_spanners`. Calculate the cost: `distance(current_man_loc, sloc) + 1 + distance(sloc, nut_loc) + 1` (for walk-to-spanner + pickup + walk-to-nut + tighten). Find the minimum cost among all valid ground spanners. If no ground spanner can reach the nut location feasibly, this option's cost is infinity.
                 - The minimum cost to tighten nut `n` in this step is `min(cost_A, cost_B)`. Store this cost and the associated plan details (which option was chosen, which spanner if Option B).
            iii. Compare the minimum cost for nut `n` with `min_cost_for_step`. If it's lower, update `min_cost_for_step`, set `best_nut_to_tighten` to `n`, and store the details of the plan (`best_plan_for_step`).
        c. After checking all nuts in `sim_loose_goal_nuts`:
            i.   **Handle Dead Ends:** If `best_nut_to_tighten` is still None (meaning `min_cost_for_step` remained infinity), it implies no remaining nut can be tightened from the current simulated state (e.g., no usable spanners left, or required locations are unreachable). In this case, return a large penalty value (e.g., 10000 + number of remaining nuts * 100) to strongly discourage this state.
            ii.  **Update Total Cost:** Add `min_cost_for_step` to the running `total_cost`.
            iii. **Update Simulation State:**
                 - Set `current_man_loc` to the location of the `best_nut_to_tighten`.
                 - Remove `best_nut_to_tighten` from `sim_loose_goal_nuts`.
                 - If the `best_plan_for_step` used the carried spanner (`type == 'use_carried'`), set `sim_carried_spanner_is_usable = False` for subsequent iterations.
                 - If the plan involved picking up a ground spanner (`type == 'pickup_and_use'`), remove that specific spanner `(s, sloc)` from the `sim_usable_ground_spanners` list. Also set `sim_carried_spanner_is_usable = False` (as the man effectively ends the step without a usable spanner in hand).
    5.  **Return Total Cost:** Once the loop finishes (all goal nuts have been simulated as tightened), return the final `total_cost`, rounded to the nearest integer.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing the task, identifying objects and static info,
        and pre-computing location distances.
        """
        self.man = None
        self.locations = set()
        self.nuts = set()
        self.spanners = set()
        self.nut_locations = {} # nut -> location
        self.goal_nuts = set() # set of nut names in the goal
        self.links = set()
        self.distances = {} # Populated by _compute_distances

        # --- Object and Static Info Extraction ---
        static_facts = task.static
        initial_state = task.initial_state
        goals = task.goals

        # Find locations and links from static facts
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                parts = get_parts(fact)
                self.links.add(fact)
                self.locations.add(parts[1])
                self.locations.add(parts[2])

        # Find objects and their initial states/locations from initial state
        initial_obj_locations = {}
        initial_loose_nuts = set()
        initial_usable_spanners = set()
        initial_carrying = {} # man -> spanner

        for fact in initial_state:
            # Use try-except for robustness against malformed facts
            try:
                parts = get_parts(fact)
                if not parts: continue # Skip empty or malformed facts
                predicate = parts[0]

                if predicate == "at" and len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    initial_obj_locations[obj] = loc
                    self.locations.add(loc) # Ensure all mentioned locations are tracked
                elif predicate == "loose" and len(parts) == 2:
                    initial_loose_nuts.add(parts[1])
                elif predicate == "usable" and len(parts) == 2:
                    initial_usable_spanners.add(parts[1])
                elif predicate == "carrying" and len(parts) == 3:
                     man_name, spanner_name = parts[1], parts[2]
                     initial_carrying[man_name] = spanner_name
                     # Assume the first entity found carrying something is the 'man'
                     if self.man is None:
                         self.man = man_name
                     elif self.man != man_name:
                          # This indicates an unexpected situation based on the assumption of one man
                          print(f"Warning: Multiple agents found carrying? ({self.man}, {man_name}). Sticking with {self.man}.")
            except IndexError:
                print(f"Warning: Skipping malformed fact in initial state: {fact}")


        # Infer object types and populate lists (heuristic approach)
        all_objects_in_at = set(initial_obj_locations.keys())
        self.nuts.update(initial_loose_nuts)
        self.spanners.update(initial_usable_spanners)
        if self.man in initial_carrying:
             self.spanners.add(initial_carrying[self.man]) # Add carried spanner to spanner list

        # Try to identify the man if not found via carrying predicate
        if self.man is None:
             # Look for objects 'at' a location that aren't nuts or spanners already identified
             potential_men = all_objects_in_at - self.nuts - self.spanners
             if len(potential_men) == 1:
                 self.man = potential_men.pop()
             elif len(potential_men) > 1:
                  # Fallback: Check common names or pick first
                  if 'bob' in potential_men: self.man = 'bob' # Common name in examples
                  else: self.man = list(potential_men)[0]
                  print(f"Warning: Multiple potential men found ({potential_men}), heuristically choosing {self.man}")

        # Final check for man identification
        if self.man is None:
             # If still no man found, maybe there are no 'carrying' facts and only one object?
             inferred_men = all_objects_in_at - self.nuts - self.spanners
             if len(inferred_men) == 1:
                 self.man = inferred_men.pop()
             elif inferred_men:
                 # Last resort: pick the first object found 'at' that isn't a known nut/spanner
                 self.man = list(inferred_men)[0]
                 print(f"Warning: Man identification uncertain. Choosing {self.man} as potential man.")
             else:
                # Check if man is mentioned only in goals or operators (less likely for init state)
                # If absolutely no candidate, raise error.
                raise ValueError("Could not identify the man object in the task.")

        # Refine nut/spanner lists based on all objects found and naming conventions
        # This helps catch nuts/spanners that might not be initially loose/usable but exist
        for obj in all_objects_in_at:
             if obj != self.man: # Exclude the identified man
                 # Use startswith as a heuristic if types aren't explicit
                 if obj.startswith('nut') and obj not in self.nuts: self.nuts.add(obj)
                 if obj.startswith('spanner') and obj not in self.spanners: self.spanners.add(obj)

        # Get goal nuts and their locations
        for goal in goals:
             try:
                if match(goal, "tightened", "*"):
                    parts = get_parts(goal)
                    if len(parts) == 2:
                        nut = parts[1]
                        self.goal_nuts.add(nut)
                        self.nuts.add(nut) # Ensure goal nuts are in the set of all nuts
                        # Find the location of this nut from the initial state 'at' facts
                        if nut in initial_obj_locations:
                            self.nut_locations[nut] = initial_obj_locations[nut]
                        else:
                            # This nut must exist and have a location in the initial state
                            raise ValueError(f"Location for goal nut '{nut}' not found in initial state facts.")
             except IndexError:
                 print(f"Warning: Skipping malformed goal fact: {goal}")


        # Ensure all identified nuts have known locations
        for nut in self.nuts:
             if nut not in self.nut_locations:
                 if nut in initial_obj_locations:
                     self.nut_locations[nut] = initial_obj_locations[nut]
                 else:
                      # This indicates a problem, maybe the nut isn't mentioned in 'at' initially?
                      raise ValueError(f"Location for nut '{nut}' is required but could not be determined from initial state.")


        # --- Precompute Distances ---
        # Convert locations set to list for consistent iteration order if needed, though order doesn't matter for BFS itself
        self.distances = self._compute_distances(list(self.locations), self.links)


    def _compute_distances(self, locations, links):
        """Computes shortest path distances between all pairs of locations using BFS."""
        if not locations:
            return {}

        # Initialize distances: dict mapping start_loc -> {end_loc: distance}
        distances = {loc: {other: math.inf for other in locations} for loc in locations}
        # Build adjacency list representation of the location graph
        adj = {loc: [] for loc in locations}
        for link_fact in links:
             try:
                parts = get_parts(link_fact)
                if len(parts) == 3 and parts[0] == 'link':
                    u, v = parts[1], parts[2]
                    # Ensure locations from link exist in our set before adding edges
                    if u in locations and v in locations:
                        adj[u].append(v)
                        adj[v].append(u) # Assume links are bidirectional
             except IndexError:
                 print(f"Warning: Skipping malformed link fact: {link_fact}")


        # Run BFS from each location
        for start_node in locations:
            # Skip if start_node somehow isn't in the keys (shouldn't happen)
            if start_node not in distances: continue
            distances[start_node][start_node] = 0
            queue = deque([start_node])
            # Keep track of visited nodes and their distances in this specific BFS run
            visited_dist_in_run = {start_node: 0}

            while queue:
                current_node = queue.popleft()
                current_dist = visited_dist_in_run[current_node]

                # Explore neighbors using the adjacency list
                if current_node in adj: # Check if node has neighbors defined
                    for neighbor in adj[current_node]:
                        # Ensure neighbor is a valid location and not visited yet in this run
                        if neighbor in locations and neighbor not in visited_dist_in_run:
                            visited_dist_in_run[neighbor] = current_dist + 1
                            # Update the main distances dictionary
                            distances[start_node][neighbor] = current_dist + 1
                            queue.append(neighbor)
        return distances

    def get_dist(self, loc1, loc2):
        """Safely retrieves the precomputed distance between two locations."""
        if loc1 is None or loc2 is None:
            # If either location is unknown or invalid, path is impossible
            return math.inf
        # Access nested dictionary safely, returning infinity if keys don't exist
        return self.distances.get(loc1, {}).get(loc2, math.inf)


    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.
        """
        state = node.state

        # 1. Identify unmet goals
        current_tightened = set()
        for fact in state:
            try:
                if match(fact, "tightened", "*"):
                    parts = get_parts(fact)
                    if len(parts) == 2:
                        current_tightened.add(parts[1])
            except IndexError: pass # Ignore malformed facts

        loose_goal_nuts = self.goal_nuts - current_tightened

        if not loose_goal_nuts:
            return 0 # Goal reached

        # 2. Snapshot Current State
        man_loc = None
        carried_spanner = None
        usable_spanners_state = set() # Names of all usable spanners
        spanner_locations = {} # Location of spanners on the ground

        for fact in state:
            try:
                parts = get_parts(fact)
                if not parts: continue
                predicate = parts[0]

                if predicate == "at" and len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    if obj == self.man:
                        man_loc = loc
                    # Check if obj is a known spanner before storing location
                    elif obj in self.spanners:
                        spanner_locations[obj] = loc
                elif predicate == "carrying" and len(parts) == 3:
                    # Assuming structure (carrying man spanner)
                    if parts[1] == self.man:
                        carried_spanner = parts[2]
                elif predicate == "usable" and len(parts) == 2:
                    usable_spanners_state.add(parts[1])
            except IndexError: pass # Ignore malformed facts

        # If man's location is missing, heuristic cannot be computed reliably
        if man_loc is None:
             print(f"Warning: Man '{self.man}' location not found in state. Returning infinity.")
             # Returning infinity might halt search; consider a large finite number if preferred
             return math.inf

        # 3. Prepare Simulation state
        total_cost = 0.0 # Use float for potentially large sums before rounding
        current_man_loc = man_loc
        # Use a copy for simulation as we will modify it
        sim_loose_goal_nuts = set(loose_goal_nuts)
        # Check usability of the spanner currently carried
        sim_carried_spanner_is_usable = (carried_spanner is not None) and (carried_spanner in usable_spanners_state)

        # Create list of (spanner_name, location) for usable spanners currently on the ground
        sim_usable_ground_spanners = []
        for spanner, loc in spanner_locations.items():
            # Include only if it's usable and not the one being carried
            if spanner in usable_spanners_state and spanner != carried_spanner:
                sim_usable_ground_spanners.append((spanner, loc))

        # 4. Simulate Tightening Sequentially (Greedy Loop)
        while sim_loose_goal_nuts:
            best_nut_to_tighten = None
            min_cost_for_step = math.inf
            # Store details of the best plan found in this step
            best_plan_for_step = {}

            # Iterate through each remaining loose goal nut to find the cheapest one to tighten next
            for nut in sim_loose_goal_nuts:
                nut_loc = self.nut_locations.get(nut)
                # If nut location is somehow unknown, skip (should be caught in init)
                if nut_loc is None: continue

                cost_option_A = math.inf
                plan_option_A = {}
                # Option A: Use the spanner currently carried by the man, if it's usable
                if sim_carried_spanner_is_usable:
                    dist_walk = self.get_dist(current_man_loc, nut_loc)
                    # Check if path is possible
                    if dist_walk != math.inf:
                        # Cost = walk distance + 1 action for tighten
                        cost_option_A = dist_walk + 1
                        plan_option_A = {'type': 'use_carried', 'cost': cost_option_A, 'nut': nut, 'nut_loc': nut_loc}

                cost_option_B = math.inf
                plan_option_B = {}
                # Option B: Pick up a usable spanner from the ground and use it
                for spanner, spanner_loc in sim_usable_ground_spanners:
                    # Calculate distance to walk to the spanner
                    dist_to_spanner = self.get_dist(current_man_loc, spanner_loc)
                    # Calculate distance to walk from the spanner to the nut
                    dist_spanner_to_nut = self.get_dist(spanner_loc, nut_loc)

                    # Check if both paths are possible
                    if dist_to_spanner != math.inf and dist_spanner_to_nut != math.inf:
                        # Cost = walk-to-spanner + pickup + walk-to-nut + tighten
                        current_pickup_cost = dist_to_spanner + 1 + dist_spanner_to_nut + 1
                        # If this spanner offers a cheaper way than previously found for Option B
                        if current_pickup_cost < cost_option_B:
                            cost_option_B = current_pickup_cost
                            plan_option_B = {'type': 'pickup_and_use', 'cost': cost_option_B, 'nut': nut, 'nut_loc': nut_loc, 'spanner': spanner, 'spanner_loc': spanner_loc}

                # Determine the minimum cost to tighten *this* nut (comparing Option A and B)
                current_nut_min_cost = math.inf
                current_nut_plan = {}
                if cost_option_A < cost_option_B:
                    current_nut_min_cost = cost_option_A
                    current_nut_plan = plan_option_A
                elif cost_option_B != math.inf: # Check cost_option_B is valid before assigning
                    current_nut_min_cost = cost_option_B
                    current_nut_plan = plan_option_B

                # Check if tightening this nut is the best option found *so far in this step*
                if current_nut_min_cost < min_cost_for_step:
                    min_cost_for_step = current_nut_min_cost
                    best_nut_to_tighten = nut
                    best_plan_for_step = current_nut_plan

            # --- After checking all nuts for the current step ---

            # 4.c.i Handle Dead Ends: If no nut could be planned
            if best_nut_to_tighten is None:
                # This implies no remaining nut can be tightened from the current simulated state.
                # Return a large penalty value.
                # The value should be large enough to be non-competitive but finite if possible.
                # Scale penalty by remaining work.
                return 10000 + len(sim_loose_goal_nuts) * 100

            # 4.c.ii Update Total Cost: Add the cost of the chosen action for this step
            total_cost += best_plan_for_step['cost']

            # 4.c.iii Update Simulation State for the next iteration
            # Man moves to the location of the nut that was just tightened
            current_man_loc = best_plan_for_step['nut_loc']
            # Remove the tightened nut from the set of remaining nuts
            sim_loose_goal_nuts.remove(best_nut_to_tighten)

            # Update spanner availability based on the plan executed
            if best_plan_for_step['type'] == 'use_carried':
                # The carried spanner was used, so it's no longer usable in the simulation
                sim_carried_spanner_is_usable = False
            elif best_plan_for_step['type'] == 'pickup_and_use':
                # A ground spanner was picked up and used. Remove it from the available list.
                spanner_used_tuple = (best_plan_for_step['spanner'], best_plan_for_step['spanner_loc'])
                # Create a new list excluding the used spanner (safer than remove in place)
                sim_usable_ground_spanners = [s for s in sim_usable_ground_spanners if s != spanner_used_tuple]
                # After picking up and using, the man is effectively not carrying a usable spanner
                # for the start of the *next* simulated step.
                sim_carried_spanner_is_usable = False

        # 5. Return Total Cost: After the loop finishes, return the accumulated cost.
        # Round to integer, as action costs are integers.
        return int(round(total_cost))

