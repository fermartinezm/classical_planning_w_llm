import itertools
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque # For BFS calculation of distances

# Helper function to parse PDDL facts represented as strings
def get_parts(fact):
    """Extracts the components of a PDDL fact string (e.g., "(pred obj1 obj2)")."""
    return fact[1:-1].split()

class SpannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the PDDL Spanner domain.

    # Summary
    This heuristic estimates the total number of actions required to tighten all
    loose nuts specified in the goal conditions. It calculates the cost for each
    unsatisfied 'tightened' goal individually and sums these costs. The cost
    for tightening a single nut includes the actions needed for the man to potentially
    travel to a usable spanner, pick it up, travel to the nut's location, and
    finally tighten the nut. Since each tighten action consumes a usable spanner,
    the heuristic implicitly assumes a unique usable spanner is needed for each
    nut, reflecting the domain's constraints.

    # Assumptions
    - There is exactly one 'man' agent in the problem instance.
    - The locations of nuts are static (they do not move).
    - Each 'tighten_nut' action requires one 'usable' spanner and makes that spanner unusable afterwards.
    - Movement ('walk' action) cost is uniform (1 per step). The heuristic uses the shortest path distance between locations.
    - The heuristic calculates the cost for each goal nut independently and sums them. It does not attempt complex optimizations like finding a single tour for the man to visit all required locations or picking up items opportunistically. This makes it efficiently computable but potentially less accurate (and non-admissible).
    - Goal conditions only consist of `(tightened ?nut)` predicates.

    # Heuristic Initialization
    - Extracts the goal conditions from the task.
    - Parses static `link` predicates to build an undirected graph representing location connectivity.
    - Computes all-pairs shortest paths (APSP) using Breadth-First Search (BFS) starting from each location. Distances are stored.
    - Identifies the single 'man' object based on initial state facts or common naming conventions (as type information isn't directly available in the provided Task structure).
    - Stores the initial (and assumed static) location of each nut from the initial state facts.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Check Goal Completion:** If the current state satisfies all goal conditions, the heuristic value is 0.
    2.  **Identify Unsatisfied Goals:** Determine which `(tightened ?nut)` goals are not yet true in the current state AND correspond to a nut that is currently `(loose ?nut)`.
    3.  **Gather Current State Information:**
        - Find the man's current location (`man_loc`).
        - Determine if the man is carrying a spanner (`carried_spanner`).
        - Check if the carried spanner (if any) is currently `(usable ?spanner)`.
        - Find all currently usable spanners and their locations (`usable_spanners` map).
    4.  **Calculate Cost Per Unsatisfied Nut:** For each nut `n` identified in step 2 (at location `ln`):
        a.  Initialize `cost_for_nut = 0`.
        b.  **Tighten Action Cost:** Add 1 (for the `tighten_nut` action itself).
        c.  **Spanner Acquisition Cost & Location:**
            i.  **If man carries a usable spanner:** The cost is 0. The effective location where the man has the spanner (`eff_sloc`) is the man's current location (`man_loc`).
            ii. **If man does NOT carry a usable spanner:**
                - Find the usable spanner `s_best` at location `ls_best` that is closest to the man's current location (`man_loc`). Closeness is measured by the precomputed shortest path distance.
                - If no usable spanners exist anywhere, the goal is unreachable from this state (return infinity).
                - If no *reachable* usable spanner exists (distance is infinity), return infinity.
                - Add the distance `dist(man_loc, ls_best)` (cost to walk to the spanner) + 1 (cost for `pickup_spanner`) to `cost_for_nut`.
                - The effective location where the man acquires the spanner (`eff_sloc`) is `ls_best`.
        d.  **Travel to Nut Cost:** Add the distance `dist(eff_sloc, ln)` (cost to walk from where the spanner was acquired to the nut's location) to `cost_for_nut`. If this distance is infinity, the nut is unreachable (return infinity).
        e.  Add `cost_for_nut` to the `total_heuristic_value`.
    5.  **Return Total Value:** The sum of costs calculated for all unsatisfied nuts is the final heuristic estimate. Return infinity if any step indicated impossibility (no usable spanners, unreachable locations).
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information and goals.
        """
        self.goals = task.goals
        static_facts = task.static

        # --- Identify the man ---
        # Assumes one man. Tries to find based on 'at' or 'carrying' in init state.
        # Falls back to assuming 'bob' if needed (based on examples).
        self.man = None
        potential_men = set()
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                 # Simple check assuming man objects might be identifiable
                 # A proper solution would use type information if available
                 potential_men.add(parts[1])
             elif parts[0] == 'carrying':
                 potential_men.add(parts[1])
                 break # Found a carrying man, likely the one

        # Refine potential men - check if they appear in actions involving 'man' type
        # This is approximate without explicit types. Let's assume 'bob' or the first one found.
        if 'bob' in potential_men: # Prioritize 'bob' based on examples
            self.man = 'bob'
        elif potential_men:
            self.man = list(potential_men)[0] # Pick one if 'bob' not found
        else:
            # Last resort if no clues in initial state
            # print("Warning: Could not reliably determine the man's name. Assuming 'bob'.")
            self.man = 'bob'


        # --- Build Location Graph & Compute APSP ---
        locations = set()
        adj = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                loc1, loc2 = parts[1], parts[2]
                # Ensure locations mentioned in links are added
                locations.add(loc1)
                locations.add(loc2)
                adj.setdefault(loc1, set()).add(loc2)
                adj.setdefault(loc2, set()).add(loc1)

        # Also add locations from initial state 'at' facts, in case some are isolated
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                locations.add(parts[2])

        self.locations = list(locations)
        if not self.locations:
             # Handle case with no locations defined
             self.distances = {}
        else:
             self.distances = self._compute_apsp(adj, self.locations)

        # --- Store Nut Locations (assume static) ---
        self.nut_locations = {}
        # Find initial positions of all nuts
        for fact in task.initial_state:
             parts = get_parts(fact)
             # Basic check for nut type based on name convention from examples
             # A robust implementation would use type information.
             if parts[0] == 'at' and 'nut' in parts[1]:
                 nut, loc = parts[1], parts[2]
                 self.nut_locations[nut] = loc
                 if loc not in self.locations:
                     # Ensure nut locations are known, even if isolated
                     self.locations.append(loc)
                     # Recompute distances if new locations added? Or handle in get_path_cost
                     # For simplicity, assume link graph covers all relevant locations or handle disconnectedness later.


    def _compute_apsp(self, graph, locations):
        """Computes all-pairs shortest paths using BFS from each location."""
        distances = {}
        if not locations: return {} # No locations to compute paths for

        for start_node in locations:
            # Initialize distances from start_node
            dist_from_start = {loc: float('inf') for loc in locations}
            # Check if start_node is actually in the graph connectivity
            if start_node in graph or start_node in locations: # Check if it's a known location
                 dist_from_start[start_node] = 0
                 queue = deque([start_node])
                 visited = {start_node}

                 while queue:
                     current_node = queue.popleft()
                     current_dist = dist_from_start[current_node]

                     # Explore neighbors using the graph adjacency list
                     for neighbor in graph.get(current_node, set()):
                         if neighbor not in visited:
                             visited.add(neighbor)
                             dist_from_start[neighbor] = current_dist + 1
                             queue.append(neighbor)

            # Store computed distances from start_node
            for end_node, dist in dist_from_start.items():
                distances[(start_node, end_node)] = dist

        return distances

    def _get_path_cost(self, loc1, loc2):
        """
        Safely retrieves precomputed shortest path cost between two locations.
        Returns float('inf') if locations are identical, unknown, or disconnected.
        """
        if loc1 == loc2:
             return 0 # No cost to travel between the same location
        # Check if locations are known (present in the initial list derived from links/init state)
        # Note: self.locations might not be exhaustive if some objects start outside linked areas
        # The self.distances dictionary lookup handles unknown/disconnected cases implicitly.
        cost = self.distances.get((loc1, loc2), float('inf'))
        return cost


    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        """
        state = node.state

        # --- Check if goal is already reached ---
        if self.goals <= state:
            return 0

        # --- Get current state information ---
        man_loc = None
        carried_spanner = None
        is_carried_spanner_usable = False
        usable_spanners_at = {} # {spanner_name: location}
        loose_nuts_in_state = set()
        current_usable_spanners = set() # Set of names of usable spanners

        # First pass: find man location, carried item, loose nuts, and usable spanner names
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == 'at':
                obj, loc = parts[1], parts[2]
                if obj == self.man:
                    man_loc = loc
            elif pred == 'carrying':
                m, s = parts[1], parts[2]
                if m == self.man:
                    carried_spanner = s
            elif pred == 'loose':
                loose_nuts_in_state.add(parts[1])
            elif pred == 'usable':
                current_usable_spanners.add(parts[1])

        # If man's location wasn't found (invalid state?)
        if man_loc is None:
            # print(f"Warning: Man '{self.man}' location not found in state {state}.")
            return float('inf') # Cannot proceed without man's location

        # Check usability of the carried spanner
        if carried_spanner and carried_spanner in current_usable_spanners:
            is_carried_spanner_usable = True

        # Second pass: find locations of usable spanners (that are 'at' somewhere)
        for fact in state:
             parts = get_parts(fact)
             pred = parts[0]
             if pred == 'at':
                 obj, loc = parts[1], parts[2]
                 # Check if this object is a known usable spanner
                 if obj in current_usable_spanners:
                     usable_spanners_at[obj] = loc


        # --- Calculate heuristic value ---
        total_heuristic_value = 0
        unsatisfied_goal_nuts = []

        # Identify which goal nuts are still loose and need tightening
        for goal_fact in self.goals:
            # Assume goals are of the form (tightened ?nut)
            if goal_fact not in state:
                parts = get_parts(goal_fact)
                if parts[0] == 'tightened':
                    nut = parts[1]
                    # Check if this nut is currently loose
                    loose_fact = f"(loose {nut})"
                    if loose_fact in state:
                         unsatisfied_goal_nuts.append(nut)

        # If no loose nuts need tightening towards the goal state
        if not unsatisfied_goal_nuts:
             # This can happen if goals are met, or if remaining goals are not 'tightened'
             # or if the nuts for remaining goals are not currently loose.
             # If self.goals <= state was false, it implies other goals exist or state is inconsistent.
             # Heuristic returns 0 assuming only 'tightened' goals matter for cost estimation.
             return 0

        # --- Calculate cost for each unsatisfied loose nut ---
        for nut in unsatisfied_goal_nuts:
            nut_loc = self.nut_locations.get(nut)
            if nut_loc is None:
                # Should not happen if init parsing was correct
                # print(f"Warning: Location for goal nut '{nut}' not found.")
                return float('inf')

            cost_for_nut = 0
            eff_sloc = None # Effective location where man has/gets the spanner

            # 1. Spanner Acquisition Cost
            if is_carried_spanner_usable:
                walk_to_spanner_cost = 0
                pickup_cost = 0
                eff_sloc = man_loc
            else:
                # Man needs to get a usable spanner from the environment
                if not usable_spanners_at:
                    # No usable spanners available anywhere
                    return float('inf') # Dead end state

                # Find the closest usable spanner
                min_dist = float('inf')
                best_spanner_loc = None
                for spanner, loc in usable_spanners_at.items():
                    dist = self._get_path_cost(man_loc, loc)
                    if dist < min_dist:
                        min_dist = dist
                        best_spanner_loc = loc

                if best_spanner_loc is None or min_dist == float('inf'):
                     # No usable spanner is reachable from the man's location
                     return float('inf') # Dead end state

                walk_to_spanner_cost = min_dist
                pickup_cost = 1 # Cost of pickup_spanner action
                eff_sloc = best_spanner_loc

            # 2. Travel to Nut Cost (from effective spanner location)
            walk_to_nut_cost = self._get_path_cost(eff_sloc, nut_loc)
            if walk_to_nut_cost == float('inf'):
                 # Nut location is unreachable from where spanner is acquired
                 return float('inf') # Dead end state

            # 3. Tighten Action Cost
            tighten_cost = 1 # Cost of tighten_nut action

            # Sum costs for this nut
            cost_for_nut = walk_to_spanner_cost + pickup_cost + walk_to_nut_cost + tighten_cost
            total_heuristic_value += cost_for_nut

        # Ensure heuristic is non-negative
        # The calculation naturally produces non-negative costs or infinity.
        # If total_heuristic_value is 0 here, it means unsatisfied_goal_nuts was empty,
        # which should have been caught earlier.
        return total_heuristic_value
