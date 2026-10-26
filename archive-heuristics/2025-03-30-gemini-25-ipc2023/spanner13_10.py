import itertools
from fnmatch import fnmatch
from collections import deque
import math # For infinity

# Try to import the base class from the expected location
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the import fails (e.g., for standalone testing)
    class Heuristic:
        """Dummy base class for Heuristic."""
        def __init__(self, task):
            """Initialize the heuristic."""
            pass
        def __call__(self, node):
            """Calculate the heuristic value for a given node."""
            raise NotImplementedError("Heuristic base class not found.")

# Helper function to safely parse PDDL fact strings
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Handles basic validation.
    Returns a list of strings (predicate name and arguments) or an empty list if malformed.
    """
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Return empty list for malformed facts instead of raising an error
        return []
    return fact[1:-1].split()

# Helper function to match fact parts against a pattern
def parts_match(fact_parts, *pattern):
    """
    Checks if the parts of a fact (already split into a list) match a given pattern.
    Uses fnmatch to allow '*' wildcards in the pattern.
    Returns True if the parts match the pattern, False otherwise.
    """
    if len(fact_parts) != len(pattern):
        return False
    return all(fnmatch(part, pat) for part, pat in zip(fact_parts, pattern))

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Spanner domain.

    # Summary
    Estimates the remaining cost (number of actions) to reach a goal state
    where all specified nuts are tightened. The heuristic calculates the cost based on:
    1. The number of 'tighten_nut' actions required for remaining loose nuts.
    2. The number of 'pickup_spanner' actions needed if the man isn't carrying enough usable spanners.
    3. An estimated travel cost ('walk' actions) for the man to visit nut locations and potentially
       the location of the nearest required spanner.
    This heuristic is designed for Greedy Best-First Search and is likely non-admissible.

    # Assumptions
    - There is exactly one 'man' object instance in the planning problem.
    - The 'tighten_nut' action requires a 'usable' spanner and makes that spanner unusable.
    - The location graph, defined by 'link' predicates, might be disconnected. The heuristic
      returns infinity if required locations are unreachable.
    - Object names and predicate structures are consistent with the provided domain definition.
    - The `task` object passed during initialization correctly provides `initial_state`,
      `goals`, and `static` facts (including 'link' predicates).

    # Heuristic Initialization
    - Identifies the single 'man' object, all 'nut' objects, and all 'spanner' objects
      by analyzing predicates in the initial state (e.g., 'carrying', 'loose', 'usable', 'at').
      This relies on predicate patterns typical for the domain.
    - Parses static 'link' facts to build an adjacency list representation of the locations graph.
    - Computes All-Pairs Shortest Paths (APSP) using Breadth-First Search (BFS) starting
      from each known location. Distances are stored in a dictionary (`self.distances`).
      Unreachable locations will have a distance of infinity.
    - Determines the set of 'nut' objects that must be 'tightened' according to the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Extract current facts: man's location (`lm`), locations of all nuts (`ln`)
        and spanners (`ls`), the set of currently 'loose' nuts (`N_loose`), the set of currently
        'usable' spanners (`S_usable`), and the set of spanners currently carried by the man (`S_carried`).
    2.  **Identify Remaining Goals:** Determine the set of nuts (`N_rem`) that are goals (`self.goal_nuts`)
        and are currently loose (`N_loose`). Let `k = |N_rem|`.
    3.  **Check Goal Achievement:** If `k == 0`, all nut-tightening goals are met. Return heuristic value 0.
    4.  **Check Feasibility:** Count the total number of usable spanners (`|S_usable|`). If `k > |S_usable|`,
        there are not enough usable spanners to tighten all remaining nuts. The goal is unreachable. Return infinity.
    5.  **Calculate Base Action Costs:** Initialize heuristic value `h = 0`.
        - Add `k` to `h`. This accounts for the `k` necessary 'tighten_nut' actions.
        - Determine the set of usable spanners currently carried (`S_carried_usable = S_carried ∩ S_usable`).
        - Calculate the minimum number of spanners the man must pick up:
          `pickups_needed = max(0, k - |S_carried_usable|)`.
        - Add `pickups_needed` to `h`. This accounts for the necessary 'pickup_spanner' actions.
    6.  **Estimate Movement Costs:** Initialize `move_cost = 0`.
        - Get the man's current location `lm`. If `lm` is not found in the state, return infinity (invalid state).
        - **Nut Visits:** For each nut `n` in `N_rem`:
            - Find its location `ln`. If not found, return infinity (invalid state).
            - Look up the shortest distance `d = distance(lm, ln)` from the precomputed `self.distances`.
            - If `d` is infinity, the nut is unreachable. Return infinity.
            - Add `d` to `move_cost`. This sums the distances from the man's current spot to all required nut locations.
        - **Spanner Pickup Trip:** If `pickups_needed > 0`:
            - Find all usable spanners `s` that are *not* in `S_carried_usable` and get their locations `ls`.
            - If no such spanners have a location `ls` in the current state, return infinity (inconsistent state).
            - Calculate the minimum distance from the man's location `lm` to any reachable `ls`:
              `min_dist_to_pickup = min(distance(lm, ls))` over all such located, usable, uncarried spanners `s`.
            - If `min_dist_to_pickup` is infinity (no needed spanner is reachable), return infinity.
            - Add `min_dist_to_pickup` to `move_cost`. This estimates the travel cost for the *first* required pickup action.
    7.  **Combine Costs:** Add the total estimated `move_cost` to the heuristic value `h`.
    8.  **Return Value:** Return the final calculated heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information from the task.
        - Identifies objects (man, nuts, spanners).
        - Builds the location graph and computes all-pairs shortest paths.
        - Stores goal nuts.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state_facts = task.initial_state

        # --- Object Identification ---
        self.man = None
        self.all_nuts = set()
        self.all_spanners = set()
        # Use a temporary dictionary to guess object types based on predicates
        temp_locatables = {} # obj_name -> type_guess ('man', 'nut', 'spanner')

        # Pass 1: Infer types from predicates with strong type implications in init state
        for fact in initial_state_facts:
            parts = get_parts(fact)
            if not parts: continue
            # Check 'carrying(man, spanner)'
            if parts_match(parts, 'carrying', '*', '*'):
                temp_locatables[parts[1]] = 'man'
                temp_locatables[parts[2]] = 'spanner'
            # Check 'loose(nut)'
            elif parts_match(parts, 'loose', '*'):
                temp_locatables[parts[1]] = 'nut'
            # Check 'usable(spanner)'
            elif parts_match(parts, 'usable', '*'):
                 # Only type as spanner if not already typed differently (e.g., as man)
                 if parts[1] not in temp_locatables or temp_locatables[parts[1]] == 'spanner':
                     temp_locatables[parts[1]] = 'spanner'

        # Store locations and objects found at locations from 'at' predicates
        locations_found_in_at = set()
        objects_at_loc = set()
        for fact in initial_state_facts:
             parts = get_parts(fact)
             if not parts: continue
             if parts_match(parts, 'at', '*', '*'):
                 obj, loc = parts[1], parts[2]
                 objects_at_loc.add(obj)
                 locations_found_in_at.add(loc)

        # Assign identified objects to the final sets
        for obj, type_guess in temp_locatables.items():
            if type_guess == 'man': self.man = obj
            elif type_guess == 'nut': self.all_nuts.add(obj)
            elif type_guess == 'spanner': self.all_spanners.add(obj)

        # If man wasn't identified via 'carrying', try finding an object 'at' a location
        # that hasn't been identified as a nut or spanner.
        if self.man is None:
            for obj in objects_at_loc:
                 if obj not in self.all_nuts and obj not in self.all_spanners:
                     # Assume the first such untyped object found at a location is the man
                     self.man = obj
                     break

        # If man is still None after checking init state, raise an error.
        if self.man is None:
             raise ValueError("SpannerHeuristic: Could not identify the 'man' object from the initial state.")

        # Identify any remaining spanners: objects 'at' locations not identified as man or nut.
        for obj in objects_at_loc:
             if obj != self.man and obj not in self.all_nuts and obj not in self.all_spanners:
                 self.all_spanners.add(obj)

        # Ensure all nuts mentioned in goals are included in self.all_nuts
        for goal in self.goals:
             parts = get_parts(goal)
             if not parts: continue
             if parts_match(parts, 'tightened', '*'):
                 self.all_nuts.add(parts[1]) # Add nut from goal if not already known

        # --- Graph and Distances ---
        self.locations = set()
        adj = {} # Adjacency list for BFS
        # Build graph from 'link' predicates
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            if parts_match(parts, 'link', '*', '*'):
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                adj.setdefault(l1, []).append(l2)
                adj.setdefault(l2, []).append(l1)

        # Include locations mentioned in 'at' facts, even if not linked
        self.locations.update(locations_found_in_at)

        # Compute All-Pairs Shortest Paths using BFS from each location
        self.distances = {}
        for start_node in self.locations:
            # Initialize distances from start_node to all locations as infinity
            self.distances[start_node] = {loc: float('inf') for loc in self.locations}
            # Distance from a node to itself is 0
            if start_node in self.locations: # Check if start_node is valid
                 self.distances[start_node][start_node] = 0
            else:
                 continue # Skip if start_node isn't a known location

            # BFS queue
            queue = deque([start_node])
            # Set of visited nodes for this specific BFS run
            visited = {start_node}

            while queue:
                current_loc = queue.popleft()
                current_dist = self.distances[start_node][current_loc]

                # Explore neighbors
                for neighbor in adj.get(current_loc, []):
                    if neighbor in self.locations and neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[start_node][neighbor] = current_dist + 1
                        queue.append(neighbor)

        # Store the set of goal nuts for quick lookup during heuristic calculation
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            if parts_match(parts, 'tightened', '*'):
                 nut = parts[1]
                 if nut in self.all_nuts: # Ensure it's a known nut object
                     self.goal_nuts.add(nut)


    def __call__(self, node):
        """
        Calculates the heuristic estimate for the given state node.
        Returns an integer cost estimate or float('inf') if the goal is determined unreachable.
        """
        state = node.state

        # --- State Parsing ---
        man_loc = None
        nut_locs = {} # Map: nut_name -> location
        spanner_locs = {} # Map: spanner_name -> location
        current_loose_nuts = set()
        current_usable_spanners = set()
        current_carried_spanners = set() # Set of spanner names carried by the man

        # Parse the current state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip potential malformed facts

            # Check 'at(obj, loc)'
            if parts_match(parts, 'at', '*', '*'):
                obj, loc = parts[1], parts[2]
                if obj == self.man: man_loc = loc
                elif obj in self.all_nuts: nut_locs[obj] = loc
                elif obj in self.all_spanners: spanner_locs[obj] = loc
            # Check 'loose(nut)'
            elif parts_match(parts, 'loose', '*'):
                nut = parts[1]
                if nut in self.all_nuts: current_loose_nuts.add(nut)
            # Check 'usable(spanner)'
            elif parts_match(parts, 'usable', '*'):
                spanner = parts[1]
                if spanner in self.all_spanners: current_usable_spanners.add(spanner)
            # Check 'carrying(man, spanner)' - ensure it's the identified man
            elif parts_match(parts, 'carrying', self.man, '*'):
                spanner = parts[2]
                if spanner in self.all_spanners: current_carried_spanners.add(spanner)

        # --- Heuristic Calculation ---

        # 1. Identify remaining goal nuts
        remaining_nuts = self.goal_nuts.intersection(current_loose_nuts)
        num_remaining = len(remaining_nuts)

        # 2. Check if goal is already achieved
        if num_remaining == 0:
            return 0

        # 3. Check feasibility: Are there enough usable spanners?
        num_usable = len(current_usable_spanners)
        if num_remaining > num_usable:
            return float('inf') # Goal unreachable

        # 4. Calculate base action costs (tighten + pickup)
        h = num_remaining # Cost for 'tighten_nut' actions

        # Find usable spanners currently carried
        carried_usable = current_carried_spanners.intersection(current_usable_spanners)
        num_carried_usable = len(carried_usable)
        # Calculate necessary pickups
        pickups_needed = max(0, num_remaining - num_carried_usable)
        h += pickups_needed # Cost for 'pickup_spanner' actions

        # 5. Estimate movement costs

        # Man's location is essential
        if man_loc is None:
            # This indicates an invalid state where the man has no location.
            return float('inf')

        move_cost = 0
        # Cost to visit each remaining nut location from current man location
        for nut in remaining_nuts:
            n_loc = nut_locs.get(nut)
            # Check if the loose nut has a location in the current state
            if n_loc is None:
                 # Invalid state: a required loose nut is not 'at' anywhere.
                 return float('inf')
            # Look up distance, check reachability
            dist = self.distances.get(man_loc, {}).get(n_loc, float('inf'))
            if dist == float('inf'):
                # Nut location is unreachable from man's current location.
                return float('inf') # Goal unreachable
            move_cost += dist

        # Cost for the first pickup trip, if pickups are needed
        if pickups_needed > 0:
            min_dist_to_pickup = float('inf')
            found_reachable_spanner = False
            # Iterate through all usable spanners
            for spanner in current_usable_spanners:
                # Consider only those not already carried
                if spanner not in carried_usable:
                    # Find location of this spanner
                    spanner_loc = spanner_locs.get(spanner)
                    # Check if the spanner is actually located somewhere
                    if spanner_loc:
                        # Check reachability from man's location
                        dist = self.distances.get(man_loc, {}).get(spanner_loc, float('inf'))
                        if dist != float('inf'):
                            # Found a reachable spanner, update minimum distance
                            min_dist_to_pickup = min(min_dist_to_pickup, dist)
                            found_reachable_spanner = True
                    # else: A usable spanner exists, isn't carried, but has no 'at' fact. Invalid state?

            # If pickups are needed but no suitable spanner is reachable
            if not found_reachable_spanner:
                 # This implies an inconsistency or unreachable state configuration.
                 return float('inf')

            # Add the cost to reach the nearest required spanner
            move_cost += min_dist_to_pickup

        # 6. Combine costs
        h += move_cost

        # Return the final heuristic value (should be non-negative)
        return h
