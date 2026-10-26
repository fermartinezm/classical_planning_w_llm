import itertools
from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import heapq # Potentially useful for optimizations, but not strictly needed for the current logic

# Helper function to parse PDDL facts represented as strings
def get_parts(fact_string):
    """
    Extracts the predicate and arguments from a PDDL fact string.
    Example: "(at bob shed)" -> ["at", "bob", "shed"]
    """
    return fact_string[1:-1].split()

# Helper function to match parsed fact parts against a pattern
def match(fact_parts, *pattern):
    """
    Checks if the parts of a parsed fact match a given pattern.
    Allows '*' as a wildcard in the pattern.
    Example: match(["at", "bob", "shed"], "at", "*", "shed") -> True
    """
    return len(fact_parts) == len(pattern) and all(fnmatch(part, pat) for part, pat in zip(fact_parts, pattern))

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'spanner'.

    # Summary
    This heuristic estimates the total number of actions required to reach a goal state
    from the current state. The goal is typically to have specific nuts tightened.
    The heuristic calculates the cost by summing the estimated minimum actions needed
    to tighten each *remaining* goal nut individually. The cost for tightening a
    single nut includes the necessary 'walk' actions for the man, potentially one
    'pickup_spanner' action, and one 'tighten_nut' action.

    # Assumptions
    - There is exactly one 'man' object in the planning problem instance.
    - The locations of nuts are static (they do not change during the plan).
    - The 'link' predicates defining connectivity between locations are static.
    - The heuristic estimates the cost for each required nut tightening independently
      and sums these costs. This ignores the fact that tightening one nut consumes
      the usability of a spanner, potentially affecting the cost calculation for
      subsequent nuts. This makes the heuristic non-admissible but potentially
      informative for greedy search.
    - The heuristic assumes that if a solution exists, there will be a way to
      get a usable spanner to the required nut location.

    # Heuristic Initialization
    - Identifies the set of nuts that must be tightened to satisfy the goal conditions
      (e.g., finds all 'nut1' for which '(tightened nut1)' is a goal).
    - Parses the static 'link' facts to build an undirected graph representing
      location connectivity.
    - Computes all-pairs shortest path distances between all locations using Breadth-First Search (BFS)
      starting from each location. Stores these distances for efficient lookup.
    - Identifies the unique 'man' object within the problem instance.
    - Determines and stores the initial (and assumed static) location of every nut
      defined in the problem, primarily from the initial state facts.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Remaining Goals:** Determine the set of nuts `N` that are part of the goal
        (i.e., `(tightened n)` is a goal predicate for `n` in `N`) but are not currently
        tightened in the given `state`. If this set is empty, the goal is reached, and
        the heuristic value is 0.
    2.  **Gather Current State Information:** Extract the following from the `state`:
        - The current location of the man (`man_loc`).
        - The spanner the man is currently carrying, if any (`carried_spanner`).
        - The set of all spanners that are currently `usable`.
        - Whether the `carried_spanner` (if one exists) is in the set of `usable` spanners (`is_carrying_usable`).
        - A dictionary mapping each `usable` spanner currently located on the ground (not carried)
          to its location (`usable_spanners_on_ground`).
    3.  **Calculate Cost per Remaining Nut:** Iterate through each nut `n` in the set of remaining goals:
        a. Retrieve the static location of the nut (`nut_loc`).
        b. Initialize the minimum cost estimated to tighten this specific nut (`cost_for_this_nut`) to infinity.
        c. **Consider Option 1: Using the Carried Spanner:**
           - If `is_carrying_usable` is true:
             i. Calculate the shortest path distance `d1 = distance(man_loc, nut_loc)`.
             ii. If `d1` is not infinity (i.e., the nut location is reachable):
                - Estimate cost as `d1` (walk actions) + 1 (tighten action).
                - Update `cost_for_this_nut = min(cost_for_this_nut, d1 + 1)`.
        d. **Consider Option 2: Picking Up a Spanner:**
           - If `usable_spanners_on_ground` is not empty:
             i. Initialize `min_pickup_route_cost` to infinity.
             ii. For each usable spanner `s` at location `spanner_loc` on the ground:
                - Calculate distance to spanner: `d_to_s = distance(man_loc, spanner_loc)`.
                - Calculate distance from spanner to nut: `d_s_to_n = distance(spanner_loc, nut_loc)`.
                - If both `d_to_s` and `d_s_to_n` are not infinity:
                  - Estimate cost for this route: `d_to_s` (walk) + 1 (pickup) + `d_s_to_n` (walk) + 1 (tighten).
                  - Update `min_pickup_route_cost = min(min_pickup_route_cost, d_to_s + 1 + d_s_to_n + 1)`.
             iii. Update `cost_for_this_nut = min(cost_for_this_nut, min_pickup_route_cost)`.
        e. **Check Reachability:** If, after considering both options, `cost_for_this_nut` remains infinity, it implies
           this specific nut cannot be tightened from the current state (e.g., no usable spanner can reach it,
           or the locations are disconnected). In this case, return a very large number (e.g., 1,000,000)
           to signify that this state is likely a dead end or very far from the goal.
        f. **Accumulate Cost:** Add the calculated `cost_for_this_nut` to the `total_heuristic_cost`.
    4.  **Return Total Estimated Cost:** The final heuristic value is the `total_heuristic_cost` accumulated across all remaining goal nuts.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information from the task.
        Builds the location graph, computes distances, identifies goal nuts,
        the man, and nut locations.
        """
        self.task = task
        self.goals = task.goals
        self.static = task.static

        # --- Extract Goal Nuts ---
        self.goal_nuts = set()
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            # Goal is typically (tightened ?n)
            if match(parts, "tightened", "?n"):
                self.goal_nuts.add(parts[1])

        # --- Build Location Graph & Compute Distances ---
        self.locations = set()
        self.adj = {} # Adjacency list for the location graph
        for fact in self.static:
            parts = get_parts(fact)
            # Links define the graph edges
            if match(parts, "link", "?l1", "?l2"):
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.adj.setdefault(l1, set()).add(l2)
                self.adj.setdefault(l2, set()).add(l1)

        # Compute and store all-pairs shortest paths
        self.distances = self._compute_all_pairs_shortest_paths()

        # --- Find the Man object ---
        # Assumption: Exactly one man exists. Find its name.
        self.man = None
        potential_men = set()
        # Look for objects involved in 'at' or 'carrying' predicates in the initial state
        for fact in task.initial_state:
             parts = get_parts(fact)
             if match(parts, "at", "?obj", "?loc"):
                 # Check if the location is known to filter out non-location objects
                 if parts[2] in self.locations:
                     # Crude check: if it's not obviously a nut or spanner by name
                     if not parts[1].startswith("nut") and not parts[1].startswith("spanner"):
                          potential_men.add(parts[1])
             elif match(parts, "carrying", "?m", "?s"):
                 potential_men.add(parts[1]) # Object carrying something is likely the man

        if len(potential_men) == 1:
            self.man = potential_men.pop()
        elif not potential_men:
             # Try finding man from operator parameters if possible (not directly available here)
             # Last resort: check all objects defined in the problem? Requires access to object list.
             # For now, raise error if ambiguous or not found.
             raise ValueError("Could not uniquely identify the 'man' object in the task based on initial state.")
        else:
             # This case suggests multiple potential men, violating the assumption.
             # We could pick one arbitrarily, but it's better to signal the issue.
             raise ValueError(f"Ambiguous 'man' object. Found: {potential_men}")


        # --- Find Nut Locations (assuming static) ---
        # Nuts locations are needed to calculate travel distances.
        self.nut_locations = {} # Map nut name -> location
        # Check initial state first
        for fact in task.initial_state:
            parts = get_parts(fact)
            if match(parts, "at", "?n", "?l"):
                 # Assume objects starting with 'nut' are nuts (relies on naming convention)
                 if parts[1].startswith("nut"):
                     self.nut_locations[parts[1]] = parts[2]

        # Verify all goal nuts have known locations
        for nut in self.goal_nuts:
            if nut not in self.nut_locations:
                # If not found in initial state, maybe it's static? (Unlikely for 'at')
                # Or maybe the problem assumes it starts tightened? (Handled by goal check later)
                # If a goal nut's location is unknown, the heuristic cannot be computed reliably.
                raise ValueError(f"Could not find initial location for goal nut: {nut}")

    def _bfs(self, start_node):
        """
        Performs Breadth-First Search starting from start_node on the location graph
        to find the shortest distance to all reachable locations.

        Args:
            start_node: The location to start the BFS from.

        Returns:
            A dictionary mapping reachable location names to their shortest distance
            (number of 'walk' steps) from start_node.
        """
        distances = {loc: float('inf') for loc in self.locations}
        if start_node not in self.locations:
             # Should not happen if graph is built correctly from static links
             return {}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_loc = queue.popleft()
            current_dist = distances[current_loc]

            # Explore neighbors
            for neighbor in self.adj.get(current_loc, set()):
                if distances[neighbor] == float('inf'): # If not visited yet
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        return distances

    def _compute_all_pairs_shortest_paths(self):
        """
        Computes the shortest path distance between all pairs of locations using BFS.

        Returns:
            A dictionary where keys are tuples (loc1, loc2) and values are the
            shortest distances. Returns float('inf') if locations are disconnected.
        """
        all_distances = {}
        for loc1 in self.locations:
            # Run BFS starting from loc1 to find distances to all other locations
            distances_from_loc1 = self._bfs(loc1)
            for loc2 in self.locations:
                # Store the computed distance (or infinity if unreachable)
                all_distances[(loc1, loc2)] = distances_from_loc1.get(loc2, float('inf'))
        return all_distances

    def get_dist(self, loc1, loc2):
        """
        Retrieves the precomputed shortest distance between two locations.

        Args:
            loc1: The starting location name.
            loc2: The destination location name.

        Returns:
            The shortest distance (integer) or float('inf') if unreachable or
            locations are invalid.
        """
        # Handle case where start and end are the same
        if loc1 == loc2:
            return 0
        return self.distances.get((loc1, loc2), float('inf'))

    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.

        Args:
            node: A node object containing the current state (.state).

        Returns:
            An estimated cost (integer or float) to reach the goal from the current state.
            Returns 0 if the goal is already satisfied. Returns a large number if the goal
            seems unreachable from the current state.
        """
        state = node.state

        # --- Find which goal nuts are already tightened ---
        tightened_nuts_in_state = set()
        for fact in state:
            parts = get_parts(fact)
            if match(parts, "tightened", "?n"):
                tightened_nuts_in_state.add(parts[1])

        # --- Identify the set of nuts that still need tightening ---
        remaining_nuts = self.goal_nuts - tightened_nuts_in_state
        if not remaining_nuts:
            # All goal nuts are tightened, goal is reached.
            return 0

        # --- Get current state details needed for calculation ---
        man_loc = None
        carried_spanner = None
        usable_spanners = set() # Set of names of usable spanners
        spanners_at_locs = {}   # Map from spanner name -> location (for spanners on the ground)

        for fact in state:
            parts = get_parts(fact)
            # Find man's location
            if match(parts, "at", self.man, "?loc"):
                man_loc = parts[2]
            # Find what the man is carrying
            elif match(parts, "carrying", self.man, "?s"):
                carried_spanner = parts[2]
            # Find all usable spanners
            elif match(parts, "usable", "?s"):
                usable_spanners.add(parts[1])
            # Find locations of objects that are not the man and not known nuts
            # Assume these are spanners (relies on object types/names)
            elif match(parts, "at", "?obj", "?loc"):
                 obj_name = parts[1]
                 if obj_name != self.man and obj_name not in self.nut_locations:
                     # Further check if it looks like a spanner? e.g., obj_name.startswith("spanner")
                     spanners_at_locs[obj_name] = parts[2]

        # Check if essential information was found
        if man_loc is None:
             # This indicates an invalid or unexpected state.
             # Return a high value or raise an error.
             # print(f"Warning: Could not find location for man '{self.man}' in state.")
             return float('inf')

        # Determine if the carried spanner is usable
        is_carrying_usable = carried_spanner is not None and carried_spanner in usable_spanners

        # Filter spanners on the ground to find only the usable ones
        usable_spanners_on_ground = {
            s: loc for s, loc in spanners_at_locs.items() if s in usable_spanners
        }

        # --- Calculate heuristic cost by summing costs for each remaining nut ---
        total_heuristic_cost = 0
        for nut in remaining_nuts:
            # Get the location of the current nut we need to tighten
            nut_loc = self.nut_locations.get(nut)
            if nut_loc is None:
                 # Should have been caught in init, but as a safeguard:
                 # print(f"Warning: Location unknown for remaining nut '{nut}'.")
                 return float('inf') # Cannot calculate cost without nut location

            cost_for_this_nut = float('inf')

            # --- Option 1: Use the spanner the man is already carrying ---
            if is_carrying_usable:
                walk_cost = self.get_dist(man_loc, nut_loc)
                if walk_cost != float('inf'):
                    # Cost = walking distance + 1 action (tighten)
                    cost_for_this_nut = min(cost_for_this_nut, walk_cost + 1)

            # --- Option 2: Pick up a usable spanner from the ground ---
            min_pickup_route_cost = float('inf')
            if usable_spanners_on_ground:
                for spanner, spanner_loc in usable_spanners_on_ground.items():
                    # Cost = walk to spanner + pickup + walk to nut + tighten
                    walk1_cost = self.get_dist(man_loc, spanner_loc)
                    walk2_cost = self.get_dist(spanner_loc, nut_loc)

                    # Check if both locations are reachable
                    if walk1_cost != float('inf') and walk2_cost != float('inf'):
                        current_pickup_cost = walk1_cost + 1 + walk2_cost + 1
                        min_pickup_route_cost = min(min_pickup_route_cost, current_pickup_cost)

            # Update the cost for this nut considering the pickup option
            cost_for_this_nut = min(cost_for_this_nut, min_pickup_route_cost)

            # --- Check if tightening this nut is possible ---
            if cost_for_this_nut == float('inf'):
                # If cost is still infinity, it means this nut cannot be tightened
                # from the current state (no usable spanner path found).
                # This state is likely a dead end regarding this goal nut.
                # Return a large value to strongly discourage exploring this path.
                return 1_000_000

            # --- Accumulate the cost for this nut ---
            total_heuristic_cost += cost_for_this_nut

        # The final heuristic value is the sum of minimum estimated costs for each remaining nut.
        return total_heuristic_cost

