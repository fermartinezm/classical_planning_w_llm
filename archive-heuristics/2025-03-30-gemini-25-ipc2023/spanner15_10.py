import itertools
from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this base class exists and is imported

# Helper functions
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: "(at bob shed)" -> ["at", "bob", "shed"]
    """
    # Removes starting '(' and ending ')' then splits by space
    return fact[1:-1].split()

def match(fact, *args):
    """
    Checks if a PDDL fact matches a given pattern, allowing wildcards ('*').
    - `fact`: The fact string (e.g., "(at spanner1 location1)").
    - `args`: A sequence of strings representing the pattern (e.g., "at", "spanner*", "*").
    Returns `True` if the fact's parts match the pattern elements according to fnmatch, `False` otherwise.
    """
    parts = get_parts(fact)
    # The number of parts in the fact must match the number of elements in the pattern
    if len(parts) != len(args):
        return False
    # Check each part against the corresponding pattern argument using fnmatch for wildcard support
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    # Summary
    Estimates the cost to tighten all goal nuts in the Spanner domain for use with Greedy Best-First Search.
    The heuristic calculates the cost for each loose goal nut sequentially. It simulates the process
    where the man travels to a usable spanner (if not already carrying one), picks it up,
    travels to the nut's location, and tightens it. The choice of which spanner
    to use for each nut is made greedily to minimize the estimated travel distance
    for tightening that specific nut. The heuristic correctly accounts for the fact that each spanner
    can only be used once for a tighten action.

    # Assumptions
    - There is exactly one 'man' agent in the problem instance.
    - Nut locations are static; they do not change during the plan execution.
    - Each 'tighten_nut' action requires one 'usable' spanner as a precondition,
      and makes that spanner unusable as an effect.
    - The 'link' predicates define an undirected graph representing connectivity
      between locations. The cost of traversing a single link (walking) is 1.

    # Heuristic Initialization
    - Identifies the single 'man' object by checking predicates like 'carrying' or inferring from object lists.
      (Note: Relies on conventions like 'bob' or uniqueness if direct type info isn't available).
    - Identifies all objects of type 'nut' and 'spanner'.
    - Parses the static 'link' predicates from the task's static facts to build an
      adjacency list representation of the location graph.
    - Computes all-pairs shortest paths (APSP) using Breadth-First Search (BFS)
      starting from each location. Stores these distances in a dictionary for efficient O(1) lookup during heuristic evaluation.
    - Determines the fixed location for each 'nut' based on the 'at' predicates in the initial state.
    - Identifies the set of 'nut' objects that must be in the 'tightened' state according to the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Remaining Goals:** Find all nuts `n` such that `(tightened n)` is a goal predicate and `(loose n)` is true in the current state `s`. Let this set be `N_loose`.
    2.  **Goal Check:** If `N_loose` is empty, the current state satisfies the goal conditions related to nuts, so the heuristic value is 0. Return 0.
    3.  **Extract Current State Information:**
        - Find the man's current location (`l_m`) from `(at man loc)` facts.
        - Identify which spanner the man is carrying (`s_carried`), if any, from `(carrying man spanner)` facts.
        - Find all currently usable spanners (`S_usable`) from `(usable spanner)` facts.
        - Map all spanners currently on the ground to their locations (`L_spanners`) using `(at spanner loc)` facts.
    4.  **Check Resource Availability (Spanners):**
        - Determine if the carried spanner `s_carried` is usable (`s_carried_usable`).
        - Identify the set of usable spanners on the ground (`S_ground_usable`).
        - Calculate the total number of available usable spanners: `count_usable = (1 if s_carried_usable else 0) + len(S_ground_usable)`.
        - If `count_usable < len(N_loose)`, there are not enough usable spanners to tighten all remaining loose goal nuts. Return infinity (`float('inf')`) as the goal is unreachable from this state due to resource constraints.
    5.  **Initialize Simulation Variables:**
        - Set the total estimated cost `h = 0`.
        - Keep track of the man's simulated location, initialized to the current location: `l_current = l_m`.
        - Maintain a boolean flag indicating if the man is hypothetically carrying a usable spanner: `carrying_usable_sim = (s_carried_usable is not None)`.
        - Keep a mutable copy of the usable spanners on the ground for the simulation: `available_ground_spanners_sim = dict(S_ground_usable)`.
    6.  **Iterate Through Loose Nuts:** Process each nut `n` in `N_loose` one by one (the order of processing might affect the heuristic value but not correctness; an arbitrary order is used here).
        a.  Retrieve the location `l_n` of the current nut `n` (pre-calculated in `__init__`).
        b.  **Tighten Cost:** Increment `h` by 1 (representing the cost of the `tighten_nut` action itself).
        c.  **Spanner Acquisition & Travel Cost:**
            i.  **If `carrying_usable_sim` is true:** The simulation assumes the man uses the spanner they are carrying.
                - Calculate the walking distance: `dist_walk = get_dist(l_current, l_n)`.
                - If `dist_walk` is infinity (nut location unreachable), return infinity.
                - Add `dist_walk` to `h`.
                - Set `carrying_usable_sim = False` (the carried spanner is now considered used).
            ii. **If `carrying_usable_sim` is false:** The simulation assumes the man needs to pick up a spanner from the ground.
                - Increment `h` by 1 (representing the cost of the `pickup_spanner` action).
                - Find the "best" available spanner `s_best` from `available_ground_spanners_sim`. "Best" is defined as the spanner located at `l_s` that minimizes the total travel distance: `dist(l_current, l_s) + dist(l_s, l_n)`.
                - Iterate through `available_ground_spanners_sim` to find this `s_best` and the minimum combined distance `min_combined_dist`.
                - If no usable spanner can be found on the ground, or if all reachable spanners lead to unreachable nut locations (all combined distances are infinity), return infinity.
                - Add `min_combined_dist` to `h`.
                - Remove `s_best` from `available_ground_spanners_sim` (this spanner is now considered used).
        d.  **Update Simulated State:** Update the man's simulated location for the next iteration: `l_current = l_n`.
    7.  **Return Total Cost:** After iterating through all nuts in `N_loose`, return the final accumulated heuristic value `h`.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static
        # Use initial state to find static object properties like nut locations
        initial_state = task.initial_state
        self.infinity = float('inf')

        # --- Object Identification ---
        self.man = None
        self.nuts = set()
        self.spanners = set()
        self.locations = set()
        all_objects_in_init = set() # Keep track of all mentioned objects

        # First pass: identify locations from links, gather potential objects
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                self.locations.add(l1)
                self.locations.add(l2)
                all_objects_in_init.add(l1)
                all_objects_in_init.add(l2)

        # Second pass: use initial state to identify types and the man
        for fact in initial_state:
             parts = get_parts(fact)
             pred = parts[0]
             # Add all mentioned objects
             all_objects_in_init.update(parts[1:])
             # Use predicates to infer types and identify the man
             if pred == 'at': # (at ?obj ?loc)
                 obj, loc = parts[1], parts[2]
                 self.locations.add(loc) # Add location if not seen in links
                 # Infer type based on common usage/names if needed later
             elif pred == 'carrying': # (carrying ?m - man ?s - spanner)
                 self.man = parts[1] # Strong indicator for man
                 self.spanners.add(parts[2])
             elif pred == 'loose' or pred == 'tightened': # (loose ?n - nut), (tightened ?n - nut)
                 self.nuts.add(parts[1])
             elif pred == 'usable': # (usable ?s - spanner)
                 self.spanners.add(parts[1])

        # Refine object sets: remove locations from potential nuts/spanners/man
        potential_locatables = all_objects_in_init - self.locations
        self.nuts = self.nuts.intersection(potential_locatables)
        self.spanners = self.spanners.intersection(potential_locatables)

        # Attempt to find the man if not found via 'carrying'
        if not self.man:
             potential_men = potential_locatables - self.nuts - self.spanners
             if len(potential_men) == 1:
                 self.man = potential_men.pop()
             elif 'bob' in potential_locatables: # Fallback to common name 'bob'
                 self.man = 'bob'
             # Add more robust man detection if needed, e.g., checking action parameters
             if not self.man:
                 raise ValueError("Could not uniquely identify the 'man' object in the task.")

        # Final cleanup of sets
        self.nuts -= {self.man}
        self.spanners -= {self.man}


        # --- Goal Nuts ---
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nut_name = get_parts(goal)[1]
                if nut_name in self.nuts: # Ensure it's a known nut
                    self.goal_nuts.add(nut_name)

        # --- Nut Locations (Assume static from initial state) ---
        self.nut_locations = {}
        for fact in initial_state:
            if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                 if obj in self.nuts:
                     # Ensure the location is known
                     if loc in self.locations:
                         self.nut_locations[obj] = loc
                     else:
                         print(f"Warning: Nut '{obj}' is at unknown location '{loc}' in initial state.")
                         # Decide how to handle this - maybe treat as unreachable?
                         # For now, we store it but distance calculation might fail.

        # --- Location Graph and Distances ---
        adj = {loc: set() for loc in self.locations}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                # Check if locations are valid before adding links
                if l1 in self.locations and l2 in self.locations:
                    adj[l1].add(l2)
                    adj[l2].add(l1)

        self.distances = self._compute_distances(adj)


    def _compute_distances(self, adj):
        """Computes all-pairs shortest paths using BFS."""
        distances = {}
        # Initialize all distances to infinity
        for loc1 in self.locations:
            for loc2 in self.locations:
                distances[(loc1, loc2)] = self.infinity

        for start_node in self.locations:
            # Distance to self is 0
            distances[(start_node, start_node)] = 0
            queue = deque([(start_node, 0)])
            # Keep track of visited nodes and their shortest distance found so far
            visited_dist = {start_node: 0}

            while queue:
                current_node, dist = queue.popleft()

                # Explore neighbors
                for neighbor in adj.get(current_node, set()): # Use .get for safety
                    # If neighbor not visited or found a shorter path
                    if neighbor not in visited_dist or visited_dist[neighbor] > dist + 1:
                        new_dist = dist + 1
                        visited_dist[neighbor] = new_dist
                        distances[(start_node, neighbor)] = new_dist
                        queue.append((neighbor, new_dist))
        return distances

    def get_dist(self, loc1, loc2):
        """Returns the shortest distance between two locations."""
        # Ensure both locations are known, otherwise return infinity
        if loc1 not in self.locations or loc2 not in self.locations:
            return self.infinity
        # Default to infinity if pair not found (e.g., disconnected graph)
        return self.distances.get((loc1, loc2), self.infinity)


    def __call__(self, node):
        state = node.state

        # --- Identify loose goal nuts ---
        loose_goal_nuts = set()
        for nut in self.goal_nuts:
            # Check if the nut exists and is loose in the current state
            if f"(loose {nut})" in state:
                loose_goal_nuts.add(nut)

        if not loose_goal_nuts:
            return 0 # Goal reached

        # --- Current State Info ---
        man_loc = None
        carried_spanner = None
        usable_spanners_state = set() # Set of usable spanner names
        spanner_locations = {} # Location of spanners on ground

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "at":
                obj, loc = parts[1], parts[2]
                if obj == self.man:
                    man_loc = loc
                elif obj in self.spanners: # Check if obj is a known spanner
                    # Ensure location is known
                    if loc in self.locations:
                        spanner_locations[obj] = loc
                    # else: ignore spanner at unknown location?
            elif pred == "carrying" and parts[1] == self.man: # Ensure it's the known man carrying
                spanner_name = parts[2]
                if spanner_name in self.spanners: # Check if it's a known spanner
                    carried_spanner = spanner_name
            elif pred == "usable":
                spanner_name = parts[1]
                if spanner_name in self.spanners: # Check if obj is a known spanner
                     usable_spanners_state.add(spanner_name)

        if man_loc is None:
             # Man's location is essential
             # print(f"Warning: Man '{self.man}' location not found in state.")
             return self.infinity

        # --- Available Usable Spanners ---
        usable_carried = carried_spanner if carried_spanner in usable_spanners_state else None
        usable_on_ground = {
            s: loc for s, loc in spanner_locations.items() if s in usable_spanners_state
        }

        num_loose_nuts = len(loose_goal_nuts)
        num_usable_available = (1 if usable_carried else 0) + len(usable_on_ground)

        if num_usable_available < num_loose_nuts:
            # Not enough spanners to achieve the goal
            return self.infinity

        # --- Simulate Tightening Sequentially ---
        h = 0
        current_man_loc = man_loc
        # Flag to track if the simulator thinks the man holds a usable spanner
        carrying_usable_sim = usable_carried is not None
        # Create a mutable copy of usable spanners on ground for simulation
        available_ground_spanners_sim = dict(usable_on_ground)

        # Process nuts one by one. Using a list preserves an order, though arbitrary.
        # Sorting might improve heuristic quality but adds overhead.
        sorted_loose_nuts = list(loose_goal_nuts)

        for nut in sorted_loose_nuts:
            # Ensure nut location is known from initialization
            if nut not in self.nut_locations:
                 # print(f"Error: Location for nut '{nut}' was not found during init.")
                 return self.infinity # Cannot proceed without nut location
            nut_loc = self.nut_locations[nut]

            tighten_cost = 1
            pickup_cost = 0
            walk_cost = 0

            if carrying_usable_sim:
                # Use the hypothetically carried spanner
                dist_to_nut = self.get_dist(current_man_loc, nut_loc)
                if dist_to_nut == self.infinity: return self.infinity # Nut location unreachable
                walk_cost = dist_to_nut
                carrying_usable_sim = False # Spanner is now used up in simulation
            else:
                # Need to pick up a spanner from the ground
                if not available_ground_spanners_sim:
                    # Should not happen if initial check passed, but indicates an issue.
                    # print("Error: Ran out of simulated ground spanners unexpectedly.")
                    return self.infinity

                pickup_cost = 1
                best_spanner_for_nut = None
                min_combined_dist = self.infinity

                # Find spanner minimizing travel: dist(man->spanner) + dist(spanner->nut)
                for spanner, spanner_loc in available_ground_spanners_sim.items():
                    dist_to_spanner = self.get_dist(current_man_loc, spanner_loc)
                    dist_spanner_to_nut = self.get_dist(spanner_loc, nut_loc)

                    # Only consider paths where both segments are reachable
                    if dist_to_spanner != self.infinity and dist_spanner_to_nut != self.infinity:
                        combined_dist = dist_to_spanner + dist_spanner_to_nut
                        # Use '<=' to handle cases with equal distance, ensuring one is chosen
                        if combined_dist <= min_combined_dist:
                            min_combined_dist = combined_dist
                            best_spanner_for_nut = spanner

                if best_spanner_for_nut is None:
                     # No path found: man cannot reach any available usable spanner
                     # or cannot reach the nut from any available usable spanner.
                     return self.infinity

                walk_cost = min_combined_dist
                # Remove the chosen spanner from available ones for subsequent steps
                del available_ground_spanners_sim[best_spanner_for_nut]

            # Accumulate costs for tightening this nut
            h += tighten_cost + pickup_cost + walk_cost
            # Update man's simulated location for the next nut calculation
            current_man_loc = nut_loc

        return h
