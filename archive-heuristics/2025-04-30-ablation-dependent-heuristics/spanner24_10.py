import math
from collections import deque

class spannerHeuristic:
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    The heuristic estimates the cost to reach the goal (tighten all target nuts)
    by simulating a greedy plan. It iteratively selects the loose goal nut
    that appears easiest to tighten next, calculates the estimated cost for
    that step (walking to the nut, acquiring a usable spanner if needed,
    and tightening), adds this cost to the total heuristic value, and updates
    the simulated state (man's location, available spanners, remaining nuts).
    This process repeats until all goal nuts are tightened.

    Assumptions:
    - The input task represents a solvable problem instance within the spanner domain.
    - There is a single man object identifiable by being the only object not
      classified as a spanner, nut, or location based on predicate usage.
    - Object types (man, spanner, nut, location) can be inferred from predicate
      usage in the initial state, goal, and static facts.
    - Location graph defined by 'link' facts is static.
    - Nut locations are static.

    Heuristic Initialization:
    In the constructor (__init__), the heuristic pre-processes the task information:
    1. Parses all facts from the initial state, goal, and static information.
    2. Identifies all objects and locations.
    3. Infers object types (man, spanners, nuts, locations) based on the predicates
       they appear in ('carrying', 'usable', 'loose', 'tightened', 'at', 'link').
    4. Stores the set of goal nuts.
    5. Stores the static location of each nut.
    6. Builds an adjacency list representation of the location graph based on 'link' facts.
    7. Computes all-pairs shortest path distances between all locations using BFS.
       These distances represent the minimum number of 'walk' actions required.

    Step-By-Step Thinking for Computing Heuristic:
    The heuristic function (__call__) takes a state and the task as input.
    1. Extracts the current state information: man's location, spanners carried,
       usable spanners at locations, and loose nuts.
    2. Identifies the set of loose nuts that are also goal nuts. If this set is empty,
       the goal is reached, and the heuristic returns 0.
    3. Initializes the total heuristic value `h` to 0.
    4. Initializes a simulated state for the greedy process: current man's location,
       sets of usable spanners available at locations and carried by the man, and
       the set of remaining loose goal nuts.
    5. Enters a loop that continues as long as there are remaining loose goal nuts:
        a. Finds the minimum estimated cost to tighten *any* single nut among the
           remaining loose goal nuts.
        b. For each remaining loose goal nut `n` at location `loc_n`:
            i. Calculates the cost to walk the man from the current simulated
               location to `loc_n`: `dist(current_man_loc, loc_n)`.
            ii. Calculates the cost to acquire a usable spanner at `loc_n`:
                - If the man is currently carrying a usable spanner, the cost is 0.
                - If not, it finds the nearest usable spanner `s` at `loc_s` among
                  those available at locations. The cost is estimated as the walk
                  to `loc_s` (`dist(current_man_loc, loc_s)`) plus 1 for the pickup
                  action, plus the walk from `loc_s` to `loc_n` (`dist(loc_s, loc_n)`).
                  If no usable spanners are available at locations, the cost to
                  acquire a spanner is considered infinite.
            iii. The cost to perform the 'tighten' action is 1.
            iv. The total estimated cost for this nut is the sum of walk-to-nut,
                spanner-acquisition, and tighten costs. Note that if a spanner
                is picked up, the walk-to-nut cost is implicitly included in the
                spanner-acquisition cost calculation (walk from spanner location
                to nut location).
        c. Selects the nut `best_nut` that has the minimum total estimated cost.
        d. If the minimum cost is infinite (no way to get a spanner or reach a nut),
           it means the remaining task is impossible from this state, and the
           heuristic returns a large number (e.g., 1000000).
        e. Adds the `min_cost_this_step` to the total heuristic value `h`.
        f. Updates the simulated state:
            - The man's location becomes `loc_of_best_nut`.
            - `best_nut` is removed from the set of remaining loose goal nuts.
            - The spanner used for tightening is consumed: if a carried spanner
              was used, one is removed from the set of carried spanners; if a
              spanner was picked up from a location, it is removed from the set
              of usable spanners at locations. The man is no longer carrying
              the consumed spanner (if he picked it up).
    7. Once the loop finishes (all loose goal nuts are processed), the accumulated
       value `h` is returned as the heuristic estimate.
    """

    def __init__(self, task):
        self.task = task
        self.all_objects = set()
        self.all_locations = set()
        self.man = None
        self.all_spanners = set()
        self.all_nuts = set()
        self.goal_nuts = set()
        self.nut_locs = {} # Static nut locations

        # Parse all facts to identify objects and locations
        all_facts_strs = set(task.initial_state) | set(task.goals) | set(task.static)
        all_facts = [self.parse_fact(f) for f in all_facts_strs]

        for pred, objs in all_facts:
            if pred and objs: # Ensure fact is valid
                self.all_objects.update(objs)
                if pred == 'link' and len(objs) > 1:
                    self.all_locations.update(objs)
                elif pred == 'at' and len(objs) > 1:
                    # The second object in 'at' is a location
                    self.all_locations.add(objs[1])

        # Identify potential spanners and nuts based on predicates they appear in
        potential_spanner_objs = set()
        potential_nut_objs = set()
        for pred, objs in all_facts:
            if pred == 'carrying' and len(objs) > 1:
                 potential_spanner_objs.add(objs[1]) # Spanner is the second arg
            elif pred == 'usable' and len(objs) > 0:
                potential_spanner_objs.add(objs[0]) # Spanner is the first arg
            elif (pred == 'loose' or pred == 'tightened') and len(objs) > 0:
                potential_nut_objs.add(objs[0]) # Nut is the first arg

        self.all_spanners = potential_spanner_objs
        self.all_nuts = potential_nut_objs

        # Identify the man: the object that is not a spanner, nut, or location
        non_snp_nut_loc = self.all_objects - self.all_spanners - self.all_nuts - self.all_locations
        if len(non_snp_nut_loc) == 1:
            self.man = list(non_snp_nut_loc)[0]
        elif len(non_snp_nut_loc) > 1:
             # Attempt to find the one in an 'at' fact that isn't classified
             found_man = False
             for pred, objs in all_facts:
                 if pred == 'at' and len(objs) > 0 and objs[0] in non_snp_nut_loc:
                     self.man = objs[0]
                     found_man = True
                     # print(f"Warning: Multiple objects not classified. Assuming '{self.man}' is the man based on initial 'at' fact.")
                     break
             if not found_man:
                  # Fallback to first non-classified object.
                  self.man = list(non_snp_nut_loc)[0]
                  # print(f"Warning: Multiple objects not classified. Assuming '{self.man}' is the man.")
        else:
             # Could not identify the man object.
             # print("Warning: Could not identify the man object.")
             self.man = None # Heuristic might return large value if man is needed

        # Populate nut_locs (static)
        for pred, objs in all_facts:
            if pred == 'at' and len(objs) > 1 and objs[0] in self.all_nuts:
                self.nut_locs[objs[0]] = objs[1]

        # Populate goal_nuts
        for pred, objs in all_facts:
            if pred == 'tightened' and len(objs) > 0 and objs[0] in self.all_nuts:
                self.goal_nuts.add(objs[0])

        # Build graph and compute distances
        self.graph = {loc: [] for loc in self.all_locations}
        for pred, objs in all_facts:
            if pred == 'link' and len(objs) > 1:
                l1, l2 = objs
                if l1 in self.graph and l2 in self.graph: # Ensure locations are known
                     self.graph[l1].append(l2)
                     self.graph[l2].append(l1)

        self.dist = {}
        for start_loc in self.all_locations:
            self.dist[start_loc] = {loc: float('inf') for loc in self.all_locations}
            self.dist[start_loc][start_loc] = 0
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            while q:
                current_loc, d = q.popleft()
                if current_loc in self.graph: # Ensure location is in graph
                    for neighbor in self.graph.get(current_loc, []): # Use .get for safety
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.dist[start_loc][neighbor] = d + 1
                            q.append((neighbor, d + 1))

    def parse_fact(self, fact_str):
        """Helper to parse PDDL fact string into predicate and objects."""
        # Removes surrounding brackets and splits by space
        fact_str = fact_str.strip()
        if not fact_str or fact_str[0] != '(' or fact_str[-1] != ')':
             # Not a valid fact string format
             return None, []
        parts = fact_str[1:-1].split()
        if not parts:
            return None, [] # Handle empty fact string inside brackets
        predicate = parts[0]
        objects = parts[1:]
        return predicate, objects

    def get_distance(self, loc1, loc2):
        """Looks up pre-calculated distance, returns infinity if locations unknown or path doesn't exist."""
        if loc1 not in self.dist or loc2 not in self.dist.get(loc1, {}):
            return float('inf')
        return self.dist[loc1][loc2]

    def __call__(self, state):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        if self.man is None:
             # Man object not identified during initialization, cannot compute heuristic
             return 1000000 # Large value indicating potential issue

        # Extract current state information
        man_loc = None
        spanner_locs_in_state = {} # {spanner: loc} for spanners at locations
        carried_spanners_in_state = set() # {spanner} carried by man
        usable_spanners_in_state = set() # {spanner} that are usable
        loose_nuts_in_state = set() # {nut} that are loose

        for fact_str in state:
            pred, objs = self.parse_fact(fact_str)
            if pred == 'at' and len(objs) > 1:
                obj, loc = objs
                if obj == self.man:
                    man_loc = loc
                elif obj in self.all_spanners:
                    spanner_locs_in_state[obj] = loc
            elif pred == 'carrying' and len(objs) > 1:
                m, s = objs
                if m == self.man and s in self.all_spanners:
                    carried_spanners_in_state.add(s)
            elif pred == 'usable' and len(objs) > 0 and objs[0] in self.all_spanners:
                usable_spanners_in_state.add(objs[0])
            elif pred == 'loose' and len(objs) > 0 and objs[0] in self.all_nuts:
                loose_nuts_in_state.add(objs[0])

        # If man_loc is not found in the state, it's an invalid state or initial state parsing issue
        if man_loc is None:
             # This state is unreachable or malformed
             return 1000000

        # Identify loose goal nuts
        loose_goal_nuts = [n for n in self.goal_nuts if n in loose_nuts_in_state]

        # Goal reached if no loose goal nuts remain
        if not loose_goal_nuts:
            return 0

        # --- Greedy Simulation for Heuristic Calculation ---
        h = 0
        current_man_loc = man_loc
        # Keep track of usable spanners available for pickup
        current_usable_spanners_at_loc = {
            s: loc for s, loc in spanner_locs_in_state.items()
            if s in usable_spanners_in_state
        }
        # Keep track of usable spanners carried by the man
        current_usable_spanners_carried = {
             s for s in carried_spanners_in_state
             if s in usable_spanners_in_state
        }
        remaining_loose_goal_nuts = set(loose_goal_nuts)

        while remaining_loose_goal_nuts:
            min_cost_this_step = float('inf')
            best_nut = None
            best_spanner_used_source = None # 'carried' or 'at_loc'
            best_spanner_picked_up = None # The specific spanner if source is 'at_loc'

            # Iterate through remaining nuts to find the easiest one to tighten
            for nut in remaining_loose_goal_nuts:
                nut_loc = self.nut_locs.get(nut)
                if nut_loc is None:
                    # This nut's location is unknown, cannot reach it.
                    continue # Skip this nut, it will remain in remaining_loose_goal_nuts

                cost_walk_to_nut = self.get_distance(current_man_loc, nut_loc)

                # Cost to get a usable spanner
                spanner_cost = 0
                spanner_used_source = None
                spanner_picked_up = None

                if current_usable_spanners_carried:
                    # Man is carrying a usable spanner, use one of them
                    spanner_cost = 0
                    spanner_used_source = 'carried'
                    # We don't need to identify which specific spanner yet, just that one is available

                else:
                    # Man is not carrying a usable spanner, need to pick one up
                    min_spanner_pickup_total_cost = float('inf')
                    temp_best_spanner_to_pickup = None
                    temp_best_spanner_pickup_loc = None

                    # Find the nearest usable spanner at a location
                    for spanner, s_loc in current_usable_spanners_at_loc.items():
                        # Cost to walk to spanner, pick it up, and walk to nut
                        walk_to_spanner_cost = self.get_distance(current_man_loc, s_loc)
                        walk_spanner_to_nut_cost = self.get_distance(s_loc, nut_loc)

                        if walk_to_spanner_cost == float('inf') or walk_spanner_to_nut_cost == float('inf'):
                             continue # Cannot reach this spanner or cannot reach nut from spanner

                        cost_via_this_spanner = (
                            walk_to_spanner_cost + # Walk to spanner
                            1 + # Pickup action
                            walk_spanner_to_nut_cost # Walk from spanner loc to nut loc
                        )
                        if cost_via_this_spanner < min_spanner_pickup_total_cost:
                            min_spanner_pickup_total_cost = cost_via_this_spanner
                            temp_best_spanner_to_pickup = spanner
                            temp_best_spanner_pickup_loc = s_loc

                    if temp_best_spanner_to_pickup is None:
                        # No usable spanners available anywhere for pickup that can reach the nut
                        spanner_cost = float('inf')
                    else:
                        spanner_cost = min_spanner_pickup_total_cost
                        spanner_used_source = 'at_loc'
                        spanner_picked_up = temp_best_spanner_to_pickup


                # Total cost for this nut = walk to nut + spanner cost + tighten action
                # Note: If spanner was picked up, the walk to nut is already included
                # in the spanner_cost calculation (dist(s_loc, nut_loc)).
                # If spanner was carried, the walk is dist(current_man_loc, nut_loc).

                total_cost_for_nut = float('inf')
                if spanner_cost == float('inf'):
                     total_cost_for_nut = float('inf') # Cannot tighten this nut
                elif spanner_used_source == 'carried':
                     total_cost_for_nut = cost_walk_to_nut + 1 # Walk to nut + Tighten
                elif spanner_used_source == 'at_loc':
                     # Spanner cost already includes walk to spanner, pickup, and walk from spanner to nut
                     total_cost_for_nut = spanner_cost + 1 # Spanner acquisition sequence + Tighten


                # Find the minimum cost among all remaining nuts
                if total_cost_for_nut < min_cost_this_step:
                    min_cost_this_step = total_cost_for_nut
                    best_nut = nut
                    best_spanner_used_source = spanner_used_source
                    best_spanner_picked_up = spanner_picked_up


            # End of loop over remaining nuts

            if min_cost_this_step == float('inf'):
                # Cannot tighten any more nuts (unreachable or no spanners)
                # Return a large value indicating this is likely a dead end
                return 1000000

            # Add the cost of the best step to the total heuristic
            h += min_cost_this_step

            # Update the simulated state
            nut_loc_of_best_nut = self.nut_locs.get(best_nut)
            if nut_loc_of_best_nut:
                current_man_loc = nut_loc_of_best_nut

            remaining_loose_goal_nuts.remove(best_nut)

            # Consume the spanner
            if best_spanner_used_source == 'carried':
                # Remove one usable spanner from the carried set
                if current_usable_spanners_carried:
                    current_usable_spanners_carried.pop() # Remove an arbitrary element
            elif best_spanner_used_source == 'at_loc':
                # Remove the picked-up spanner from the available-at-location set
                if best_spanner_picked_up in current_usable_spanners_at_loc:
                    del current_usable_spanners_at_loc[best_spanner_picked_up]

        # All loose goal nuts have been processed
        return h
