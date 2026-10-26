import collections
import itertools
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

# Helper function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match facts (copied from Logistics example)
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Handle potential empty args list
    if len(args) == 0:
        return len(parts) == 0
    # Check if number of parts matches args, unless last arg is '*' which can match multiple parts
    # For simple PDDL facts, lengths usually match exactly.
    if len(parts) != len(args):
         # Allow '*' to potentially match if needed, but basic matching is sufficient here.
         # Let's assume exact length match for simplicity in this domain's facts.
         # If more complex matching is needed, this part needs refinement.
         pass # Allow check to proceed for now, fnmatch will handle '*'

    # Check individual parts
    # Ensure we don't compare beyond the number of parts available
    num_parts_to_compare = min(len(parts), len(args))
    for i in range(num_parts_to_compare):
        if not fnmatch(parts[i], args[i]):
            return False

    # If pattern is longer than parts, it's not a match unless extra args are '*'
    if len(args) > len(parts):
        # Check if remaining args are all '*' - simplistic check
        return all(arg == '*' for arg in args[len(parts):])

    # If parts is longer than pattern, it's not a match unless last pattern arg is '*'
    if len(parts) > len(args):
        return len(args) > 0 and args[-1] == '*'

    # If lengths match and all parts matched, return True
    return len(parts) == len(args)


class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts
    that are currently loose. It calculates the cost by summing:
    1. The number of `tighten_nut` actions needed (one per loose goal nut).
    2. The number of `pickup_spanner` actions needed (one for each spanner that
       needs to be picked up).
    3. An estimate of the `walk` actions required, calculated using a greedy
       strategy: the man repeatedly travels to the nearest required item
       (either a usable spanner or a loose nut).

    # Assumptions
    - There is exactly one man agent in the problem.
    - Nuts do not change location.
    - The `link` predicates define a static, bidirectional graph of locations.
    - The goal consists solely of `(tightened ?n)` predicates.
    - Each `tighten_nut` action makes the used spanner unusable.

    # Heuristic Initialization
    - Identifies all objects (man, spanners, nuts, locations) by parsing
      initial state, static facts, and operator definitions.
    - Stores the location of each nut (assumed static).
    - Stores the set of nuts that need to be tightened for the goal.
    - Builds a graph representation of the locations based on `link` predicates.
    - Precomputes all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Remaining Goals:** Determine the set of `LooseGoalNuts` - nuts that are part of the goal but are still `(loose)` in the current state.
    2.  **Goal Check:** If `LooseGoalNuts` is empty, the goal is reached, return heuristic value 0.
    3.  **Base Cost:** Initialize the heuristic cost `h` with the number of `LooseGoalNuts`. This accounts for the required `tighten_nut` actions.
    4.  **Agent & Spanner State:**
        - Find the man's current location (`man_loc`).
        - Identify the spanner the man is carrying (`carried_spanner`), if any.
        - Determine if the `carried_spanner` is `(usable)`.
        - Find all `(usable)` spanners currently on the ground and their locations (`usable_spanners_at`).
    5.  **Solvability Check:** Check if the total number of usable spanners (carried + on ground) is sufficient to tighten all `LooseGoalNuts`. If not, return infinity (unsolvable state).
    6.  **Pickup Cost:** Calculate how many spanners need to be picked up (`num_spanners_to_pickup`). This is `len(LooseGoalNuts)` minus 1 if the man is already carrying a usable spanner. Add this number to `h`.
    7.  **Travel Cost Estimation (Greedy Strategy):**
        - Initialize `travel_cost = 0`.
        - Keep track of the man's simulated location (`current_man_loc`), initially `man_loc`.
        - Keep track of remaining loose nuts and available spanners on the ground.
        - Maintain the state of the spanner currently held by the man (`spanner_in_hand`).
        - **Loop** while there are `LooseGoalNuts` remaining:
            a. **If man holds a usable spanner:**
               - Find the `LooseGoalNut` (`best_nut`) nearest to `current_man_loc`.
               - Add `distance(current_man_loc, location(best_nut))` to `travel_cost`.
               - Update `current_man_loc` to `location(best_nut)`.
               - Remove `best_nut` from the remaining set.
               - Set `spanner_in_hand` to `None` (spanner is used).
            b. **If man does not hold a usable spanner:**
               - Find the usable spanner on the ground (`best_spanner`) nearest to `current_man_loc`.
               - Add `distance(current_man_loc, location(best_spanner))` to `travel_cost`.
               - Update `current_man_loc` to `location(best_spanner)`.
               - Set `spanner_in_hand` to `best_spanner`.
               - Remove `best_spanner` from the available set on the ground.
               - **Now, holding a spanner**, find the `LooseGoalNut` (`best_nut`) nearest to the *new* `current_man_loc`.
               - Add `distance(current_man_loc, location(best_nut))` to `travel_cost`.
               - Update `current_man_loc` to `location(best_nut)`.
               - Remove `best_nut` from the remaining set.
               - Set `spanner_in_hand` to `None` (spanner is used).
    8.  **Total Cost:** Add the calculated `travel_cost` to `h`.
    9.  **Return:** Return the final heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static info and precomputing distances."""
        self.task = task
        self.goals = task.goals

        # --- Object and Type Extraction ---
        self.objects = collections.defaultdict(set)
        all_object_names = set()
        # Infer objects from initial state, goals, static facts, and operators
        sources = [task.initial_state, task.goals, task.static]
        # Also consider objects mentioned in operators' parameters (grounded names)
        for op in task.operators:
             sources.append({op.name}) # Add operator name itself to parse args
             sources.append(op.preconditions)
             sources.append(op.add_effects)
             sources.append(op.del_effects)

        for fact_set in sources:
            for fact in fact_set:
                # Skip facts that might not represent objects directly (e.g., predicates)
                try:
                    parts = get_parts(fact)
                    if len(parts) > 1: # Predicate name + args
                        all_object_names.update(parts[1:])
                except: # Handle potential errors if fact is not in expected format
                    pass


        # Infer types based on usage in initial state and static facts
        self.locations = set()
        self.nuts = set()
        self.spanners = set()
        self.men = set()

        # Use link predicates for locations
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'link':
                self.locations.add(parts[1])
                self.locations.add(parts[2])

        # Use initial state predicates for other types
        for fact in task.initial_state:
            parts = get_parts(fact)
            pred = parts[0]
            args = parts[1:]
            if pred == 'at':
                # If arg1 is not a known location, it might be locatable
                # If arg2 is a known location, arg1 is locatable
                if len(args) == 2 and args[1] in self.locations:
                    # Cannot determine specific type yet, but it's an object
                    all_object_names.add(args[0])
                elif len(args) == 2 and args[0] in self.locations:
                     # This case shouldn't happen based on domain typing
                     pass
            elif pred == 'carrying': # (carrying ?m - man ?s - spanner)
                if len(args) == 2:
                    self.men.add(args[0])
                    self.spanners.add(args[1])
            elif pred == 'usable': # (usable ?s - spanner)
                if len(args) == 1:
                    self.spanners.add(args[0])
            elif pred == 'tightened' or pred == 'loose': # (?n - nut)
                if len(args) == 1:
                    self.nuts.add(args[0])

        # Refine man identification if needed (e.g., from walk action)
        if not self.men:
             for op in task.operators:
                 # Grounded operator name might be like '(walk loc1 loc2 bob)'
                 op_parts = get_parts(op.name)
                 if op_parts[0] == 'walk' and len(op_parts) == 4:
                     # Assume the last argument is the man
                     man_candidate = op_parts[3]
                     # Check if it's not a known location, spanner, or nut
                     if (man_candidate not in self.locations and
                         man_candidate not in self.spanners and
                         man_candidate not in self.nuts):
                         self.men.add(man_candidate)
                         # Assume only one man
                         break

        # Ensure remaining objects are assigned a type if possible (e.g., locations)
        for obj_name in all_object_names:
             if (obj_name not in self.men and
                 obj_name not in self.spanners and
                 obj_name not in self.nuts and
                 obj_name not in self.locations):
                 # If it appears in the second position of 'at', assume location
                 is_location = False
                 for fact in task.initial_state:
                     parts = get_parts(fact)
                     if parts[0] == 'at' and len(parts) == 3 and parts[2] == obj_name:
                         is_location = True
                         break
                 if is_location:
                     self.locations.add(obj_name)
                 # Add more inference rules if needed


        if not self.men:
             raise ValueError("Could not identify the man agent in the problem.")
        if len(self.men) > 1:
             print(f"Warning: Found multiple men ({self.men}), heuristic assumes only one.")
        self.the_man = list(self.men)[0]


        # --- Static Information ---
        # Nut locations (assuming they don't move)
        self.nut_locations = {}
        for fact in task.initial_state:
             parts = get_parts(fact)
             # Ensure it's an 'at' fact with 3 parts (pred, obj, loc)
             if parts[0] == 'at' and len(parts) == 3 and parts[1] in self.nuts:
                 self.nut_locations[parts[1]] = parts[2]

        # Goal nuts
        self.goal_nuts = set()
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == 'tightened' and len(parts) == 2:
                self.goal_nuts.add(parts[1])

        # Location graph and distances
        self.adj = collections.defaultdict(list)
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'link' and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                # Ensure locations are known before adding links
                if loc1 in self.locations and loc2 in self.locations:
                    self.adj[loc1].append(loc2)
                    self.adj[loc2].append(loc1) # Assume links are bidirectional
                else:
                    print(f"Warning: Link between unknown locations? {fact}")


        # Precompute all-pairs shortest paths using BFS from each location
        self.distances = collections.defaultdict(lambda: float('inf'))
        for start_node in self.locations:
            if start_node not in self.adj and not any(start_node in v for v in self.adj.values()):
                 # Handle isolated locations if necessary
                 self.distances[start_node, start_node] = 0
                 continue

            self.distances[start_node, start_node] = 0
            queue = collections.deque([(start_node, 0)])
            visited = {start_node: 0}

            while queue:
                curr_loc, dist = queue.popleft()
                for neighbor in self.adj.get(curr_loc, []): # Use .get for safety
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        self.distances[start_node, neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
                    # Consider edge case: if a shorter path is found via another route
                    # This basic BFS finds shortest paths in unweighted graphs correctly.


    def _get_dist(self, loc1, loc2):
        """ Safely get precomputed distance, return infinity if no path or invalid loc."""
        if loc1 is None or loc2 is None or loc1 not in self.locations or loc2 not in self.locations:
            return float('inf')
        return self.distances.get((loc1, loc2), float('inf'))

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the given state node."""
        state = node.state

        # 1. Find loose goal nuts
        loose_goal_nuts = {n for n in self.goal_nuts if f"(loose {n})" in state}

        # 2. Goal check
        if not loose_goal_nuts:
            return 0

        # 3. Basic counts
        num_loose_nuts = len(loose_goal_nuts)
        # Cost for tighten_nut actions
        heuristic_cost = num_loose_nuts

        # 4. Agent & Spanner State
        man_loc = None
        carried_spanner = None
        for fact in state:
            parts = get_parts(fact)
            # Use match for robustness, though direct string comparison is faster if format is fixed
            if match(fact, "at", self.the_man, "*"):
                man_loc = parts[2]
            elif match(fact, "carrying", self.the_man, "*"):
                carried_spanner = parts[2]
            # Early exit if both found? Might miss other relevant facts. Iterate all.

        if man_loc is None:
             # This indicates an invalid or unexpected state
             print(f"Warning: Man {self.the_man} location not found in state.")
             return float('inf') # Cannot compute heuristic without man's location

        # Find usable spanners (location/carried)
        usable_spanners_on_ground = {} # {spanner_name: location}
        usable_spanner_carried = None
        usable_spanner_names = {get_parts(fact)[1] for fact in state if match(fact, "usable", "*")}

        for spanner in usable_spanner_names:
            if spanner == carried_spanner:
                usable_spanner_carried = spanner
            else:
                # Find location of this usable spanner on the ground
                spanner_loc = None
                for loc_fact in state:
                    if match(loc_fact, "at", spanner, "*"):
                        spanner_loc = get_parts(loc_fact)[2]
                        break
                if spanner_loc:
                    usable_spanners_on_ground[spanner] = spanner_loc
                # else: spanner is usable but not carried and not 'at' anywhere? Problematic state.

        # 5. Solvability Check
        num_spanners_available = len(usable_spanners_on_ground) + (1 if usable_spanner_carried else 0)
        if num_spanners_available < num_loose_nuts:
             # Not enough usable spanners for remaining nuts
             return float('inf')

        # 6. Pickup Cost
        num_spanners_to_pickup = num_loose_nuts - (1 if usable_spanner_carried else 0)
        # Ensure non-negative, although check above should guarantee this
        num_spanners_to_pickup = max(0, num_spanners_to_pickup)
        heuristic_cost += num_spanners_to_pickup # Cost for pickup_spanner actions

        # 7. Estimate travel costs using Greedy Assignment
        travel_cost = 0
        current_man_loc = man_loc
        # Make copies to modify during simulation
        remaining_loose_nuts = set(loose_goal_nuts)
        available_spanners_ground = dict(usable_spanners_on_ground)
        spanner_in_hand = usable_spanner_carried # Track spanner being carried

        while remaining_loose_nuts:
            if spanner_in_hand:
                # Find nearest nut to current location
                best_nut = None
                min_dist_nut = float('inf')
                target_nut_loc = None

                for nut in remaining_loose_nuts:
                    nut_loc = self.nut_locations.get(nut) # Use .get for safety
                    if nut_loc is None: continue # Skip if nut location unknown

                    dist = self._get_dist(current_man_loc, nut_loc)
                    if dist < min_dist_nut:
                        min_dist_nut = dist
                        best_nut = nut
                        target_nut_loc = nut_loc

                if best_nut is None or min_dist_nut == float('inf'):
                    # Cannot reach any remaining nut
                    return float('inf')

                travel_cost += min_dist_nut # Cost to walk to nut
                current_man_loc = target_nut_loc # Man moves to nut loc
                remaining_loose_nuts.remove(best_nut)
                spanner_in_hand = None # Spanner is used up (becomes unusable)

            else: # Need to pick up a spanner first
                if not available_spanners_ground:
                     # Should be caught by the earlier check, but as a safeguard
                     return float('inf')

                # Find nearest usable spanner on the ground to pick up
                best_spanner = None
                min_dist_spanner = float('inf')
                target_spanner_loc = None

                for spanner, loc in available_spanners_ground.items():
                    dist = self._get_dist(current_man_loc, loc)
                    if dist < min_dist_spanner:
                        min_dist_spanner = dist
                        best_spanner = spanner
                        target_spanner_loc = loc

                if best_spanner is None or min_dist_spanner == float('inf'):
                     # Cannot reach any available spanner
                     return float('inf')

                travel_cost += min_dist_spanner # Cost to walk to spanner
                current_man_loc = target_spanner_loc # Man moves to spanner loc
                spanner_in_hand = best_spanner # Man picks up spanner (pickup cost already added)
                del available_spanners_ground[best_spanner] # Remove spanner from ground

                # Now, with spanner in hand, find nearest nut from spanner's location
                best_nut = None
                min_dist_nut = float('inf')
                target_nut_loc = None

                for nut in remaining_loose_nuts:
                    nut_loc = self.nut_locations.get(nut)
                    if nut_loc is None: continue

                    dist = self._get_dist(current_man_loc, nut_loc)
                    if dist < min_dist_nut:
                        min_dist_nut = dist
                        best_nut = nut
                        target_nut_loc = loc # Typo? Should be nut_loc
                        target_nut_loc = nut_loc # Corrected

                if best_nut is None or min_dist_nut == float('inf'):
                     # Cannot reach any remaining nut from this spanner location
                     return float('inf')

                travel_cost += min_dist_nut # Cost to walk from spanner loc to nut loc
                current_man_loc = target_nut_loc # Man moves to nut loc
                remaining_loose_nuts.remove(best_nut)
                spanner_in_hand = None # Spanner is used up

        # 8. Total Cost
        heuristic_cost += travel_cost

        # 9. Return
        # Ensure heuristic is non-negative
        return max(0, heuristic_cost)

