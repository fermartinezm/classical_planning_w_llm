import os
import sys
from fnmatch import fnmatch
from collections import deque

# Ensure the base class directory is in the path if needed
# Example: sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Assuming Heuristic base class is accessible
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the import fails (e.g., for standalone testing)
    class Heuristic:
        def __init__(self, task): pass
        def __call__(self, node): raise NotImplementedError

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact string (e.g., "(at bob shed)")."""
    # Removes parentheses and splits by space. Handles potential extra spaces.
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Uses fnmatch for wildcard support in arguments.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts
    that are currently loose. It calculates the cost by summing the mandatory 'tighten'
    actions, the necessary 'pickup_spanner' actions, and an estimated cost for 'walk'
    actions based on a greedy simulation of the tightening process.

    # Assumptions
    - There is exactly one 'man' agent in the problem.
    - Links between locations ('link' predicate) define an undirected graph for calculating travel distances.
    - Each 'tighten_nut' action requires one unique usable spanner, which becomes unusable afterwards.
    - The state representation uses standard PDDL fact strings like '(predicate obj1 obj2)'.
    - Object type identification relies on predicate usage (e.g., 'carrying' implies man/spanner)
      or naming conventions (e.g., 'nut' in name) as explicit type info might not be readily available
      from the provided task structure.

    # Heuristic Initialization
    - Stores the task object, goal conditions, and static facts.
    - Identifies all locations from static 'link' facts and the initial state.
    - Builds a graph representation of the locations and links (adjacency list).
    - Pre-computes all-pairs shortest path distances between locations using Breadth-First Search (BFS).
      Unreachable locations have infinite distance.
    - Stores the set of nuts that need to be tightened according to the goal state ('goal_nuts').
    - Attempts to identify the single 'man' object name using predicate information from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse State:** Extract current facts:
        - Identify the man object (`man`). If not found during initialization, attempt again or fail.
        - Find the man's location (`man_location`).
        - Find the spanner carried by the man (`carried_spanner`), if any.
        - Check if the carried spanner is usable (`is_carried_spanner_usable`).
        - Collect the set of currently loose nuts (`loose_nuts`).
        - Record the location of each nut (`nut_locations`).
        - Record the location of each spanner (`all_spanner_locations`).
        - Identify usable spanners currently on the ground and their locations (`usable_spanners_at_loc`).
    2.  **Identify Remaining Goals:** Determine the set of nuts (`current_loose_goal_nuts`) that are in `loose_nuts` and also in `goal_nuts`. Let `num_remaining = len(current_loose_goal_nuts)`.
    3.  **Goal Check:** If `num_remaining` is 0, the goal conditions related to nuts are met, return heuristic value 0.
    4.  **Calculate Base Action Costs:**
        - Initialize heuristic value `h = 0`.
        - Add `num_remaining` to `h` (cost for `tighten_nut` actions).
        - Calculate the number of `pickup_spanner` actions needed: `num_pickups = num_remaining`. If `is_carried_spanner_usable` is true, decrement `num_pickups` by 1. Add `max(0, num_pickups)` to `h`.
    5.  **Check Resource Availability:** Verify if there are enough total usable spanners (carried + on ground) to tighten all remaining nuts. If not, return `float('inf')` as the state is likely unsolvable.
    6.  **Estimate Walk Costs (Greedy Simulation):**
        - Initialize simulation variables based on the current state: `current_man_loc_sim`, `carrying_usable_sim`, `available_spanners_sim` (copy of `usable_spanners_at_loc`), `nuts_to_tighten_sim` (copy of `current_loose_goal_nuts`).
        - Initialize `walk_cost = 0`.
        - Loop `num_remaining` times (once for each nut to be tightened):
            a.  **Find Best Trip:** Evaluate all possible next moves to tighten one nut from `nuts_to_tighten_sim`:
                i.  **Use Carried Spanner:** If `carrying_usable_sim` is true, calculate the walk cost (`dist(current_man_loc_sim, nut_loc)`) to reach each remaining nut's location (`nut_loc`).
                ii. **Pick Up New Spanner:** For each spanner `s` in `available_spanners_sim` at `spanner_loc`, and for each nut `n` in `nuts_to_tighten_sim` at `nut_loc`, calculate the walk cost (`dist(current_man_loc_sim, spanner_loc) + dist(spanner_loc, nut_loc)`).
            b.  **Select Minimum Cost Trip:** Choose the trip (from i or ii) with the minimum calculated walk cost. Store the chosen nut, cost, and spanner details.
            c.  **Handle Dead Ends:** If no possible trip is found (e.g., locations are unreachable, required objects missing), return `float('inf')` as the state seems unsolvable from this point.
            d.  **Update Walk Cost:** Add the minimum trip cost to `walk_cost`.
            e.  **Update Simulation State:**
                - Update `current_man_loc_sim` to the location of the nut just tightened.
                - Remove the tightened nut from `nuts_to_tighten_sim`.
                - If a carried spanner was used, set `carrying_usable_sim` to `False`.
                - If a new spanner was picked up, remove it from `available_spanners_sim` and set `carrying_usable_sim` to `False` (as it's immediately used and becomes unusable).
    7.  **Final Heuristic Value:** Return `max(0, h + walk_cost)`. Ensure the heuristic value is non-negative.
    """
    def __init__(self, task):
        super().__init__(task)
        self.task = task
        self.goals = task.goals
        static_facts = task.static

        # Extract locations and links
        self.locations = set()
        self.links = {} # Adjacency list: {loc: {neighbor1, neighbor2}}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.links.setdefault(l1, set()).add(l2)
                self.links.setdefault(l2, set()).add(l1) # Assume links are bidirectional

        # Add locations mentioned in init but maybe not in links
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                 # Ensure the location is known
                 self.locations.add(parts[2])

        # Precompute all-pairs shortest paths (APSP) using BFS
        self.distances = {} # {loc1: {loc2: distance}}
        all_locs = list(self.locations) # Use a fixed list for iteration
        for start_node in all_locs:
            self.distances[start_node] = {loc: float('inf') for loc in all_locs}
            self.distances[start_node][start_node] = 0
            queue = deque([(start_node, 0)])
            visited = {start_node} # Keep track of visited nodes in this BFS run

            while queue:
                current_node, dist = queue.popleft()

                neighbors = self.links.get(current_node, set())
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_dist = dist + 1
                        # Check if neighbor exists in distance map (should if all_locs is complete)
                        if neighbor in self.distances[start_node]:
                             self.distances[start_node][neighbor] = new_dist
                             queue.append((neighbor, new_dist))
                        # else: print(f"Warning: Neighbor {neighbor} not in location list during BFS from {start_node}")


        # Store goal nuts
        self.goal_nuts = {get_parts(g)[1] for g in self.goals if get_parts(g)[0] == 'tightened'}

        # Identify man object name (best effort during init)
        self._man_name = self._find_man_name(task)
        if self._man_name is None:
            print("Warning: Could not reliably determine the 'man' object name during initialization.")


    def _find_man_name(self, task):
        """Attempt to identify the single man object name from the task's initial state."""
        # Strategy 1: Look for objects involved in 'carrying' predicate
        carrying_subjects = set()
        man_loc_candidates = set() # Objects that have a location
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'carrying':
                # Subject (arg 1) is likely the man
                carrying_subjects.add(parts[1])
            elif parts[0] == 'at':
                man_loc_candidates.add(parts[1])

        # A man should be both 'at' a location and potentially 'carrying' something (or could carry)
        possible_men_carry = carrying_subjects.intersection(man_loc_candidates)
        if len(possible_men_carry) == 1:
            return possible_men_carry.pop()

        # Strategy 2: Infer from predicates (less reliable)
        # Find objects 'at' locations that are not clearly spanners or nuts
        spanners = set()
        nuts = set()
        for fact in task.initial_state:
             parts = get_parts(fact)
             pred = parts[0]
             if pred == 'usable': spanners.add(parts[1])
             if pred == 'loose' or pred == 'tightened': nuts.add(parts[1])
             if pred == 'carrying': spanners.add(parts[2]) # Object carried is spanner

        potential_men_infer = man_loc_candidates - spanners - nuts
        if len(potential_men_infer) == 1:
            return potential_men_infer.pop()

        # Strategy 3: Assume common names like 'bob' if present
        if 'bob' in man_loc_candidates:
            return 'bob'

        # Combine candidates if needed, but ambiguity is problematic
        # If multiple candidates from different methods, it's unclear.
        # Prioritize carrying predicate if it gave one result.
        if len(possible_men_carry) > 1:
             # print(f"Warning: Ambiguous man identification via carrying: {possible_men_carry}")
             # Maybe return the first one alphabetically?
             return sorted(list(possible_men_carry))[0]

        if len(potential_men_infer) > 1:
             # print(f"Warning: Ambiguous man identification via inference: {potential_men_infer}")
             return sorted(list(potential_men_infer))[0]

        # Fallback if no man found
        return None

    def __call__(self, node):
        state = node.state
        man = self._man_name

        # If man couldn't be identified in init, return high cost
        if man is None:
            # print("Error: Man object name unknown.")
            return float('inf')

        # --- 1. Parse current state ---
        man_location = None
        carried_spanner = None
        is_carried_spanner_usable = False
        loose_nuts = set()
        nut_locations = {} # {nut: location}
        usable_spanners = set() # Set of names of usable spanners
        all_spanner_locations = {} # {spanner: location}

        for fact in state:
            try:
                parts = get_parts(fact)
                if not parts: continue # Skip empty or malformed facts
                predicate = parts[0]
                args = parts[1:]

                if predicate == 'at' and len(args) == 2:
                    obj, loc = args
                    if obj == man:
                        man_location = loc
                    # Simple name check for type (can be improved if types available)
                    # Check if obj looks like a nut or spanner based on common naming patterns
                    # This is brittle and domain-specific.
                    if 'nut' in obj: # Example check
                        nut_locations[obj] = loc
                    if 'spanner' in obj: # Example check
                        all_spanner_locations[obj] = loc
                elif predicate == 'carrying' and len(args) == 2:
                    m, s = args
                    if m == man:
                        carried_spanner = s
                elif predicate == 'loose' and len(args) == 1:
                    loose_nuts.add(args[0])
                elif predicate == 'usable' and len(args) == 1:
                    usable_spanners.add(args[0])
            except IndexError:
                # Handle potential errors if a fact string is malformed
                # print(f"Warning: Malformed fact string encountered: {fact}")
                continue


        if man_location is None:
             # Man must be somewhere. If not found, state might be inconsistent or man name wrong.
             # print(f"Warning: Man '{man}' not found at any location in state.")
             return float('inf')

        # Determine usability of carried spanner
        if carried_spanner and carried_spanner in usable_spanners:
            is_carried_spanner_usable = True

        # Find usable spanners on the ground
        usable_spanners_at_loc = {} # {spanner: location}
        for spanner in usable_spanners:
            # Ensure it's not the one being carried and it has a location
            if spanner != carried_spanner and spanner in all_spanner_locations:
                usable_spanners_at_loc[spanner] = all_spanner_locations[spanner]

        # --- 2. Identify remaining goals ---
        current_loose_goal_nuts = loose_nuts.intersection(self.goal_nuts)
        num_remaining = len(current_loose_goal_nuts)

        if num_remaining == 0:
            return 0 # Goal state reached

        # --- 3. Calculate base action costs (tighten, pickup) ---
        h = 0
        h += num_remaining # Cost for tighten actions

        num_pickups_needed = num_remaining
        if is_carried_spanner_usable:
            num_pickups_needed -= 1
        h += max(0, num_pickups_needed) # Cost for pickup actions

        # --- 4. Check Resource Availability ---
        total_usable_spanners = len(usable_spanners_at_loc) + (1 if is_carried_spanner_usable else 0)
        if total_usable_spanners < num_remaining:
             # Not enough spanners to tighten remaining nuts
             return float('inf')

        # --- 5. Estimate walk cost using greedy simulation ---
        walk_cost = 0
        current_man_loc_sim = man_location
        carrying_usable_sim = is_carried_spanner_usable
        available_spanners_sim = dict(usable_spanners_at_loc) # Copy
        nuts_to_tighten_sim = set(current_loose_goal_nuts) # Copy
        # Keep track of the spanner being carried in simulation if applicable
        carried_spanner_sim = carried_spanner if carrying_usable_sim else None

        for _ in range(num_remaining):
            if not nuts_to_tighten_sim: break # Should finish exactly

            possible_trips = []

            # Option 1: Use carried spanner
            if carrying_usable_sim and carried_spanner_sim:
                for nut in nuts_to_tighten_sim:
                    nut_loc = nut_locations.get(nut)
                    if nut_loc is None: continue # Should have location

                    dist_to_nut = self.distances.get(current_man_loc_sim, {}).get(nut_loc, float('inf'))
                    if dist_to_nut != float('inf'):
                         possible_trips.append({'cost': dist_to_nut, 'nut': nut, 'spanner_loc': None, 'spanner': carried_spanner_sim})

            # Option 2: Pick up a spanner from the ground
            if available_spanners_sim:
                 for nut in nuts_to_tighten_sim:
                     nut_loc = nut_locations.get(nut)
                     if nut_loc is None: continue

                     for spanner, spanner_loc in available_spanners_sim.items():
                         dist_to_spanner = self.distances.get(current_man_loc_sim, {}).get(spanner_loc, float('inf'))
                         dist_spanner_to_nut = self.distances.get(spanner_loc, {}).get(nut_loc, float('inf'))

                         if dist_to_spanner != float('inf') and dist_spanner_to_nut != float('inf'):
                             total_dist = dist_to_spanner + dist_spanner_to_nut
                             possible_trips.append({'cost': total_dist, 'nut': nut, 'spanner_loc': spanner_loc, 'spanner': spanner})


            if not possible_trips:
                 # Cannot find a path to tighten any remaining nut.
                 return float('inf') # Indicate dead end / unsolvable from here

            # Find the best trip among all options (minimum cost)
            best_trip = min(possible_trips, key=lambda trip: trip['cost'])

            # Add cost and update simulation state
            walk_cost += best_trip['cost']

            chosen_nut = best_trip['nut']
            chosen_nut_loc = nut_locations[chosen_nut]
            current_man_loc_sim = chosen_nut_loc # Man moves to nut location
            nuts_to_tighten_sim.remove(chosen_nut)

            if best_trip['spanner_loc'] is None: # Used carried spanner
                carrying_usable_sim = False # Spanner becomes unusable
                carried_spanner_sim = None # No longer carrying a usable spanner
            else: # Picked up a new spanner
                chosen_spanner = best_trip['spanner']
                if chosen_spanner in available_spanners_sim:
                     del available_spanners_sim[chosen_spanner]
                # Man picks up, moves, tightens. Ends up not carrying a usable spanner.
                carrying_usable_sim = False
                carried_spanner_sim = None

        h += walk_cost
        # Ensure heuristic is non-negative
        return max(0, h)

