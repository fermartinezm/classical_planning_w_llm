# from heuristics.heuristic_base import Heuristic # Assuming this base class exists

from fnmatch import fnmatch
from collections import deque
import math

# Helper functions from Logistics example
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define a dummy Heuristic base class if running standalone for testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the real one is not available
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
            # In a real scenario, task object would have more info like objects, init state etc.
            # For this heuristic, we only rely on goals and static facts from task init.
            # The state is passed to __call__.

        def __call__(self, node):
            raise NotImplementedError("Subclasses must implement this method")


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts
    that are part of the goal. It considers the travel distance for the man to reach
    spanners and nuts, the cost of picking up a spanner, and the cost of tightening
    a nut. It simulates a greedy process where the man addresses goal nuts one by one,
    always picking up the nearest available usable spanner to his current location
    before traveling to the nut.

    # Assumptions
    - The graph of locations defined by 'link' predicates is connected for all relevant
      locations (man start, spanner locations, nut locations).
    - There are enough usable spanners available in the initial state to tighten all
      goal nuts. If not, the heuristic returns infinity.
    - The man can only carry one spanner at a time.
    - Spanners become unusable after one use.
    - All goal nuts that need tightening are present at some location in the current state.

    # Heuristic Initialization
    - Build a graph of locations based on 'link' predicates found in static facts.
    - Compute all-pairs shortest path distances between these locations using BFS.
    - Identify the set of nuts that need to be tightened (goal nuts) from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic estimates the cost by simulating a greedy process:
    1. Extract the man's current location, the spanner he might be carrying, and its usability from the state.
    2. Extract the locations of all spanners and nuts from the state.
    3. Identify which spanners are currently usable.
    4. Identify which goal nuts are *not* tightened in the current state. These are the nuts that still need tightening.
    5. Filter the nuts needing tightening to include only those that are present at a location in the current state. If any goal nut needing tightening is not located, return infinity.
    6. If there are no goal nuts needing tightening (and located), the heuristic is 0 (goal achieved for relevant nuts).
    7. Initialize total cost to 0.
    8. Set the current man location to the actual man's location in the state.
    9. Create a set of available usable spanners based on the state (excluding the one the man might be carrying).
    10. Store locations of all spanners present in the state for quick lookup.
    11. If the man is currently carrying a usable spanner:
        - Find the goal nut needing tightening that is nearest to the man's current location.
        - Add the distance to this nut plus 1 (for the tighten action) to the total cost.
        - Update the man's current location to the nut's location.
        - Mark this nut as addressed (remove from the list of nuts to process). The carried spanner is now considered used/unusable.
    12. Sort the remaining goal nuts needing tightening to be processed (e.g., by their initial distance from the man's starting location). This fixed order helps make the heuristic deterministic.
    13. For each remaining goal nut needing tightening:
        - Find the available usable spanner that is nearest to the man's *current* location.
        - If no usable spanner is available, return infinity (problem likely unsolvable).
        - Add the distance to the chosen spanner's location plus 1 (for pickup) to the total cost.
        - Update the man's current location to the spanner's location.
        - Mark the chosen spanner as used (remove from the available list).
        - Add the distance from the spanner's location (man's current location) to the nut's location plus 1 (for tighten) to the total cost.
        - Update the man's current location to the nut's location.
    14. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances and identifying goal nuts.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Build the location graph from static facts
        self.graph = {} # Adjacency list: {loc: [neighbor1, neighbor2, ...]}
        all_locations_set = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                all_locations_set.add(l1)
                all_locations_set.add(l2)
                self.graph.setdefault(l1, []).append(l2)
                self.graph.setdefault(l2, []).append(l1) # Assuming links are bidirectional

        # 2. Compute all-pairs shortest paths using BFS
        self.distances = {}
        all_locations_list = list(all_locations_set) # Use a list for consistent iteration order

        for start_node in all_locations_list:
            self.distances[start_node] = {}
            queue = deque([(start_node, 0)])
            visited = {start_node}

            while queue:
                (current_node, dist) = queue.popleft()
                self.distances[start_node][current_node] = dist

                # Ensure current_node is in graph keys before accessing neighbors
                if current_node in self.graph:
                    for neighbor in self.graph[current_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))

        # Handle unreachable locations within the static graph
        for loc1 in all_locations_list:
             for loc2 in all_locations_list:
                 if loc2 not in self.distances[loc1]:
                     self.distances[loc1][loc2] = float('inf') # Cannot reach

        # 3. Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened":
                nut_name = args[0]
                self.goal_nuts.add(nut_name)

    def get_distance(self, loc1, loc2):
        """Safely get distance between two locations using precomputed distances."""
        # Check if locations are known from the static graph
        if loc1 not in self.distances or loc2 not in self.distances.get(loc1, {}):
             # This might happen if a location exists in the state but wasn't in static links.
             # Assuming all relevant locations are covered by static links for distance calculation.
             # If not, they are unreachable from the static graph, so distance is infinity.
             return float('inf')
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings)

        # Check if goal is already met for all goal nuts
        if all(f"(tightened {nut})" in state for nut in self.goal_nuts):
            return 0

        # Extract relevant information from the current state
        man_name = None
        man_location = None
        carried_spanner = None
        usable_spanners_in_state = set() # Names of usable spanners
        all_objects_loc = {}             # {object_name: location} # Includes man, spanners, nuts

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                all_objects_loc[obj] = loc
            elif parts[0] == 'carrying':
                man, spanner = parts[1], parts[2]
                man_name = man # Found the man's name
                carried_spanner = spanner
            elif parts[0] == 'usable':
                spanner = parts[1]
                usable_spanners_in_state.add(spanner)
            # We don't strictly need 'loose' or 'tightened' facts here, as we check against goal_nuts

        # Identify man's name and location
        if man_name is None:
             # If man isn't carrying anything, try to find him in 'at' facts.
             # This requires knowing the man's object name. Assume 'bob' if not found.
             # A robust planner would provide object types or names.
             # Let's try to find an object that is at a location and is not a known spanner or goal nut.
             potential_men = [obj for obj in all_objects_loc if obj not in usable_spanners_in_state and obj != carried_spanner and obj not in self.goal_nuts] # Exclude known spanners/nuts
             if potential_men:
                 man_name = potential_men[0] # Pick one if multiple? Assume one man.
             else:
                 # Cannot find the man object. Problem state is weird.
                 return float('inf') # Cannot solve without man

        man_location = all_objects_loc.get(man_name)
        if man_location is None:
             # Man is not at any location? Should not happen in valid states.
             return float('inf')

        # Identify nuts that are goal nuts and are not yet tightened
        nuts_needing_tightening = {nut for nut in self.goal_nuts if f"(tightened {nut})" not in state}

        # Identify locations of all nuts and spanners present in the state
        # We need locations for all objects that might be involved: man, spanners, nuts
        # all_objects_loc already contains locations for all objects with an 'at' fact.
        # We need to identify which objects are nuts and spanners.
        # Infer types from predicates they appear in (fragile without type info from task)
        all_nut_names_in_state = {p[1] for p in [get_parts(f) for f in state] if p[0] in ['loose', 'tightened']}
        all_spanner_names_in_state = {p[1] for p in [get_parts(f) for f in state] if p[0] in ['usable']}
        all_spanner_names_in_state.update({p[2] for p in [get_parts(f) for f in state] if p[0] == 'carrying'}) # Add carried spanner

        all_nuts_loc = {n: loc for n, loc in all_objects_loc.items() if n in all_nut_names_in_state}
        all_spanners_loc = {s: loc for s, loc in all_objects_loc.items() if s in all_spanner_names_in_state}


        carried_spanner_usable = (carried_spanner is not None) and (carried_spanner in usable_spanners_in_state)

        # Filter nuts needing tightening to include only those that are located in the state
        loose_goal_nuts_to_process = {nut for nut in nuts_needing_tightening if nut in all_nuts_loc}

        # If any goal nut needing tightening is not located, the problem is unsolvable from this state
        if len(nuts_needing_tightening) > len(loose_goal_nuts_to_process):
             return float('inf')

        # If no goal nuts need tightening (and are located), the goal is met
        if not loose_goal_nuts_to_process:
            return 0

        total_cost = 0
        current_man_location = man_location
        available_usable_spanners = {s for s in usable_spanners_in_state if s != carried_spanner} # Set of names

        # Store locations of available usable spanners for quick lookup
        available_usable_spanners_loc = {s: all_spanners_loc[s] for s in available_usable_spanners if s in all_spanners_loc}


        # Step 11: Handle carried usable spanner first
        if carried_spanner_usable:
            # Find the nearest goal nut needing tightening to the man's current location
            nearest_nut = None
            min_dist = float('inf')
            nearest_nut_loc = None

            # Iterate over nuts that need tightening AND are located
            for nut_name in loose_goal_nuts_to_process:
                nut_loc = all_nuts_loc.get(nut_name)
                # nut_loc should not be None here due to filtering above, but defensive check
                if nut_loc is None: continue

                dist = self.get_distance(current_man_location, nut_loc)
                if dist < min_dist:
                    min_dist = dist
                    nearest_nut = nut_name
                    nearest_nut_loc = nut_loc

            if nearest_nut is not None:
                # Cost to go to the nut and tighten it
                total_cost += self.get_distance(current_man_location, nearest_nut_loc) + 1 # +1 for tighten
                current_man_location = nearest_nut_loc
                loose_goal_nuts_to_process.remove(nearest_nut) # This nut is now addressed
                # The carried spanner is now unusable, no longer available.
            # else: This case means man carries usable spanner but no loose goal nuts remain.
            # This should be caught by the initial goal check.

        # Step 12: Sort remaining goal nuts needing tightening to be processed
        # Sort by distance from the *initial* man location for determinism
        initial_man_location = man_location # Store the original location
        sorted_loose_goal_nuts = sorted(list(loose_goal_nuts_to_process), # Convert set to list for sorting
                                        key=lambda nut_name: self.get_distance(initial_man_location, all_nuts_loc.get(nut_name, float('inf')))) # Use inf if nut location unknown

        # Step 13: Process remaining goal nuts needing tightening
        for nut_name in sorted_loose_goal_nuts:
            nut_loc = all_nuts_loc.get(nut_name)
            # nut_loc should not be None here due to filtering, but defensive check
            if nut_loc is None:
                 return float('inf') # Cannot reach nut

            # Find the nearest available usable spanner to the man's *current* location
            nearest_spanner = None
            min_travel_cost_to_spanner = float('inf')
            chosen_spanner_loc = None

            # Iterate over spanner names still available
            available_spanner_names_list = list(available_usable_spanners) # Convert set to list for consistent iteration order
            for spanner_name in available_spanner_names_list:
                 spanner_loc = available_usable_spanners_loc.get(spanner_name) # Get location
                 if spanner_loc is None:
                      # Should not happen if available_usable_spanners_loc is built correctly
                      continue # Skip this spanner

                 travel_cost = self.get_distance(current_man_location, spanner_loc)
                 if travel_cost < min_travel_cost_to_spanner:
                     min_travel_cost_to_spanner = travel_cost
                     nearest_spanner = spanner_name
                     chosen_spanner_loc = spanner_loc

            # If no usable spanner is available, the problem is likely unsolvable
            if nearest_spanner is None:
                return float('inf') # Cannot complete the task

            # Cost to go to spanner and pick it up
            total_cost += self.get_distance(current_man_location, chosen_spanner_loc) + 1 # +1 for pickup
            current_man_location = chosen_spanner_loc # Man is now at spanner location
            available_usable_spanners.remove(nearest_spanner) # Spanner is now used/carried
            del available_usable_spanners_loc[nearest_spanner] # Remove from location map

            # Cost to go from spanner location to nut location and tighten it
            total_cost += self.get_distance(current_man_location, nut_loc) + 1 # +1 for tighten
            current_man_location = nut_loc # Man is now at nut location
            # Nut is now tightened, removed implicitly by iterating over sorted_loose_goal_nuts

        return total_cost
