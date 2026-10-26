from fnmatch import fnmatch
from collections import deque
import copy
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            pass


# Helper functions (copied and adapted from Logistics example)
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

# BFS function
def bfs_distances(start_location, link_facts, all_locations):
    """Computes shortest path distances from start_location to all other locations."""
    distances = {loc: float('inf') for loc in all_locations}
    if start_location not in all_locations:
         # Start location is not in the known locations graph, cannot reach anywhere
         return distances # All distances remain infinity

    distances[start_location] = 0
    queue = deque([start_location])

    # Build adjacency list from link facts
    adj = {loc: [] for loc in all_locations}
    for fact in link_facts:
        _, l1, l2 = get_parts(fact)
        if l1 in adj and l2 in adj: # Only add links between known locations
            adj[l1].append(l2)
            adj[l2].append(l1) # Links are bidirectional for walk
        # else: ignore links involving unknown locations

    while queue:
        current_loc = queue.popleft()

        for neighbor in adj.get(current_loc, []): # Use .get for safety
            if distances[neighbor] == float('inf'):
                 distances[neighbor] = distances[current_loc] + 1
                 queue.append(neighbor)

    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It uses a greedy approach that simulates the process of acquiring spanners
    and traveling to nut locations.

    # Assumptions
    - Spanners are consumed after tightening one nut.
    - Links between locations are bidirectional for the 'walk' action.
    - Nuts do not move from their initial locations.
    - The problem instance is solvable (enough usable spanners exist and locations are connected).
      If not solvable, the heuristic might return infinity.

    # Heuristic Initialization
    - Extracts all location objects and link facts to build the location graph.
    - Identifies all goal nuts and their fixed locations from the initial state.
    - Identifies the man object and all spanner objects from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic simulates a greedy plan to tighten all loose goal nuts:
    1. Identify the man's current location, the set of loose goal nuts,
       usable spanners carried by the man, and usable spanners at locations based on the current state.
    2. If there are no loose goal nuts, the heuristic is 0.
    3. Initialize total estimated cost to 0. Keep track of the man's current location
       and the sets of available spanners (carried and at locations).
    4. While there are still loose goal nuts remaining:
        a. For each remaining loose goal nut, calculate the estimated cost to get a spanner
           and reach the nut's location from the man's current location:
           - If the man is currently carrying a usable spanner: The cost is the travel distance
             from the man's current location to the nut's location.
           - If the man is not carrying a usable spanner: Find the closest location with an
             available usable spanner. The cost is the travel distance from the man's current
             location to the spanner location, plus 1 for the pickup action, plus the travel
             distance from the spanner location to the nut's location.
           - If no usable spanners are available anywhere, the problem is unsolvable from this state;
             return infinity.
        b. Select the loose goal nut that minimizes the cost calculated in step 4a.
        c. Add the minimum cost (travel + pickup if needed) to the total estimated cost.
           Add 1 for the 'tighten_nut' action.
        d. Update the man's current location to the location of the nut just processed.
        e. Mark the nut as processed (remove from the set of remaining nuts).
        f. Mark the spanner used as consumed (remove from carried set or location inventory).
        g. Recompute distances from the new man's location for the next iteration.
    5. Return the total estimated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about the task.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state
        self.static_facts = task.static

        # Identify man object, spanner objects, nut objects, and locations
        self.all_locations = set()
        self.link_facts = set()
        self.nut_locations = {} # Map nut object to its location (nuts don't move)
        self.goal_nuts = set()
        self.man_obj = None
        self.initial_spanners = set() # Store all spanner objects

        # Identify goal nuts from goal state
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'tightened':
                 self.goal_nuts.add(parts[1])

        # First pass on initial state to identify objects and initial locations/states
        initial_locations = {} # {obj: loc}
        initial_carried = set() # {spanner}
        initial_usable = set() # {spanner}

        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                initial_locations[obj] = loc
                self.all_locations.add(loc)
            elif parts[0] == 'carrying':
                carrier, obj = parts[1], parts[2]
                initial_carried.add(obj)
            elif parts[0] == 'usable':
                obj = parts[1]
                initial_usable.add(obj)
            elif parts[0] == 'loose':
                 # We don't need initial loose nuts here, only goal nuts that are loose in state
                 pass # Handled in __call__

        # Identify spanners: objects that are initially carried or usable or explicitly named spanner
        # This is a heuristic way to identify spanners without type info in facts
        all_initial_objects = set(initial_locations.keys()) | initial_carried | initial_usable | self.goal_nuts
        for obj in all_initial_objects:
             # Simple heuristic based on name or initial state facts
             if 'spanner' in obj.lower() or obj in initial_carried or obj in initial_usable:
                  self.initial_spanners.add(obj)


        # Identify man: the object that is initially at a location and is not a spanner or nut
        for obj in initial_locations:
             if obj not in self.initial_spanners and obj not in self.goal_nuts:
                  self.man_obj = obj
                  break # Assuming only one man

        # Store initial nut locations for goal nuts
        for nut in self.goal_nuts:
             if nut in initial_locations:
                  self.nut_locations[nut] = initial_locations[nut]
                  self.all_locations.add(initial_locations[nut]) # Ensure nut location is in all_locations
             # If a goal nut is not in initial_locations, its location is unknown, problem likely unsolvable.
             # The heuristic will handle this by returning infinity if nut_locations.get(nut) is None


        # Extract link facts and add locations from them
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                self.link_facts.add(fact)
                self.all_locations.add(parts[1])
                self.all_locations.add(parts[2])

        # Ensure man's initial location is in all_locations
        if self.man_obj and self.man_obj in initial_locations:
             self.all_locations.add(initial_locations[self.man_obj])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify state components
        current_man_location = None
        usable_spanners_carried = set()
        usable_spanners_at_loc = {} # {location: {spanner1, spanner2}, ...}
        loose_nuts_in_state = set()

        # Determine usability of spanners in the current state
        spanners_usable_in_state = {s for s in self.initial_spanners if f'(usable {s})' in state}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj == self.man_obj:
                    current_man_location = loc
                elif obj in spanners_usable_in_state: # It's a usable spanner
                    if loc not in usable_spanners_at_loc:
                        usable_spanners_at_loc[loc] = set()
                    usable_spanners_at_loc[loc].add(obj)
            elif parts[0] == 'carrying' and parts[1] == self.man_obj:
                 spanner = parts[2]
                 if spanner in spanners_usable_in_state: # It's a usable spanner
                     usable_spanners_carried.add(spanner)
            elif parts[0] == 'loose':
                 nut = parts[1]
                 loose_nuts_in_state.add(nut)

        # Identify loose goal nuts
        loose_goal_nuts = self.goal_nuts.intersection(loose_nuts_in_state)

        # 2. If no loose goal nuts, return 0.
        if not loose_goal_nuts:
            return 0

        # Check if man's current location is known
        if current_man_location is None:
             # Man's location is unknown, cannot plan
             return float('inf')

        # 3. Initialize variables for greedy simulation
        total_cost = 0
        current_loc = current_man_location
        remaining_nuts = set(loose_goal_nuts)
        available_carried = set(usable_spanners_carried)
        available_at_loc = copy.deepcopy(usable_spanners_at_loc)

        # 4. Greedy loop
        while remaining_nuts:
            # 4g. Recompute distances from the current man location for the current iteration
            distances_from_current_loc = bfs_distances(current_loc, self.link_facts, self.all_locations)

            min_cost_to_process_nut = float('inf')
            best_nut = None
            spanner_source_loc_for_best_nut = None # Location where the spanner for the best nut was obtained (current_loc if carried, L_S if picked up)
            used_carried_spanner_for_best = False

            # 4a. Calculate cost to get spanner and reach each remaining nut
            candidate_nuts = list(remaining_nuts) # Iterate over a copy
            for nut in candidate_nuts:
                nut_loc = self.nut_locations.get(nut)
                if nut_loc is None or distances_from_current_loc.get(nut_loc, float('inf')) == float('inf'):
                    # This nut's location is unknown or unreachable from current location
                    continue # Cannot process this nut from here

                cost_travel_pickup = float('inf')
                spanner_loc_for_this_nut = None
                used_carried_spanner_for_this = False

                # Option 1: Use a carried spanner (prioritized if available and reachable)
                if available_carried:
                    travel_cost = distances_from_current_loc[nut_loc] # Already checked reachability above
                    cost_travel_pickup_option1 = travel_cost
                    cost_travel_pickup = cost_travel_pickup_option1
                    spanner_loc_for_this_nut = current_loc # Conceptually, spanner is already with man
                    used_carried_spanner_for_this = True


                # Option 2: Pick up a spanner from a location
                # Only consider pickup if no carried spanners are available
                if not available_carried:
                    closest_spanner_loc = None
                    min_dist_to_spanner = float('inf')

                    for loc, spanners in available_at_loc.items():
                        if spanners: # Check if there are usable spanners at this location
                            dist_to_spanner_loc = distances_from_current_loc.get(loc, float('inf'))
                            if dist_to_spanner_loc != float('inf') and dist_to_spanner_loc < min_dist_to_spanner:
                                min_dist_to_spanner = dist_to_spanner_loc
                                closest_spanner_loc = loc

                    if closest_spanner_loc is not None:
                        # Calculate cost via pickup
                        # Need distance from spanner location to nut location
                        distances_from_spanner_loc = bfs_distances(closest_spanner_loc, self.link_facts, self.all_locations)
                        dist_spanner_to_nut = distances_from_spanner_loc.get(nut_loc, float('inf'))

                        if dist_spanner_to_nut != float('inf'):
                             cost_travel_pickup_option2 = min_dist_to_spanner + 1 + dist_spanner_to_nut
                             cost_travel_pickup = cost_travel_pickup_option2
                             spanner_loc_for_this_nut = closest_spanner_loc
                             used_carried_spanner_for_this = False # Explicitly false


                # If we found a way to get a spanner and reach the nut
                if cost_travel_pickup != float('inf'):
                    # Total cost for this nut includes travel/pickup + tighten
                    cost_for_this_nut = cost_travel_pickup + 1

                    # Select the nut with the minimum total cost
                    if cost_for_this_nut < min_cost_to_process_nut:
                        min_cost_to_process_nut = cost_for_this_nut
                        best_nut = nut
                        spanner_source_loc_for_best_nut = spanner_loc_for_this_nut # Store where the spanner came from
                        used_carried_spanner_for_best = used_carried_spanner_for_this


            # 4b. Check if any nut is reachable
            if best_nut is None:
                 # No remaining nut is reachable with an available spanner from the current location
                 # This state is likely a dead end for tightening remaining nuts
                 return float('inf') # Problem unsolvable from this state

            # 4c. Add cost and update total
            total_cost += min_cost_to_process_nut

            # 4d. Update man's location
            current_loc = self.nut_locations[best_nut]

            # 4e. Mark nut as processed
            remaining_nuts.remove(best_nut)

            # 4f. Mark spanner as consumed
            if used_carried_spanner_for_best:
                 # Remove an arbitrary spanner from the carried set
                 if available_carried: # Should not be empty if this path was chosen
                     available_carried.pop() # Remove an arbitrary element
            else: # Used a spanner from a location (spanner_source_loc_for_best_nut is the pickup location)
                 pickup_loc = spanner_source_loc_for_best_nut
                 if pickup_loc in available_at_loc and available_at_loc[pickup_loc]:
                     # Remove an arbitrary spanner from that location
                     # Find the spanner object to remove (arbitrary)
                     spanner_to_remove = next(iter(available_at_loc[pickup_loc]))
                     available_at_loc[pickup_loc].remove(spanner_to_remove)

                     if not available_at_loc[pickup_loc]:
                         del available_at_loc[pickup_loc] # Clean up empty location entry


        # 5. Return total estimated cost
        return total_cost
