from collections import deque
from fnmatch import fnmatch
import math # Import math for infinity


# Helper functions to parse PDDL facts represented as strings
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
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    # A simpler check is just zipping and checking all match, fnmatch handles wildcards
    if len(parts) != len(args):
         return False # Added this check for robustness
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic: # Inherit from Heuristic if available in the environment
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It considers the cost of moving the man, picking up usable spanners, and moving
    the spanners (via the man) to the nut locations, and finally tightening the nuts.
    It greedily assigns available usable spanners to loose goal nuts, prioritizing
    the use of a spanner the man is already carrying, and then iteratively picking
    the nut-spanner pair that minimizes the travel and action cost from the man's
    current location.

    # Assumptions
    - Links between locations are bidirectional.
    - Each usable spanner can tighten exactly one nut.
    - The problem is unsolvable if there are fewer usable spanners than loose goal nuts.
    - All locations, nuts, spanners, and the man are reachable within the defined links.

    # Heuristic Initialization
    - Identify the man's name, all spanner names, all nut names, and all location names
      from static facts.
    - Identify all nuts that are goals (need to be tightened).
    - Build the location graph based on `link` facts.
    - Precompute all-pairs shortest paths (distances) between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify all usable spanners and their current locations (either on the ground or carried by the man).
    3. Identify all loose nuts that are goal conditions and their current locations.
    4. If the number of loose goal nuts exceeds the number of usable spanners, the problem is unsolvable; return infinity.
    5. If there are no loose goal nuts, the goal is reached; return 0.
    6. Initialize the total heuristic cost `h` to 0.
    7. Check if the man is currently carrying a spanner that is also usable according to the state.
    8. If the man is carrying a usable spanner:
       - Find the loose goal nut closest to the man's current location.
       - Calculate the cost to tighten this nut using the carried spanner: Distance(man_loc, nut_loc) + 1 (tighten).
       - Add this cost to `h`.
       - Update the man's current location to the nut's location.
       - Mark the carried spanner and the chosen nut as "used" for subsequent steps.
    9. Greedily assign the remaining loose goal nuts to the remaining usable spanners:
       - While there are still loose goal nuts to tighten:
         - Find the pair of (remaining loose goal nut, remaining usable spanner) that minimizes the total trip cost from the man's current location: Distance(man_loc, spanner_loc) + 1 (pickup) + Distance(spanner_loc, nut_loc) + 1 (tighten).
         - If no such reachable pair exists, the remaining nuts are unreachable; return infinity.
         - Add the minimum trip cost to `h`.
         - Update the man's current location to the location of the nut that was just tightened.
         - Mark the chosen nut and spanner as "used".
    10. Return the total calculated cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing distances.
        """
        self.goals = task.goals
        static_facts = task.static

        self.man_name = None
        self.all_spanners = set()
        self.all_nuts = set()
        locations = set()
        adj = {} # Adjacency list for location graph

        # Parse static facts to identify objects and build the location graph
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            predicate = parts[0]
            if predicate == 'man' and len(parts) == 2:
                self.man_name = parts[1]
            elif predicate == 'spanner' and len(parts) == 2:
                self.all_spanners.add(parts[1])
            elif predicate == 'nut' and len(parts) == 2:
                self.all_nuts.add(parts[1])
            elif predicate == 'location' and len(parts) == 2:
                 locations.add(parts[1])
            elif predicate == 'link' and len(parts) == 3:
                l1, l2 = parts[1:]
                adj.setdefault(l1, []).append(l2)
                adj.setdefault(l2, []).append(l1) # Assuming links are bidirectional
                locations.add(l1)
                locations.add(l2)

        # Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "?n"):
                self.goal_nuts.add(get_parts(goal)[1])

        # Compute all-pairs shortest paths (distances)
        self.distances = {}
        all_locations = list(locations) # Use a list for consistent iteration order if needed, though BFS doesn't require it
        for start_loc in all_locations:
            self.distances[start_loc] = {start_loc: 0}
            q = deque([(start_loc, 0)])
            visited = {start_loc}

            while q:
                curr_loc, dist = q.popleft()

                if curr_loc in adj:
                    for neighbor in adj[curr_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[start_loc][neighbor] = dist + 1
                            q.append((neighbor, dist + 1))

    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance, returns infinity if unreachable."""
        if loc1 == loc2:
            return 0
        if loc1 in self.distances and loc2 in self.distances[loc1]:
            return self.distances[loc1][loc2]
        return float('inf') # Locations are unreachable

    def __call__(self, node):
        """Compute the domain-dependent heuristic value for the given state."""
        state = node.state
        h = 0

        # 1. Identify man's current location
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man_name, "?loc"):
                man_loc = get_parts(fact)[2]
                break
        if man_loc is None:
             # Man location not found, should not happen in valid states
             return float('inf')

        # 2. Identify usable spanners and their locations
        usable_spanners = {} # {spanner_name: location}
        carried_spanner = None

        for fact in state:
            if match(fact, "usable", "?s"):
                spanner_name = get_parts(fact)[1]
                # Find spanner location (either at a location or carried)
                spanner_loc = None
                for loc_fact in state:
                    if match(loc_fact, "at", spanner_name, "?loc"):
                        spanner_loc = get_parts(loc_fact)[2]
                        break
                    if match(loc_fact, "carrying", self.man_name, spanner_name):
                        spanner_loc = man_loc # Carried spanner is at man's location
                        carried_spanner = spanner_name # Mark as carried
                        break
                if spanner_loc:
                    usable_spanners[spanner_name] = spanner_loc
                # else: usable spanner exists but is not at a location and not carried? Invalid state.


        # 3. Identify loose goal nuts and their locations
        loose_goal_nuts = {} # {nut_name: location}
        for nut_name in self.goal_nuts:
            if f'(loose {nut_name})' in state:
                 nut_loc = None
                 for fact in state:
                     if match(fact, "at", nut_name, "?loc"):
                         nut_loc = get_parts(fact)[2]
                         break
                 if nut_loc:
                     loose_goal_nuts[nut_name] = nut_loc
                 # else: loose goal nut exists but is not at any location? Invalid state.


        # 4. Check solvability based on resource count
        if len(loose_goal_nuts) > len(usable_spanners):
            return float('inf')

        # 5. Goal check
        if len(loose_goal_nuts) == 0:
            return 0

        current_man_loc = man_loc
        nuts_to_tighten = list(loose_goal_nuts.keys())
        available_spanners_info = list(usable_spanners.items()) # List of (spanner, loc) tuples

        # 8. Handle carried spanner first if available and usable
        # Check if the carried spanner is in the list of currently usable spanners
        is_carried_spanner_usable = carried_spanner is not None and carried_spanner in usable_spanners

        if is_carried_spanner_usable:
            nut_to_use_carried_on = None
            min_dist = float('inf')

            # Find the closest loose goal nut to the man's current location
            for nut in nuts_to_tighten:
                dist = self.get_distance(current_man_loc, loose_goal_nuts[nut])
                if dist < min_dist:
                    min_dist = dist
                    nut_to_use_carried_on = nut

            # Check if the closest nut is reachable
            if nut_to_use_carried_on is not None and min_dist != float('inf'):
                # Cost to use carried spanner for this nut: walk + tighten
                h += min_dist + 1
                # Update state for subsequent nuts
                current_man_loc = loose_goal_nuts[nut_to_use_carried_on]
                nuts_to_tighten.remove(nut_to_use_carried_on)
                # Remove the carried spanner from available list
                available_spanners_info = [(s, l) for s, l in available_spanners_info if s != carried_spanner]
            # else: Carried spanner cannot reach any remaining nut? This nut is unreachable.
            # The main loop will handle this if there are other nuts, or we return inf if this was the only nut.


        # 9. Greedily assign remaining nuts to remaining spanners
        # Find the pair (nut, spanner) that minimizes the trip cost from the man's current location
        while nuts_to_tighten:
            best_nut = None
            best_spanner = None
            min_trip_cost = float('inf')
            best_spanner_loc = None # Store spanner loc to remove correctly

            for nut in nuts_to_tighten:
                nut_loc = loose_goal_nuts[nut]
                for spanner, spanner_loc in available_spanners_info:
                    # Cost = walk from man to spanner + pickup + walk from spanner to nut + tighten
                    dist_man_to_spanner = self.get_distance(current_man_loc, spanner_loc)
                    dist_spanner_to_nut = self.get_distance(spanner_loc, nut_loc)

                    if dist_man_to_spanner == float('inf') or dist_spanner_to_nut == float('inf'):
                        continue # Cannot reach this spanner or cannot reach nut from spanner location

                    trip_cost = dist_man_to_spanner + 1 + dist_spanner_to_nut + 1

                    if trip_cost < min_trip_cost:
                        min_trip_cost = trip_cost
                        best_nut = nut
                        best_spanner = spanner
                        best_spanner_loc = spanner_loc


            if best_nut is not None and min_trip_cost != float('inf'):
                h += min_trip_cost
                current_man_loc = loose_goal_nuts[best_nut] # Man ends up at the nut location
                nuts_to_tighten.remove(best_nut)
                # Remove the used spanner from the available list
                available_spanners_info = [(s, l) for s, l in available_spanners_info if s != best_spanner]
            else:
                # Cannot tighten remaining nuts with available spanners (unreachable or no spanners left)
                return float('inf')

        # 10. Return total cost
        return h
