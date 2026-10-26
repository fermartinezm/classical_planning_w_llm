# Required imports
from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this base class is available

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at bob location1)" -> ["at", "bob", "location1"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It follows a greedy strategy: the man iteratively moves to the nearest loose nut,
    ensures he has a usable spanner (fetching one if necessary), and tightens the nut.
    The cost includes movement (walk), picking up a spanner, and tightening a nut.

    # Assumptions
    - Nuts are static (do not change location).
    - Spanners become unusable after one use for tightening a nut.
    - The man can carry at most one spanner at a time. Picking up a new spanner implies
      replacing the one currently carried (if any).
    - There are enough usable spanners available initially to tighten all nuts in the problem.
    - The location graph defined by 'link' predicates is connected for all relevant locations (man, nuts, spanners).
    - The man object name starts with 'bob' or 'man' (based on examples and common conventions).

    # Heuristic Initialization
    - Extract the goal conditions (all nuts tightened).
    - Extract static facts, specifically the 'link' predicates to build the location graph.
    - Precompute shortest path distances between all pairs of locations using BFS.
    - Identify all nut objects from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the man.
    2. Identify which spanner (if any) the man is carrying and its usability status.
    3. Identify the locations of all usable spanners currently on the ground.
    4. Identify all nuts that are still 'loose'.
    5. Initialize the total estimated cost to 0.
    6. Keep track of the man's current location and whether he is carrying a usable spanner in the heuristic's simulation.
    7. Maintain a set of usable spanner locations on the ground that are available to be picked up in the heuristic's simulation.
    8. While there are loose nuts remaining:
       a. Find the loose nut `N` whose location `L_N` is closest to the man's current location.
       b. Calculate the shortest distance `d1` from the man's current location to `L_N`. Add `d1` to the total cost. Update the man's current location to `L_N` for the simulation.
       c. Check if the man needs a usable spanner at `L_N`. This is true if he is not currently carrying a usable spanner in the simulation.
       d. If a usable spanner is needed:
          i. Find the usable spanner location `L_S` (among those on the ground available in the simulation) that is closest to the man's current location (`L_N`).
          ii. If no usable spanners are available on the ground, and the man doesn't have one, the problem is likely unsolvable (return a large value like float('inf')).
          iii. Calculate the shortest distance `d2` from the man's current location (`L_N`) to `L_S`. Add `d2` to the total cost. Update the man's current location to `L_S` for the simulation.
          iv. Add the cost of the `pickup_spanner` action (1) to the total cost.
          v. Mark the man as carrying a usable spanner in the simulation. Remove `L_S` from the set of available usable spanner locations on the ground in the simulation.
          vi. Calculate the shortest distance `d3` from the man's current location (`L_S`) back to the nut location (`L_N`). Add `d3` to the total cost. Update the man's current location to `L_N` for the simulation.
       e. Add the cost of the `tighten_nut` action (1) to the total cost.
       f. Mark the processed nut as no longer loose (by removing it from the list). Mark the spanner the man is carrying as no longer usable in the simulation.
       g. Re-sort the remaining loose nuts based on the man's new location for the next iteration.
    9. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and computing distances."""
        self.goals = task.goals
        self.static_facts = task.static

        # Extract all locations and links from static facts
        self.locations = set()
        self.links = {} # Adjacency list for the location graph

        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.links.setdefault(loc1, set()).add(loc2)
                self.links.setdefault(loc2, set()).add(loc1) # Links are bidirectional

        # Precompute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # Identify all nut objects from the goal conditions
        self.all_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'tightened':
                self.all_nuts.add(parts[1])

    def _bfs(self, start_location):
        """Perform BFS to find shortest distances from start_location to all other locations."""
        distances = {loc: float('inf') for loc in self.locations}

        if start_location in self.locations:
            distances[start_location] = 0
            queue = deque([start_location])

            while queue:
                current_loc = queue.popleft()
                current_dist = distances[current_loc]

                if current_loc in self.links:
                    for neighbor in self.links[current_loc]:
                        if distances[neighbor] == float('inf'):
                            distances[neighbor] = current_dist + 1
                            queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Get the precomputed shortest distance between two locations."""
        # Handle cases where locations might not be in the precomputed map
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
             return float('inf') # Path doesn't exist or location is unknown

        dist = self.distances[loc1][loc2]
        # If dist is inf, it means no path exists
        return dist


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify current state facts
        man_loc = None
        man_carrying_spanner = False
        carried_spanner_usable = False
        usable_spanner_locs_on_ground = set()
        loose_nuts = set()
        nut_locations = {} # Nuts are static, get their locations from the state

        # Assuming there is only one man object, and its name starts with 'bob' or 'man'
        man_obj = None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                # Infer man object name - this is a potential point of failure
                # if man object naming convention is different.
                if obj.startswith('bob') or obj.startswith('man'):
                     man_obj = obj
                     man_loc = loc
                elif obj in self.all_nuts:
                     nut_locations[obj] = loc
                elif obj.startswith('spanner') and '(usable ' + obj + ')' in state:
                     usable_spanner_locs_on_ground.add(loc)
            elif parts[0] == 'carrying':
                 carrier, spanner = parts[1], parts[2]
                 # Infer man object name if not found via 'at' yet
                 if man_obj is None and (carrier.startswith('bob') or carrier.startswith('man')):
                     man_obj = carrier
                 if carrier == man_obj: # Check if it's the identified man
                     man_carrying_spanner = True
                     if '(usable ' + spanner + ')' in state:
                         carried_spanner_usable = True
            elif parts[0] == 'loose':
                 nut = parts[1]
                 if nut in self.all_nuts: # Only consider nuts that are part of the goal
                    loose_nuts.add(nut)

        # If all nuts are tightened, heuristic is 0
        if not loose_nuts:
            return 0

        # Check if man_loc was found (should always be true in a valid state)
        if man_loc is None:
             # Cannot find the man's location, problem state is likely malformed
             return float('inf')

        total_cost = 0
        current_man_loc = man_loc
        current_man_has_usable_spanner = man_carrying_spanner and carried_spanner_usable
        # Need a mutable copy of usable spanner locations on ground for heuristic simulation
        available_usable_spanner_locs = set(usable_spanner_locs_on_ground)

        # Create a list of loose nuts to process
        nuts_to_tighten = list(loose_nuts)

        # Ensure all loose nuts have a known location
        if any(n not in nut_locations for n in nuts_to_tighten):
             # A loose nut's location is unknown, problem state is likely malformed
             return float('inf')


        while nuts_to_tighten:
            # Sort remaining loose nuts by distance from current man location
            # Use nut_locations[n] to get the location of nut n
            nuts_to_tighten.sort(key=lambda n: self.get_distance(current_man_loc, nut_locations[n]))

            # a. Find the nearest loose nut (it's now the first one after sorting)
            nut_to_process = nuts_to_tighten.pop(0)
            nut_loc = nut_locations[nut_to_process]

            # b. Cost to move to nut
            dist_to_nut = self.get_distance(current_man_loc, nut_loc)
            if dist_to_nut == float('inf'):
                 # Cannot reach the nut location from current position
                 return float('inf')

            total_cost += dist_to_nut
            current_man_loc = nut_loc # Man is now at the nut location

            # c. Check if the man needs a usable spanner at nut_loc
            if not current_man_has_usable_spanner:
                # i. Find the nearest usable spanner location
                nearest_spanner_loc = None
                min_dist_to_spanner = float('inf')

                # Consider spanners on the ground that are available in the simulation
                for s_loc in available_usable_spanner_locs:
                    dist = self.get_distance(current_man_loc, s_loc)
                    if dist < min_dist_to_spanner:
                        min_dist_to_spanner = dist
                        nearest_spanner_loc = s_loc

                if nearest_spanner_loc is None or min_dist_to_spanner == float('inf'):
                    # No usable spanners left to pick up (in simulation), and man doesn't have one
                    # This means the problem is likely unsolvable from this state
                    return float('inf')

                # iii. Cost to fetch spanner: move to spanner
                total_cost += min_dist_to_spanner
                current_man_loc = nearest_spanner_loc # Man is now at the spanner location

                # iv. Cost to pickup
                total_cost += 1

                # v. Update state for heuristic simulation: man now carries a usable spanner
                current_man_has_usable_spanner = True
                available_usable_spanner_locs.remove(nearest_spanner_loc) # This spanner is now carried

                # vi. Cost to return to nut location
                dist_return_to_nut = self.get_distance(current_man_loc, nut_loc)
                if dist_return_to_nut == float('inf'):
                     # Cannot return to the nut location from the spanner location
                     return float('inf')

                total_cost += dist_return_to_nut
                current_man_loc = nut_loc # Man is back at the nut location with a usable spanner

            # d. Cost to tighten nut
            total_cost += 1

            # e. Update state for heuristic simulation: spanner used, becomes unusable
            current_man_has_usable_spanner = False # The spanner carried is now unusable

            # The processed nut is removed from nuts_to_tighten by pop(0)

        # All nuts are processed
        return total_cost
