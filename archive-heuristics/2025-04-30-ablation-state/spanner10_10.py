from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure fact has at least as many parts as args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    loose goal nuts. It considers the travel cost for the man to reach
    nut locations and spanner locations, plus the cost of pickup and tighten actions.
    It uses a greedy approach, assuming the man will sequentially address
    each loose goal nut by first going to the nearest available usable spanner,
    picking it up, and then going to the nearest remaining loose goal nut to tighten it.

    # Assumptions:
    - Nuts are static (do not change location).
    - Spanners are single-use (become unusable after one tighten action).
    - The man can carry only one spanner at a time.
    - There are enough usable spanners initially to tighten all goal nuts.
    - The location graph is connected such that all relevant locations (man's start, spanner locations, nut locations) are reachable from each other.
    - The man object can be identified from the initial state.

    # Heuristic Initialization
    - Parses static facts to build the location graph (links).
    - Collects all locations mentioned in static links and initial state 'at' facts.
    - Computes all-pairs shortest paths between all collected locations using BFS.
    - Identifies the static locations of all goal nut objects from the initial state.
    - Stores the set of goal conditions to identify goal nuts.
    - Identifies the man object name from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify all nuts that are loose and are part of the goal. These are the targets.
    3. Identify all spanners that are usable in the current state.
    4. Determine if the man is currently carrying a usable spanner.
    5. Identify the locations of usable spanners that are currently at locations (not carried).
    6. Identify the locations of loose goal nuts.
    7. Initialize the total heuristic cost to 0.
    8. Set the man's current location as the starting point for the sequence of actions.
    9. Create a list of available usable spanner locations (from step 5).
    10. Create a list of remaining loose goal nut locations (from step 6).
    11. Track whether the man is currently carrying a usable spanner (from step 4).
    12. Loop for the number of loose goal nuts identified in step 2:
       a. If the man is not currently carrying a usable spanner:
          - Find the nearest location from the current man's location among the available spanner locations (from step 9) using precomputed distances.
          - If no reachable spanner location is found, return infinity (problem likely unsolvable or heuristic limitation).
          - Add the travel distance to this spanner location to the total cost.
          - Update the man's current location to the spanner location.
          - Remove the chosen spanner location from the list of available spanner locations.
          - Add 1 to the total cost for the pickup action.
          - Mark the man as carrying a usable spanner.
       b. Find the nearest location from the current man's location among the remaining nut locations (from step 10) using precomputed distances.
       c. If no reachable nut location is found, return infinity (problem likely unsolvable or heuristic limitation).
       d. Add the travel distance to this nut location to the total cost.
       e. Update the man's current location to the nut location.
       f. Remove the chosen nut location from the list of remaining nut locations.
       g. Add 1 to the total cost for the tighten action.
       h. Mark the man as NOT carrying a usable spanner (since it was used).
    13. Return the total computed cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing distances.
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # Collect all locations mentioned in static links and initial state 'at' facts
        self.locations = set()
        self.links = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.links.setdefault(loc1, []).append(loc2)
                self.links.setdefault(loc2, []).append(loc1) # Links are bidirectional
                self.locations.add(loc1)
                self.locations.add(loc2)

        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 _, loc = get_parts(fact)[1:]
                 self.locations.add(loc)

        # Compute all-pairs shortest paths using BFS for all identified locations
        self.dist = {}
        for start_loc in list(self.locations):
            self.dist[start_loc] = self._bfs(start_loc)

        # Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                _, nut = get_parts(goal)
                self.goal_nuts.add(nut)

        # Identify static locations of nuts (assuming nuts are static)
        self.nut_locations = {}
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 if obj in self.goal_nuts: # Only track locations for goal nuts
                     self.nut_locations[obj] = loc

        # Identify the man object name (using the most robust inference based on initial state)
        self.man_name = None
        # Find the object that is 'at' a location in the initial state
        # and is not a goal nut and not an initial usable spanner.
        locatable_objects_init = {get_parts(f)[1] for f in initial_state if match(f, "at", "*", "*")}
        initial_usable_spanners = {get_parts(f)[1] for f in initial_state if match(f, "usable", "*")}
        man_candidates = locatable_objects_init - self.goal_nuts - initial_usable_spanners

        if len(man_candidates) == 1:
             self.man_name = list(man_candidates)[0]
        else:
             # Fallback: Try to find the object that is 'at' a location and is the first arg of a 'carrying' fact in initial state.
             man_from_carrying = None
             for fact in initial_state:
                 if match(fact, "carrying", "*", "*"):
                     man_from_carrying = get_parts(fact)[1]
                     break
             if man_from_carrying and man_from_carrying in locatable_objects_init:
                 self.man_name = man_from_carrying
             else:
                 # Last resort: Assume the first object in an initial 'at' fact is the man.
                 for fact in initial_state:
                     if match(fact, "at", "*", "*"):
                         self.man_name = get_parts(fact)[1]
                         break

        if self.man_name is None:
             # print("Error: Could not identify man object during heuristic initialization.")
             pass # Will handle in __call__ by returning inf


    def _bfs(self, start_loc):
        """
        Performs BFS from start_loc to find distances to all reachable locations within the known locations.
        Returns a dictionary {location: distance}.
        """
        distances = {loc: float('inf') for loc in self.locations}
        if start_loc not in self.locations:
             # This should not happen with the updated location collection in __init__
             # print(f"Warning: BFS started from location '{start_loc}' not in the collected locations set.")
             return {} # Cannot compute distances

        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            current_loc = queue.popleft()

            if current_loc in self.links:
                for neighbor in self.links[current_loc]:
                    # Ensure neighbor is one of the collected locations
                    if neighbor in self.locations and distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)
        return distances

    def get_location(self, obj, state):
        """Finds the current location of an object in the state."""
        # Check if object is at a location
        for fact in state:
            if match(fact, "at", obj, "*"):
                return get_parts(fact)[2]

        # Check if object is carried by the man
        # Need the man's name first
        current_man_name = None
        # Use the man_name identified during initialization
        if self.man_name:
             current_man_name = self.man_name
        else:
             # Fallback: Try to find the man in the current state if not found in init
             for fact in state:
                 if match(fact, "carrying", "*", "*"):
                      current_man_name = get_parts(fact)[1]
                      break
             # If current_man_name is still None, we cannot find the man.
             # This case should ideally be handled by the check for self.man_name in __call__.


        if current_man_name and obj != current_man_name: # Man cannot carry himself
            for fact in state:
                if match(fact, "carrying", current_man_name, obj):
                     # If carried, its location is the man's location
                     # Find the man's location directly from state facts
                     for man_loc_fact in state:
                          if match(man_loc_fact, "at", current_man_name, "*"):
                               return get_parts(man_loc_fact)[2]
                     # If man is carrying but not at a location, something is wrong with the state
                     return None # Should not happen


        return None # Object not found at a location or carried

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        if self.man_name is None:
             # Man object not identified during init, cannot compute heuristic
             # print("Heuristic cannot compute: Man object not identified.")
             return float('inf')

        # 1. Identify man's location
        man_location = self.get_location(self.man_name, state)

        if man_location is None:
             # Man's location not found in state, should not happen in valid states
             # print(f"Warning: Man location not found in state for man '{self.man_name}'. State: {state}")
             return float('inf') # Should not happen in valid states

        # Ensure man's current location is in the distance map keys
        if man_location not in self.dist:
             # This location was not in the initial links or initial 'at' facts.
             # This is unexpected in a well-formed problem.
             # print(f"Warning: Man's current location '{man_location}' not in precomputed distances.")
             # Attempt to compute distances from this new location if it's a known location name
             if man_location in self.locations:
                 self.dist[man_location] = self._bfs(man_location)
             # If it's still not in self.dist after _bfs or not reachable, return inf
             if man_location not in self.dist or not self.dist[man_location]: # Check if dist map is empty or start is unreachable
                  return float('inf')


        # 2. Identify loose goal nuts
        loose_goal_nuts = {n for n in self.goal_nuts if f"(loose {n})" in state}

        # If all goal nuts are tightened, heuristic is 0
        if not loose_goal_nuts:
            return 0

        # 3. Identify usable spanners currently in the state (at locations or carried)
        # We need to find all objects that are currently usable.
        usable_spanners_in_state = set()
        # Iterate through all objects that are mentioned in relevant predicates in the state
        potential_objects_in_state = set()
        for fact in state:
             parts = get_parts(fact)
             if parts[0] in ["at", "carrying", "usable", "loose", "tightened"]:
                  potential_objects_in_state.update(parts[1:]) # Add all arguments except predicate

        for obj in potential_objects_in_state:
             if f"(usable {obj})" in state:
                  usable_spanners_in_state.add(obj)


        # 4. Check if man is carrying a usable spanner
        man_carrying_usable_spanner = False
        carried_spanner_name = None
        for fact in state:
            if match(fact, "carrying", self.man_name, "*"):
                carried_spanner_name = get_parts(fact)[2]
                if carried_spanner_name in usable_spanners_in_state:
                    man_carrying_usable_spanner = True
                break # Assuming man carries at most one thing

        # 5. Get locations of usable spanners currently at locations
        # These are usable spanners that are NOT being carried by the man
        current_usable_spanner_locs = {} # {spanner_name: location}
        for spanner in usable_spanners_in_state:
             if spanner != carried_spanner_name: # Exclude the one being carried
                 loc = self.get_location(spanner, state)
                 if loc: # Should always have a location if not carried
                     current_usable_spanner_locs[spanner] = loc
                 # Ensure spanner location is in the distance map keys
                 if loc and loc not in self.dist:
                      # print(f"Warning: Usable spanner location '{loc}' not in precomputed distances.")
                      if loc in self.locations:
                           self.dist[loc] = self._bfs(loc)
                      if loc not in self.dist or not self.dist[loc]:
                           # Cannot compute distances from this spanner location
                           # This spanner is effectively unreachable for the heuristic calculation
                           # print(f"Heuristic returning inf: Cannot compute distances from usable spanner location '{loc}'.")
                           return float('inf')


        # 6. Get locations of loose goal nuts
        current_loose_nut_locs = {nut: self.nut_locations[nut] for nut in loose_goal_nuts}
        # Ensure nut locations are in the distance map keys
        for nut, loc in current_loose_nut_locs.items():
             if loc not in self.dist:
                  # print(f"Warning: Loose goal nut location '{loc}' not in precomputed distances.")
                  if loc in self.locations:
                       self.dist[loc] = self._bfs(loc)
                  if loc not in self.dist or not self.dist[loc]:
                       # Cannot compute distances to this nut location
                       # print(f"Heuristic returning inf: Cannot compute distances to loose goal nut location '{loc}'.")
                       return float('inf')


        # 7. Initialize cost and state for the greedy sequence
        total_cost = 0
        current_man_loc = man_location
        # Use a list of locations for available spanners
        available_spanner_locations_list = list(current_usable_spanner_locs.values())
        # Use a list of locations for remaining nuts
        remaining_nut_locations_list = list(current_loose_nut_locs.values())

        carrying_spanner_now = man_carrying_usable_spanner

        # Greedy loop: Address one nut at a time
        nuts_to_tighten_count = len(loose_goal_nuts)

        for _ in range(nuts_to_tighten_count):
            # Need a spanner?
            if not carrying_spanner_now:
                # Find nearest available spanner location from current_man_loc
                if not available_spanner_locations_list:
                    # Ran out of usable spanners at locations.
                    # This should not happen in solvable instances with enough spanners.
                    # print("Heuristic returning inf: Ran out of usable spanner locations.")
                    return float('inf')

                nearest_spanner_loc = None
                min_dist_to_spanner = float('inf')
                # Need to ensure current_man_loc is a valid key in self.dist
                if current_man_loc not in self.dist:
                     # This case should be handled by the check after getting man_location
                     # print(f"Heuristic returning inf: Current man location '{current_man_loc}' not in distance map keys before going to spanner.")
                     return float('inf')

                for s_loc in available_spanner_locations_list:
                    # Check if s_loc is a valid key in the distance map from current_man_loc
                    if s_loc in self.dist[current_man_loc]:
                         dist = self.dist[current_man_loc][s_loc]
                         if dist < min_dist_to_spanner:
                             min_dist_to_spanner = dist
                             nearest_spanner_loc = s_loc

                if nearest_spanner_loc is None:
                     # Cannot reach any available spanner location
                     # print("Heuristic returning inf: Cannot reach any usable spanner location from current man location.")
                     return float('inf')

                # Travel to spanner
                total_cost += min_dist_to_spanner
                current_man_loc = nearest_spanner_loc
                available_spanner_locations_list.remove(nearest_spanner_loc) # This spanner location is now 'used' conceptually for pickup

                # Pickup spanner
                total_cost += 1
                carrying_spanner_now = True

            # Need to go to a nut?
            # Find nearest remaining nut location from current_man_loc
            if not remaining_nut_locations_list:
                 # Should not happen if the loop runs nuts_to_tighten_count times
                 # and we remove one nut location per iteration.
                 break # Safety break

            nearest_nut_loc = None
            min_dist_to_nut = float('inf')
            # Need to ensure current_man_loc is a valid key in self.dist
            if current_man_loc not in self.dist:
                 # This case should be handled by the check after getting man_location
                 # print(f"Heuristic returning inf: Current man location '{current_man_loc}' not in distance map keys before going to nut.")
                 return float('inf')


            for n_loc in remaining_nut_locations_list:
                 # Check if n_loc is a valid key in the distance map from current_man_loc
                 if n_loc in self.dist[current_man_loc]:
                     dist = self.dist[current_man_loc][n_loc]
                     if dist < min_dist_to_nut:
                         min_dist_to_nut = dist
                         nearest_nut_loc = n_loc

            if nearest_nut_loc is None:
                 # Cannot reach any remaining nut location
                 # print("Heuristic returning inf: Cannot reach any remaining loose goal nut location from current man location.")
                 return float('inf')

            # Travel to nut
            total_cost += min_dist_to_nut
            current_man_loc = nearest_nut_loc
            remaining_nut_locations_list.remove(nearest_nut_loc) # This nut is about to be tightened

            # Tighten nut
            total_cost += 1
            carrying_spanner_now = False # Spanner is used up

        return total_cost
