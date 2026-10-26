from fnmatch import fnmatch
from collections import deque

# Assume Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic: # Inherit from Heuristic in the actual planner environment
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    goal nuts. It does this by considering each loose goal nut sequentially.
    For each nut, it calculates the minimum cost to get the man to the nut's
    location carrying a usable spanner, given the man's current location
    and the set of available usable spanners. The cost includes walking,
    picking up a spanner (if needed), and tightening the nut. The man's
    location is updated after each nut is processed, and the used spanner
    is removed from the available set.

    # Assumptions
    - The goal is to tighten a specific set of nuts.
    - Each nut requires one usable spanner, and a spanner becomes unusable
      after one use.
    - Nuts' locations are static.
    - Spanners' initial locations (if not carried) are static.
    - The man is the only agent who can move and perform actions.
    - The graph of locations connected by 'link' predicates is connected,
      at least for relevant locations (man start, spanner locations, nut locations).
    - A man can carry multiple spanners simultaneously.

    # Heuristic Initialization
    - Identify all locations and the 'link' relationships from static facts
      to build a graph.
    - Compute all-pairs shortest path distances between locations using BFS.
    - Identify all nuts and their static locations from static facts.
    - Identify the man object from static facts.
    - Store the set of goal nuts from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the state to identify:
       - The man's current location.
       - The set of usable spanners.
       - The set of loose nuts.
       - The set of spanners the man is currently carrying.
       - The physical location of each object (spanners on the ground, nuts).
    2. Filter the loose nuts to include only those that are part of the goal.
       Let this set be `loose_goal_nuts`. If this set is empty, the goal is reached,
       and the heuristic is 0.
    3. Identify the set of usable spanners the man is currently carrying.
    4. If the total number of usable spanners (carried or on the ground) is less
       than the number of `loose_goal_nuts`, return a large value, as the problem
       is likely unsolvable from this state.
    5. Initialize the total cost to 0. Set the man's current location to his
       location in the input state. Keep track of the set of usable spanners
       that haven't been "used" yet in this heuristic calculation.
    6. If the man is initially carrying one or more usable spanners:
       - For each usable spanner the man is carrying (process in some order, e.g., alphabetical):
         - Select the next nut from `loose_goal_nuts` (e.g., the first one in a sorted list).
         - Calculate the cost to walk from the man's current location to the nut's
           location and tighten it (distance + 1). Add this cost to the total.
         - Remove the processed nut from `loose_goal_nuts`.
         - Remove the used spanner from the set of available usable spanners.
         - Update the man's current location to the nut's location.
    7. While there are still nuts remaining in `loose_goal_nuts`:
       a. Select the next nut from `loose_goal_nuts` (e.g., the first one in the sorted list).
       b. Get the nut's static location.
       c. Find the usable spanner from the *available* set that minimizes
          the cost of: walking from the man's current location to the spanner's
          location, picking up the spanner (cost 1), and walking from the
          spanner's location to the nut's location.
       d. Add this minimum travel+pickup cost plus the tighten cost (1) to the
          total cost.
       e. Remove the processed nut from `loose_goal_nuts` and the chosen spanner
          from the available set.
       f. Update the man's current location to the nut's location.
    8. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and
        precomputing shortest path distances.
        """
        # self.goals = task.goals # Assuming task object has goals attribute
        # self.static = task.static # Assuming task object has static attribute

        # Dummy attributes if not running within the planner framework
        try:
            self.goals = task.goals
            self.static = task.static
        except AttributeError:
            print("Warning: Running spannerHeuristic without a full Task object.")
            self.goals = frozenset()
            self.static = frozenset()


        # Extract all locations and links from static facts
        self.locations = set()
        self.links = {} # Adjacency list: location -> set of connected locations

        # Store static locations of nuts
        self.nut_locations = {}

        # Identify the man (assuming there's only one man object)
        self.man = None

        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'link':
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.links.setdefault(loc1, set()).add(loc2)
                self.links.setdefault(loc2, set()).add(loc1) # Links are bidirectional
            elif parts[0] == 'at' and 'nut' in parts[1]: # Assuming nut objects contain 'nut' in their name
                 # This is a static location for a nut
                 nut, loc = parts[1], parts[2]
                 self.nut_locations[nut] = loc
                 self.locations.add(loc) # Add nut location to known locations
            elif parts[0] == 'at' and 'man' in parts[1] and self.man is None: # Assuming man object contains 'man' in its name and there's only one
                 # This is the initial location of the man. We only need the object name.
                 self.man = parts[1]
                 self.locations.add(parts[2]) # Add man initial location to known locations

        # Ensure all locations mentioned in goals are included (nut locations are static)
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'tightened':
                 nut = parts[1]
                 # The location of the nut is needed, which is static
                 if nut in self.nut_locations:
                     self.locations.add(self.nut_locations[nut])

        # Compute all-pairs shortest paths using BFS
        self.shortest_paths = {}
        for start_loc in self.locations:
            self.shortest_paths[start_loc] = self._bfs(start_loc)

        # Store goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == 'tightened'}

    def _bfs(self, start_node):
        """
        Perform Breadth-First Search to find shortest distances from start_node
        to all other nodes in the location graph.
        """
        distances = {loc: float('inf') for loc in self.locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_loc = queue.popleft()

            if current_loc in self.links: # Check if current_loc has any links
                for neighbor in self.links[current_loc]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)

        return distances

    def get_distance(self, loc1, loc2):
        """Get the shortest distance between two locations."""
        if loc1 not in self.shortest_paths or loc2 not in self.shortest_paths[loc1]:
             # This can happen if a location from the state was not in static links
             # or if the graph is disconnected. Treat as unreachable.
             return float('inf')
        return self.shortest_paths[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Parse state
        current_man_location = None
        usable_spanners_in_state = set()
        loose_nuts_in_state = set()
        man_carried_spanners = set()

        # Map object to its current location in the state (only for objects that can be 'at' a location)
        obj_locations_in_state = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                obj_locations_in_state[obj] = loc
                if obj == self.man:
                    current_man_location = loc
            elif parts[0] == 'carrying':
                carrier, spanner = parts[1], parts[2]
                if carrier == self.man:
                    man_carried_spanners.add(spanner)
            elif parts[0] == 'usable':
                spanner = parts[1]
                usable_spanners_in_state.add(spanner)
            elif parts[0] == 'loose':
                nut = parts[1]
                loose_nuts_in_state.add(nut)

        # Ensure man location is found (should always be the case in valid states)
        if current_man_location is None:
             # This state is likely invalid or represents an unsolvable scenario
             return float('inf') # Or a large number

        # Determine initial locations for usable spanners in the input state
        spanner_locations_in_state = {} # Map usable spanner to its location (physical or man's location if carried)
        for spanner in usable_spanners_in_state:
             if spanner in man_carried_spanners:
                  # If carried, its location is the man's current location in the state
                  spanner_locations_in_state[spanner] = current_man_location
             elif spanner in obj_locations_in_state:
                  # If on the ground, its location is where it is 'at' in the state
                  spanner_locations_in_state[spanner] = obj_locations_in_state[spanner]
             # Else: Usable spanner exists but is not at a location and not carried? Ignore.


        # 2. Filter loose nuts to goal nuts
        loose_goal_nuts = list(loose_nuts_in_state.intersection(self.goal_nuts))

        # 3. Goal reached?
        if not loose_goal_nuts:
            return 0

        # 4. Check resource availability
        if len(loose_goal_nuts) > len(usable_spanners_in_state):
             # Not enough usable spanners for all goal nuts
             # Return a large value to indicate this state is likely not on a solution path
             return 1000 * len(self.goal_nuts)

        # Sort nuts for deterministic heuristic calculation (alphabetical)
        loose_goal_nuts.sort()

        total_cost = 0
        available_usable_spanners = set(usable_spanners_in_state)
        man_carried_usable_spanners = man_carried_spanners.intersection(available_usable_spanners)

        # 6. Handle initially carried usable spanners
        # Process nuts using carried spanners first
        carried_spanners_list = sorted(list(man_carried_usable_spanners)) # Deterministic order
        for carried_spanner in carried_spanners_list:
             if not loose_goal_nuts: break # Stop if all nuts are processed

             # Use the carried spanner for the next nut in the sorted list
             nut_to_tighten = loose_goal_nuts.pop(0)
             nut_location = self.nut_locations[nut_to_tighten]

             # Cost: walk from current man location to nut location + tighten
             cost = self.get_distance(current_man_location, nut_location) + 1
             total_cost += cost

             # Update state for next nut calculation
             available_usable_spanners.discard(carried_spanner)
             current_man_location = nut_location

        # 7. Process remaining nuts requiring pickup
        while loose_goal_nuts:
            # Pick the next nut in the sorted list
            nut_to_tighten = loose_goal_nuts.pop(0)
            nut_location = self.nut_locations[nut_to_tighten]

            # Find the best available usable spanner
            min_spanner_trip_cost = float('inf')
            best_spanner = None

            # Need to iterate over spanners that are *actually* available and have a known location
            spanners_to_consider = available_usable_spanners.intersection(spanner_locations_in_state.keys())

            if not spanners_to_consider:
                 # Should not happen if the initial check (step 4) passed,
                 # unless some usable spanners are not at a known location/carried.
                 # Treat as unsolvable from here.
                 return 1000 * len(self.goal_nuts) # Large value

            for spanner in spanners_to_consider:
                spanner_location = spanner_locations_in_state[spanner]

                # Cost to get this spanner and take it to the nut:
                # walk from man's current location to spanner's location + pickup + walk from spanner's location to nut's location
                cost_get_spanner = self.get_distance(current_man_location, spanner_location) + 1 + self.get_distance(spanner_location, nut_location)

                if cost_get_spanner < min_spanner_trip_cost:
                    min_spanner_trip_cost = cost_get_spanner
                    best_spanner = spanner

            # If no reachable usable spanner was found
            if best_spanner is None:
                 return 1000 * len(self.goal_nuts) # Large value

            # Add cost for this nut: spanner trip cost + tighten
            total_cost += min_spanner_trip_cost + 1

            # Update state for next nut calculation
            available_usable_spanners.discard(best_spanner)
            current_man_location = nut_location

        # 8. Return total cost
        return total_cost

