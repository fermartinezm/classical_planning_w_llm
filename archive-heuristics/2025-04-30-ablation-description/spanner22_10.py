from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    # Ensure we don't try to match more args than parts
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the cost to tighten all required nuts. It does this
    by considering each loose nut that needs tightening and calculating the
    estimated cost to get the man to the nut's location with a usable spanner
    and perform the tightening action. It processes nuts sequentially, accounting
    for the man's movement and spanner acquisition.

    # Assumptions
    - The man can carry only one spanner at a time (implied by the single `(carrying ?m ?s)` predicate).
    - A spanner becomes unusable after one tightening action.
    - The locations form an undirected graph connected by `link` predicates.
    - The cost of walk, pickup_spanner, and tighten_nut actions is 1.
    - The man's name can be identified from a `(carrying ?m ?s)` fact if present,
      otherwise it is assumed to be the first object in an `(at ?obj ?loc)` fact
      that is not a goal nut. If still not found, it defaults to 'bob'.

    # Heuristic Initialization
    - Extracts the set of nuts that need to be tightened from the goal state.
    - Builds a graph of locations based on `link` predicates.
    - Computes shortest path distances between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all loose nuts that are part of the goal state (i.e., need to be tightened). If none, heuristic is 0.
    2. Extract the man's current location, identify if he is carrying a usable spanner, and find the locations of all usable spanners not currently carried by him. The man's name is determined by looking for the object in a `(carrying ?m ?s)` fact; if none exists, it attempts to find the first object in an `(at ?obj ?loc)` fact that is not a goal nut; if still not found, it defaults to 'bob'.
    3. Initialize total heuristic cost to 0.
    4. Set the man's current location for calculation purposes to his actual current location from the state.
    5. Set a flag `man_has_usable_spanner` indicating if the man currently possesses a usable spanner based on the state.
    6. Create a mutable dictionary `current_available_usable_spanners` mapping spanner names to locations for usable spanners not carried by the man.
    7. Create a list of the loose nuts that need tightening. Sort this list by the shortest distance from the man's initial location in the state to the nut's location.
    8. Iterate through the sorted list of nuts:
       a. Get the location of the current nut (`nut_loc`).
       b. Calculate the shortest distance from the man's current calculated location (`current_man_loc`) to `nut_loc`. If unreachable, return infinity.
       c. Add this distance to the total cost. Update `current_man_loc` to `nut_loc`.
       d. If `man_has_usable_spanner` is False:
          i. Find the usable spanner in `current_available_usable_spanners` that is closest to the man's current calculated location.
          ii. If no usable spanners are available, return infinity (unsolvable).
          iii. Calculate the shortest distance from `current_man_loc` to the closest spanner's location (`spanner_loc`). If unreachable, return infinity.
          iv. Add this distance to the total cost. Update `current_man_loc` to `spanner_loc`.
          v. Add 1 to the cost for the `pickup_spanner` action.
          vi. Set `man_has_usable_spanner` to True. Remove the picked-up spanner from `current_available_usable_spanners`.
       e. Add 1 to the cost for the `tighten_nut` action.
       f. Set `man_has_usable_spanner` to False (as the spanner used became unusable).
    9. Return the total cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and precomputing distances."""
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not affected by actions.
        self.static = task.static

        # Extract goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Build location graph and compute distances
        self.locations = set()
        self.location_graph = {}

        # Find all locations from static links
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)

        # Initialize graph
        for loc in self.locations:
            self.location_graph[loc] = []

        # Add edges from links
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph[loc1].append(loc2)
                self.location_graph[loc2].append(loc1) # Links are bidirectional

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_loc):
        """Performs BFS from a start location to find distances to all other locations."""
        dist = {loc: float('inf') for loc in self.locations}
        if start_loc in dist: # Ensure start_loc is a known location
            dist[start_loc] = 0
            queue = deque([start_loc])

            while queue:
                curr = queue.popleft()
                if curr in self.location_graph: # Ensure location exists in graph
                    for neighbor in self.location_graph[curr]:
                        if dist[neighbor] == float('inf'):
                            dist[neighbor] = dist[curr] + 1
                            queue.append(neighbor)
        return dist

    def get_distance(self, loc1, loc2):
        """Returns the shortest distance between two locations."""
        # Check if start location is in our precomputed distances
        if loc1 not in self.distances:
             # This location wasn't a node in the graph built from links.
             # If loc1 and loc2 are the same, distance is 0.
             if loc1 == loc2: return 0
             # Otherwise, it's unreachable from/to a known graph.
             return float('inf')

        # Check if end location is reachable from start location
        if loc2 not in self.distances[loc1]:
             # loc2 is not reachable from loc1
             return float('inf')

        return self.distances[loc1][loc2]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify loose nuts that need tightening
        loose_goal_nuts = {
            nut for nut in self.goal_nuts
            if f"(loose {nut})" in state
        }

        # If all goal nuts are tightened, heuristic is 0.
        if not loose_goal_nuts:
            return 0

        # 2. Extract current state information
        man_name = None
        carried_spanner = None # The specific spanner carried
        man_has_usable_spanner = False # Flag indicating if the carried spanner is usable

        # First pass to find man's name and carried spanner
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "carrying":
                 man_name = parts[1]
                 carried_spanner = parts[2]
                 # Now check if this carried spanner is usable
                 if f"(usable {carried_spanner})" in state:
                     man_has_usable_spanner = True
                 break # Assuming only one man and one carried spanner fact

        # If man_name wasn't found via 'carrying', try finding him via 'at'
        # This is still a guess without type info. Let's assume the first object
        # in an 'at' fact that is not a goal nut is the man.
        if man_name is None:
             for fact in state:
                 parts = get_parts(fact)
                 if parts[0] == "at":
                     obj = parts[1]
                     # Check if this object is NOT a nut from the goal
                     if obj not in self.goal_nuts:
                         # Assume this is the man. This is fragile.
                         man_name = obj
                         break
             # If still not found, fall back to 'bob'
             if man_name is None:
                 man_name = 'bob' # Fallback assumption based on example

        man_location = None
        available_usable_spanners = {} # {spanner_name: location}
        nut_locations = {} # {nut_name: location}

        # Second pass to get locations and available spanners
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                if obj == man_name:
                    man_location = loc
                elif obj in self.goal_nuts: # It's a nut
                    nut_locations[obj] = loc
                else: # Assume it's a spanner if not man or nut
                    spanner_name = obj
                    is_usable = f"(usable {spanner_name})" in state
                    is_carried = (carried_spanner is not None and carried_spanner == spanner_name) # Check against the spanner found in the first pass
                    if is_usable and not is_carried:
                        available_usable_spanners[spanner_name] = loc

        # Ensure we found man's location
        if man_location is None:
             # This state is likely invalid or doesn't match expected structure
             return float('inf') # Cannot compute heuristic

        # Ensure we found locations for all loose goal nuts
        for nut in loose_goal_nuts:
             if nut not in nut_locations:
                 # This nut is loose and a goal, but its location is unknown? Invalid state.
                 return float('inf')

        # 3-10. Prepare and perform sequential calculation
        total_cost = 0
        current_man_loc = man_location
        # man_has_usable_spanner is already set based on the initial state

        # Create a mutable copy of available spanners
        current_available_usable_spanners = dict(available_usable_spanners)

        # Sort loose nuts by distance from the man's initial location
        # Need to handle cases where locations are not linked (distance is inf)
        # Use a safe distance lookup for sorting key
        nuts_to_tighten_sorted = sorted(
            loose_goal_nuts,
            key=lambda nut: self.get_distance(man_location, nut_locations.get(nut, None)) # Use .get for safety
        )

        for nut in nuts_to_tighten_sorted:
            nut_loc = nut_locations.get(nut) # Use .get for safety

            # Should not happen if previous check passed, but for safety
            if nut_loc is None:
                 return float('inf')

            # Cost to get man to the nut location
            dist_to_nut = self.get_distance(current_man_loc, nut_loc)
            if dist_to_nut == float('inf'):
                 return float('inf') # Cannot reach the nut

            total_cost += dist_to_nut
            current_man_loc = nut_loc

            # Cost to get a usable spanner if needed
            if not man_has_usable_spanner:
                if not current_available_usable_spanners:
                    return float('inf') # No usable spanners left

                # Find the closest available usable spanner
                closest_spanner = None
                min_dist_to_spanner = float('inf')
                spanner_loc = None

                # Iterate over a copy of items to allow deletion during iteration
                for s, s_loc in list(current_available_usable_spanners.items()):
                    dist = self.get_distance(current_man_loc, s_loc)
                    if dist < min_dist_to_spanner:
                        min_dist_to_spanner = dist
                        closest_spanner = s
                        spanner_loc = s_loc

                if closest_spanner is None or min_dist_to_spanner == float('inf'):
                     return float('inf') # Cannot reach any usable spanner

                total_cost += min_dist_to_spanner # Move man to spanner location
                current_man_loc = spanner_loc

                total_cost += 1 # Pickup spanner action
                man_has_usable_spanner = True
                del current_available_usable_spanners[closest_spanner] # This spanner is now "used" in the sequence

            # Cost to tighten the nut
            total_cost += 1
            man_has_usable_spanner = False # The spanner used became unusable

        return total_cost
