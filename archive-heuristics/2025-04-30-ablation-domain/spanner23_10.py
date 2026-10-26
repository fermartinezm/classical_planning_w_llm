# Assuming Heuristic base class is available in the execution environment
# from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import deque

# Helper functions
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
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding argument (with fnmatch support)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Define the heuristic class
class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the number of nuts remaining, the cost to acquire a spanner if needed,
    and the travel cost to reach the nearest loose nut.

    # Assumptions
    - Bob is the only agent.
    - Bob can carry at most one spanner.
    - Links between locations are bidirectional.
    - Usable spanners are available either carried by Bob or on the ground in solvable states.
    - The cost of any action (move, pickup, tighten) is 1.

    # Heuristic Initialization
    - Build a graph representing locations and links from static facts.
    - Precompute all-pairs shortest path distances between locations using BFS.
    - Identify the set of goal nuts from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the goal is reached by verifying if all goal conditions are met in the state. If yes, the heuristic value is 0.
    2. Identify all goal nuts that are currently NOT tightened. These are the nuts that need work.
    3. The base heuristic value is the count of these nuts (representing the minimum number of 'tighten' actions required).
    4. Determine Bob's current location from the state. If Bob's location cannot be found, return infinity (indicating a potentially unsolvable state).
    5. Check if Bob is currently carrying a usable spanner.
    6. If Bob is NOT carrying a usable spanner:
       - Find the locations of all usable spanners that are currently on the ground.
       - If there are usable spanners on the ground:
         - Add 1 to the heuristic (representing the 'pickup' action cost).
         - Calculate the shortest distance from Bob's current location to the nearest location containing a usable spanner on the ground. Add this distance to the heuristic (representing the travel cost to acquire a spanner).
       - If there are no usable spanners on the ground (and Bob isn't carrying one), return infinity (indicating a potentially unsolvable state).
    7. Find the current locations of all the nuts identified in step 2 (nuts that need tightening). If any such nut's location cannot be found, return infinity.
    8. Calculate the shortest distance from Bob's current location to the nearest location among the nuts identified in step 2. Add this distance to the heuristic (representing the initial travel cost to reach the work area).
    9. The total heuristic value is the sum accumulated from steps 3, 6 (if applicable), and 8.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and computing distances.
        """
        # The base class constructor stores task.goals and task.static
        super().__init__(task)

        # Build the location graph from static link facts
        self.location_graph = {}
        locations = set()
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1) # Links are bidirectional

        self.locations = list(locations) # Store locations for easy iteration

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[(start_loc, start_loc)] = 0

            while q:
                current_loc, dist = q.popleft()

                if current_loc in self.location_graph:
                    for neighbor in self.location_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[(start_loc, neighbor)] = dist + 1
                            q.append((neighbor, dist + 1))

        # Identify goal nuts from the task goals
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state

        # 1. Check if goal is reached
        if self.goals <= state:
             return 0

        h = 0

        # 2. Identify nuts that need tightening (goal nuts not yet tightened)
        nuts_to_tighten = {nut for nut in self.goal_nuts if f"(tightened {nut})" not in state}

        # This set should be non-empty if self.goals <= state is False, assuming standard PDDL goals.
        # Add cost for tighten actions
        h += len(nuts_to_tighten)

        # 3. Find Bob's location
        bob_loc = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                bob_loc = get_parts(fact)[2]
                break
        if bob_loc is None:
             # Bob's location is unknown - problem state.
             return float('inf')

        # 4. Check if Bob is carrying a usable spanner
        bob_carrying_usable_spanner = False
        # No need to find which spanner, just if he has *a* usable one
        for fact in state:
            if match(fact, "carrying", "bob", "*"):
                 carried_spanner = get_parts(fact)[2]
                 # Check if the carried spanner is usable by looking it up in the state
                 if f"(usable {carried_spanner})" in state:
                      bob_carrying_usable_spanner = True
                 break # Assuming Bob carries at most one spanner

        # 5. Cost to get a spanner if needed
        if not bob_carrying_usable_spanner:
            # Find usable spanners on the ground
            usable_spanner_ground_locs = set()
            for fact in state:
                 if match(fact, "at", "*", "*"):
                      obj, loc = get_parts(fact)[1:3]
                      # Check if the object at this location is a usable spanner
                      if f"(usable {obj})" in state:
                           usable_spanner_ground_locs.add(loc)

            if usable_spanner_ground_locs:
                # Add cost for pickup action
                h += 1
                # Add travel cost to nearest usable spanner on the ground
                min_dist_to_spanner = float('inf')
                for loc in usable_spanner_ground_locs:
                     # Ensure the location is in our precomputed distances (i.e., reachable)
                     if (bob_loc, loc) in self.distances:
                         min_dist_to_spanner = min(min_dist_to_spanner, self.distances[(bob_loc, loc)])

                # If min_dist_to_spanner is still inf, it means no usable spanner on ground is reachable
                if min_dist_to_spanner == float('inf'):
                     return float('inf') # Unreachable spanner
                 # Add the travel cost
                h += min_dist_to_spanner
            else:
                 # No usable spanners on the ground, and Bob isn't carrying one. Unsolvable state?
                 # Assuming solvable implies usable spanners exist and are reachable.
                 # If none are on ground and Bob isn't carrying one, this state is likely unsolvable.
                 return float('inf') # No usable spanners available

        # 6. Cost to travel to the nearest loose goal nut
        loose_nut_locations = {}
        for nut in nuts_to_tighten:
             # Find location of this loose nut
             nut_loc = None
             for fact in state:
                  if match(fact, "at", nut, "*"):
                       nut_loc = get_parts(fact)[2]
                       break
             if nut_loc is None:
                  # Loose nut is not located anywhere? Problem state.
                  return float('inf')
             loose_nut_locations[nut] = nut_loc

        # Find the minimum distance from Bob's current location to any loose nut location
        if loose_nut_locations: # This should be true if nuts_to_tighten is non-empty
            min_dist_to_nut = float('inf')
            for nut, loc in loose_nut_locations.items():
                 # Ensure the location is in our precomputed distances (i.e., reachable)
                 if (bob_loc, loc) in self.distances:
                      min_dist_to_nut = min(min_dist_to_nut, self.distances[(bob_loc, loc)])

            # If min_dist_to_nut is still inf, it means no loose nut is reachable
            if min_dist_to_nut == float('inf'):
                 return float('inf') # Unreachable nut
            # Add the travel cost
            h += min_dist_to_nut
        # else: nuts_to_tighten was non-empty, but loose_nut_locations is empty?
        # This case should be caught by the nut_loc is None check inside the loop.

        return h
