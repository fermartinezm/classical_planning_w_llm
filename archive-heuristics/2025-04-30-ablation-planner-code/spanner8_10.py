from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is provided elsewhere
# from heuristics.heuristic_base import Heuristic

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs_shortest_paths(locations, links):
    """
    Computes shortest path distances between all pairs of locations.
    locations: set of all location names (strings)
    links: set of (loc1, loc2) tuples representing bidirectional links
    Returns: dict mapping (loc1, loc2) -> distance. Returns float('inf') for unreachable pairs.
    """
    distances = {}
    adj = {loc: set() for loc in locations}
    for l1, l2 in links:
        adj[l1].add(l2)
        adj[l2].add(l1) # Assuming links are bidirectional

    for start_node in locations:
        q = deque([(start_node, 0)])
        visited = {start_node}
        distances[(start_node, start_node)] = 0

        while q:
            current_loc, dist = q.popleft()

            for neighbor in adj[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[(start_node, neighbor)] = dist + 1
                    q.append((neighbor, dist + 1))

    # Fill in unreachable pairs with infinity
    for l1 in locations:
        for l2 in locations:
            if (l1, l2) not in distances:
                distances[(l1, l2)] = float('inf')

    return distances

# Define the heuristic class
class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the number of nuts remaining, the number of spanners that need to be picked up,
    and the travel distance required to reach the locations of the loose nuts.

    # Assumptions:
    - The man can carry multiple spanners simultaneously.
    - Solvable instances have enough usable spanners available in total (carried or at locations)
      to tighten all goal nuts.
    - Links between locations are bidirectional.

    # Heuristic Initialization
    - Identify all locations, the man, all nuts, and all spanners from the task definition.
    - Identify the set of nuts that need to be tightened (goal nuts).
    - Parse the static `link` facts to build the graph of locations.
    - Precompute all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify all nuts that are currently `loose` and are part of the goal. These are the nuts that still need tightening.
    3. If there are no loose goal nuts, the state is a goal state, and the heuristic value is 0.
    4. Identify all usable spanners the man is currently `carrying`.
    5. Identify all usable spanners located on the ground (`at` a location).
    6. Check if the total number of available usable spanners (carried + at locations) is less than the number of loose goal nuts. If so, the problem is unsolvable from this state, return infinity.
    7. Calculate the number of additional spanners the man needs to pick up: This is the number of loose goal nuts minus the number of usable spanners he is currently carrying, capped at zero if he already carries enough or more.
    8. Identify the set of *distinct* locations where loose goal nuts are situated.
    9. Calculate the sum of shortest path distances from the man's current location to each distinct loose goal nut location. If any loose goal nut location is unreachable, the state is unsolvable, return infinity.
    10. The heuristic value is the sum of:
        - The number of loose goal nuts (representing the `tighten_nut` actions).
        - The number of additional spanners needed (representing the `pickup_spanner` actions).
        - The sum of distances calculated in step 9 (representing a lower bound / estimate of travel cost to reach the nuts).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information and precomputing distances."""
        # Parse objects to get names by type
        objects_by_type = {}
        # task.objects is a list of strings like "obj_name - type_name"
        for obj_str in task.objects:
            parts = obj_str.strip().split(" - ")
            if len(parts) == 2:
                name, type = parts
                if type not in objects_by_type:
                    objects_by_type[type] = []
                objects_by_type[type].append(name)
            # Note: Assuming standard PDDL object definition format

        self.locations = objects_by_type.get("location", [])
        # Assuming exactly one man object
        self.man = objects_by_type.get("man", [])[0] if objects_by_type.get("man") else None
        self.all_nuts = objects_by_type.get("nut", [])
        self.all_spanners = objects_by_type.get("spanner", [])

        # Parse links from static facts and build adjacency list
        links = set()
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                # Ensure locations are in our list of known locations
                if loc1 in self.locations and loc2 in self.locations:
                    links.add((loc1, loc2))
                # Note: Ignoring links involving unknown locations

        # Precompute all-pairs shortest paths
        self.distances = bfs_shortest_paths(self.locations, links)

        # Identify goal nuts (nuts that must be tightened)
        # Goal facts are strings like "(tightened nut1)"
        self.goal_nuts = {get_parts(goal)[1] for goal in task.goals if match(goal, "tightened", "*")}
        # Filter goal nuts to only include actual nut objects defined in the problem
        self.goal_nuts = {nut for nut in self.goal_nuts if nut in self.all_nuts}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", self.man, "*"):
                man_location = get_parts(fact)[2]
                break
        # If man_location is None, the state is likely invalid or unreachable
        if man_location is None:
             return float('inf')

        # 2. Identify loose nuts that are also goal nuts
        # We only care about nuts that are loose AND need to be tightened (are in goal_nuts)
        # If a nut is loose but not in goal_nuts, we don't need to tighten it.
        # If a nut is in goal_nuts but not loose, it must already be tightened.
        loose_goal_nuts = {nut for nut in self.goal_nuts if f"(loose {nut})" in state}

        # 3. If no loose goal nuts, goal achieved
        if len(loose_goal_nuts) == 0:
            return 0

        # 4. Identify carried usable spanners
        carried_usable_spanners = {
            get_parts(fact)[2] for fact in state
            if match(fact, "carrying", self.man, "*") and f"(usable {get_parts(fact)[2]})" in state
        }

        # 5. Identify usable spanners at locations
        usable_spanners_at_locs = {
             get_parts(fact)[1] for fact in state
             if match(fact, "at", "*", "*") and get_parts(fact)[1] in self.all_spanners and f"(usable {get_parts(fact)[1]})" in state
        }

        # 6. Check for unsolvable state (not enough usable spanners in total)
        total_usable_spanners = len(carried_usable_spanners) + len(usable_spanners_at_locs)
        if len(loose_goal_nuts) > total_usable_spanners:
             return float('inf') # Unsolvable

        # 7. Calculate needed pickups
        needed_pickups = max(0, len(loose_goal_nuts) - len(carried_usable_spanners))

        # 8. Identify distinct locations of loose goal nuts
        loose_goal_nut_locations = set()
        for nut in loose_goal_nuts:
             # Find the current location of this loose nut
             nut_location = None
             for fact in state:
                  if match(fact, "at", nut, "*"):
                       nut_location = get_parts(fact)[2]
                       break
             if nut_location: # Add location if found
                  loose_goal_nut_locations.add(nut_location)
             # Note: Assuming all loose goal nuts have a location in the state

        # 9. Calculate sum of distances from man to distinct loose goal nut locations
        sum_dist_to_nuts = 0
        for loc in loose_goal_nut_locations:
             # Check if distance was computed (i.e., location is reachable from man's current location)
             if (man_location, loc) not in self.distances or self.distances[(man_location, loc)] == float('inf'):
                  # A loose nut is in an unreachable location
                  return float('inf') # Unsolvable
             sum_dist_to_nuts += self.distances[(man_location, loc)]

        # 10. Calculate heuristic value
        # h = (tighten actions) + (pickup actions) + (travel cost to nuts)
        heuristic_value = len(loose_goal_nuts) + needed_pickups + sum_dist_to_nuts

        return heuristic_value
