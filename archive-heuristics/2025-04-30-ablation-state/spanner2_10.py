from fnmatch import fnmatch
from collections import deque

# Assume Heuristic base class is available from the planning framework
# from heuristics.heuristic_base import Heuristic

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

def build_location_graph(static_facts):
    """Builds an adjacency list graph from link facts."""
    graph = {}
    locations = set()
    for fact in static_facts:
        if match(fact, "link", "*", "*"):
            _, loc1, loc2 = get_parts(fact)
            locations.add(loc1)
            locations.add(loc2)
            graph.setdefault(loc1, []).append(loc2)
            graph.setdefault(loc2, []).append(loc1) # Links are bidirectional
    return graph, locations

def bfs(graph, start_node, all_locations):
    """Computes shortest distances from start_node to all reachable nodes."""
    distances = {loc: float('inf') for loc in all_locations}

    # Ensure start_node is in the distances map and graph structure
    if start_node not in distances:
         distances[start_node] = float('inf')
    if start_node not in graph:
         graph[start_node] = [] # Add start_node to graph if isolated

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_loc = queue.popleft()

        # Ensure current_loc is in graph keys before iterating neighbors
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if distances.get(neighbor, float('inf')) == float('inf'): # Use .get for safety
                    distances[neighbor] = distances[current_loc] + 1
                    queue.append(neighbor)

    return distances


class spannerHeuristic: # Inherit from Heuristic in actual use
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all goal nuts.
    It sums the number of tighten actions, the number of spanner pickup actions
    required, and an estimate of the movement cost to reach the necessary locations
    (nut locations and spanner pickup locations).

    # Assumptions:
    - There is only one man.
    - Spanners become unusable after one use.
    - Links between locations are bidirectional.
    - The goal is to tighten a specific set of nuts.
    - Object names follow simple conventions (e.g., contain "man", "spanner", "nut").

    # Heuristic Initialization
    - Builds the location graph from static link facts.
    - Identifies all possible locations mentioned in static facts.
    - Stores the set of goal facts to identify goal nuts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the set of goal nuts that are currently in a 'loose' state. If this set is empty, the goal is reached for these nuts, and the heuristic is 0.
    2. Extract the current location of the man.
    3. Determine if the man is currently carrying a spanner, and if so, whether that spanner is 'usable'.
    4. Identify all 'usable' spanners that are currently located on the ground (not being carried) and their locations.
    5. Calculate the cost component for 'tighten_nut' actions: This is simply the number of loose goal nuts, as each requires one such action.
    6. Calculate the cost component for 'pickup_spanner' actions: To tighten `N` nuts, the man needs `N` usable spanners throughout the plan. If the man is already carrying a usable spanner, he needs to pick up `N-1` additional spanners. If he is not carrying a usable spanner, he needs to pick up `N` spanners. This count cannot be less than zero.
    7. Calculate the cost component for 'walk' actions (movement): The man needs to visit the location of each loose goal nut to tighten it. He also needs to visit the location of each spanner he needs to pick up.
       - Identify the locations of all loose goal nuts.
       - Determine which usable spanners need to be picked up (the `pickups_needed` closest usable spanners that are on the ground). Identify their locations.
       - The set of required locations to visit includes all loose goal nut locations and the locations of the spanners to be picked up.
       - Compute the shortest path distances from the man's current location to all other locations using Breadth-First Search (BFS) on the location graph.
       - Estimate the movement cost as the sum of the shortest path distances from the man's current location to each of the required locations to visit. If any required location is unreachable, the state is likely unsolvable, and a very high heuristic value is returned.
    8. Sum the individual cost components (tighten, pickup, movement) to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic."""
        self.goals = task.goals
        self.static_facts = task.static

        # Build the location graph and get all locations from static facts
        self.location_graph, self.all_locations = build_location_graph(self.static_facts)

        # Identify goal nuts from the goal conditions
        self.goal_nuts = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened":
                nut = args[0]
                self.goal_nuts.add(nut)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # 1. Identify loose goal nuts
        loose_goal_nuts = {
            nut for nut in self.goal_nuts
            if f"(loose {nut})" in state
        }

        num_loose_goal_nuts = len(loose_goal_nuts)

        # If all goal nuts are tightened, the heuristic is 0.
        if num_loose_goal_nuts == 0:
            return 0

        # Extract relevant state information
        man_obj = None
        man_loc = None
        man_carrying_spanner = None # Store the spanner object if carried
        spanner_locations = {} # Map spanner object to its location
        spanner_usable_status = {} # Map spanner object to usable status (True/False)
        nut_locations = {} # Map nut object to its location
        all_spanners = set() # Set of all spanner objects
        all_nuts = set() # Set of all nut objects

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                # Simple check based on name convention
                if "man" in obj:
                     man_obj = obj
                     man_loc = loc
                elif "spanner" in obj:
                     all_spanners.add(obj)
                     spanner_locations[obj] = loc
                elif "nut" in obj:
                     all_nuts.add(obj)
                     nut_locations[obj] = loc
            elif parts[0] == "carrying":
                 carrier, spanner = parts[1], parts[2]
                 # Assuming the carrier is the man object identified earlier
                 # (or if only one man, this must be him)
                 man_carrying_spanner = spanner
                 all_spanners.add(spanner) # Ensure spanner is tracked
            elif parts[0] == "usable":
                 spanner = parts[1]
                 spanner_usable_status[spanner] = True
                 all_spanners.add(spanner) # Ensure spanner is tracked
            elif parts[0] == "loose":
                 nut = parts[1]
                 all_nuts.add(nut) # Ensure nut is tracked

        # If man object wasn't found (invalid state?), return high cost
        if man_obj is None or man_loc is None:
             return 1000000

        # Ensure all spanners mentioned in state are in our tracking dicts
        for spanner in all_spanners:
             spanner_usable_status.setdefault(spanner, False) # Default to not usable if not explicitly stated

        # Check if man is carrying a usable spanner
        man_carrying_usable = (man_carrying_spanner is not None and
                               spanner_usable_status.get(man_carrying_spanner, False))


        # Find usable spanners not carried by the man
        usable_spanners_not_carried = {
            s for s in all_spanners
            if s != man_carrying_spanner and spanner_usable_status.get(s, False)
        }
        usable_spanner_locations = {
            spanner_locations[s] for s in usable_spanners_not_carried if s in spanner_locations
        }

        # 5. Cost for tightening actions
        tighten_cost = num_loose_goal_nuts

        # 6. Cost for pickup actions
        pickups_needed = max(0, num_loose_goal_nuts - (1 if man_carrying_usable else 0))
        pickup_cost = pickups_needed

        # 7. Cost for movement
        # Locations of loose goal nuts
        loose_nut_locations = {nut_locations[nut] for nut in loose_goal_nuts if nut in nut_locations}

        # Compute distances from man's current location to all other locations
        # Ensure man_loc is included in the graph structure for BFS if it's an isolated location
        if man_loc not in self.location_graph:
             self.location_graph[man_loc] = []
             self.all_locations.add(man_loc)

        distances_from_man = bfs(self.location_graph, man_loc, self.all_locations)

        # Identify locations of spanners to be picked up
        spanner_pickup_locations = set()
        if pickups_needed > 0:
            # Find the `pickups_needed` closest usable spanner locations from man_loc
            # Filter reachable usable spanners and sort by distance
            reachable_usable_spanner_locations = {
                 sloc for sloc in usable_spanner_locations
                 if distances_from_man.get(sloc, float('inf')) != float('inf')
            }

            if len(reachable_usable_spanner_locations) < pickups_needed:
                 # Not enough reachable usable spanners for the required nuts
                 # This state is likely unsolvable.
                 return 1000000 # Large number

            dist_to_spanners = [(distances_from_man[sloc], sloc) for sloc in reachable_usable_spanner_locations]
            dist_to_spanners.sort()
            spanner_pickup_locations = {sloc for d, sloc in dist_to_spanners[:pickups_needed]}

        # The set of all locations the man needs to visit (nuts + spanners to pick up)
        locations_to_visit = loose_nut_locations.union(spanner_pickup_locations)

        # Estimate movement cost as sum of distances from man's location to each required location
        movement_cost = 0
        unreachable_target = False
        for loc in locations_to_visit:
            dist = distances_from_man.get(loc, float('inf'))
            if dist == float('inf'):
                 unreachable_target = True
                 break # If any required location is unreachable
            movement_cost += dist

        # If any required location is unreachable, return a very high heuristic value
        if unreachable_target:
             return 1000000 # Large number indicating likely unsolvable state


        # 8. Total heuristic value
        total_cost = tighten_cost + pickup_cost + movement_cost

        return total_cost

# Note: This code assumes the existence of a base class `Heuristic`
# and a `Task` object with `goals`, `static`, and `state` attributes
# as shown in the example code files.
