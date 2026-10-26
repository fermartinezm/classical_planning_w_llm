from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def build_location_graph(static_facts):
    """Build an adjacency list representation of the location graph from 'link' facts."""
    graph = {}
    # Collect all locations mentioned in link facts to ensure all nodes are in the graph
    locations = set()
    for fact in static_facts:
         parts = get_parts(fact)
         if parts[0] == 'link':
             locations.add(parts[1])
             locations.add(parts[2])

    for loc in locations:
        graph[loc] = [] # Initialize all known locations

    for fact in static_facts:
        if match(fact, "link", "*", "*"):
            _, loc1, loc2 = get_parts(fact)
            # Add edges, ensuring locations exist in the graph dict
            if loc1 in graph and loc2 in graph:
                graph[loc1].append(loc2)
                graph[loc2].append(loc1) # Links are bidirectional
    return graph

def bfs_shortest_paths(graph, start_node):
    """Compute shortest path distances from start_node to all other nodes using BFS."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node is not in the graph (e.g., isolated location not in any link)
         # Distance to itself is 0, others remain inf.
         if start_node in distances: # Check if it was added during graph initialization
             distances[start_node] = 0
         return distances # Cannot reach anything if start is not in the navigable graph

    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Ensure current_node has neighbors in the graph dict
        if current_node in graph:
             for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

def compute_all_pairs_shortest_paths(graph):
    """Compute shortest path distances between all pairs of locations in the graph."""
    all_distances = {}
    # Compute distances from every node in the graph
    for start_node in graph:
        all_distances[start_node] = bfs_shortest_paths(graph, start_node)
    return all_distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It sums the number of remaining tighten actions, the minimum number of spanner
    pickup actions required, and an estimate of the travel cost to reach the
    locations where work needs to be done and tools acquired.

    # Assumptions
    - The goal is to tighten all nuts that were initially loose (or are currently loose
      if the goal is specified dynamically). This heuristic works based on the set
      of currently loose nuts.
    - A spanner becomes unusable after one tighten action.
    - The man can carry multiple spanners simultaneously.
    - Travel cost between linked locations is 1.
    - Shortest path distances between locations can be precomputed from static facts.
    - The problem is solvable (i.e., enough usable spanners exist in total).

    # Heuristic Initialization
    - Build the location graph from `link` predicates found in static facts.
    - Compute all-pairs shortest path distances between all locations in the graph.
    - Store the distance function for quick lookup.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  Identify the man's current location. If the man's location cannot be determined
        or is not in the navigable graph, the state is likely unsolvable, return infinity.
    2.  Identify all nuts that are currently loose and their locations.
    3.  If there are no loose nuts, the goal is effectively reached from this perspective,
        return 0. (Note: The heuristic checks `task.goal_reached(state)` first for a true 0 value).
    4.  Count the number of usable spanners the man is currently carrying.
    5.  Identify all usable spanners on the ground and their locations.
    6.  Calculate the number of additional usable spanners the man needs to pick up from
        the ground. This is the maximum of 0 and (number of loose nuts - number of
        usable spanners carried).
    7.  Check if enough usable spanners are available in total (carried + on ground)
        to tighten all loose nuts. If not, the state is unsolvable, return infinity.
    8.  Identify the set of locations the man *must* visit to make progress:
        - The location of every loose nut.
        - The locations of the required number of closest usable spanners on the ground
          that need to be picked up.
    9.  Estimate the travel cost: Compute the sum of shortest path distances from the
        man's current location to each location in the set of required visit locations.
        If any required location is unreachable, return infinity.
    10. Sum the costs:
        - Number of loose nuts (representing tighten actions).
        - Number of spanners to pick up (representing pickup actions).
        - Estimated travel cost (sum of distances calculated in step 9).

    This heuristic provides a lower bound on the number of tighten and pickup actions
    and adds a simple, non-admissible estimate for the necessary travel. It aims to
    guide the search towards states where fewer tasks remain and the required items/locations
    are closer.
    """

    def __init__(self, task):
        """Initialize the heuristic by precomputing location distances."""
        self.task = task # Store task for goal checking
        self.static_facts = task.static # Store static facts

        # Build the location graph from 'link' predicates
        self.location_graph = build_location_graph(self.static_facts)

        # Compute all-pairs shortest paths
        self.all_pairs_distances = compute_all_pairs_shortest_paths(self.location_graph)

        # Helper to get distance between two locations
        # Returns infinity if locations are not in the graph or unreachable
        self.get_distance = lambda l1, l2: self.all_pairs_distances.get(l1, {}).get(l2, float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 0. Check if goal is reached
        if self.task.goal_reached(state):
             return 0

        # 1. Identify the man's current location.
        man_location = None
        man_name = None # Need man's name to check 'carrying' predicate

        # Infer man's name and location from state facts
        potential_men = set()
        potential_spanners = set()
        potential_nuts = set()

        # First pass to identify potential types based on predicates
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'carrying':
                if len(parts) == 3: # (carrying ?m ?s)
                    potential_men.add(parts[1])
                    potential_spanners.add(parts[2])
            elif parts[0] == 'usable':
                 if len(parts) == 2: # (usable ?s)
                    potential_spanners.add(parts[1])
            elif parts[0] in ['loose', 'tightened']:
                 if len(parts) == 2: # (loose ?n), (tightened ?n)
                    potential_nuts.add(parts[1])

        # Second pass to find man's location, assuming the man is the locatable object
        # that is not a spanner or nut based on the types inferred above.
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj_name = get_parts(fact)[1]
                loc_name = get_parts(fact)[2]
                # If this object is not a known spanner or nut, assume it's the man
                if obj_name not in potential_spanners and obj_name not in potential_nuts:
                    man_name = obj_name
                    man_location = loc_name
                    break # Found the man and his location

        if man_location is None or man_location not in self.location_graph:
             # Man location not found or is in an isolated location not in the graph
             # This state is likely unsolvable via walking
             return float('inf')

        # 2. Identify all nuts that are currently loose and their locations.
        loose_nuts = [] # List of (nut_name, location)
        # Collect all nuts mentioned in state (loose or tightened) to find their locations
        all_nuts_in_state = set()
        for fact in state:
             if match(fact, "loose", "*"):
                 all_nuts_in_state.add(get_parts(fact)[1])
             elif match(fact, "tightened", "*"):
                 all_nuts_in_state.add(get_parts(fact)[1])

        # Find locations for all nuts identified
        nut_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj_name = get_parts(fact)[1]
                loc_name = get_parts(fact)[2]
                if obj_name in all_nuts_in_state:
                    nut_locations[obj_name] = loc_name

        # Filter for loose nuts and get their locations
        for fact in state:
             if match(fact, "loose", "*"):
                 nut_name = get_parts(fact)[1]
                 if nut_name in nut_locations:
                     loose_nuts.append((nut_name, nut_locations[nut_name]))
                 # else: loose nut location not found in state? Problematic state.

        num_loose_nuts = len(loose_nuts)

        # If no loose nuts, but goal not reached, there might be other goal conditions
        # or an issue with state representation vs goal definition.
        # Assuming goal is purely about tightening nuts, if num_loose_nuts is 0,
        # and goal_reached was false, something is wrong.
        # However, the initial goal check handles the 0 case correctly.
        # If we reach here, num_loose_nuts > 0 (unless the goal includes non-nut conditions,
        # which the example domain doesn't show).

        # 3. Count usable spanners carried by the man.
        carried_usable_spanners = [] # List of spanner_name
        carried_spanners = set()
        for fact in state:
             if match(fact, "carrying", man_name, "*"):
                 carried_spanners.add(get_parts(fact)[2])

        for spanner_name in carried_spanners:
             if f'(usable {spanner_name})' in state:
                 carried_usable_spanners.append(spanner_name)

        num_carried_usable = len(carried_usable_spanners)

        # 4. Identify usable spanners on the ground and their locations.
        ground_usable_spanners = [] # List of (spanner_name, location)
        all_spanners_in_state = potential_spanners # Use inferred potential spanners

        # Find locations for all spanners identified
        spanner_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj_name = get_parts(fact)[1]
                loc_name = get_parts(fact)[2]
                if obj_name in all_spanners_in_state:
                    spanner_locations[obj_name] = loc_name

        # Filter for usable spanners on the ground
        for spanner_name in all_spanners_in_state:
             if f'(usable {spanner_name})' in state:
                 # Check if this usable spanner is on the ground (not carried)
                 if spanner_name not in carried_spanners:
                     if spanner_name in spanner_locations:
                         ground_usable_spanners.append((spanner_name, spanner_locations[spanner_name]))
                     # else: usable spanner is not carried and not at a location? Problematic state.


        # 5. Calculate the number of additional spanners needed.
        num_pickups_needed = max(0, num_loose_nuts - num_carried_usable)

        # 6. Check solvability based on available spanners.
        if num_pickups_needed > len(ground_usable_spanners):
             # Not enough usable spanners exist in the state to tighten all loose nuts
             return float('inf')

        # 7. Identify required visit locations.
        required_nut_locations = {loc for nut, loc in loose_nuts}

        # Find the locations of the 'num_pickups_needed' closest usable spanners on the ground.
        # Sort ground usable spanners by distance from man's current location.
        # Handle cases where man_location might not be in the graph (already checked above, but defensive)
        if man_location not in self.all_pairs_distances:
             return float('inf') # Should be caught earlier, but double check

        ground_usable_spanners.sort(key=lambda item: self.get_distance(man_location, item[1]))

        required_pickup_locations = {loc for spanner, loc in ground_usable_spanners[:num_pickups_needed]}

        # Combine required locations
        required_visit_locations = required_nut_locations.union(required_pickup_locations)

        # 8. Estimate travel cost.
        # Simple sum of distances from man's current location to each required location.
        travel_cost = 0
        for loc in required_visit_locations:
             dist = self.get_distance(man_location, loc)
             if dist == float('inf'):
                 # If a required location is unreachable, the problem is unsolvable from this state.
                 return float('inf')
             travel_cost += dist

        # 9. Sum the costs.
        total_cost = num_loose_nuts # Cost for tighten actions (1 per nut)
        total_cost += num_pickups_needed # Cost for pickup actions (1 per pickup)
        total_cost += travel_cost # Estimated travel cost

        return total_cost
