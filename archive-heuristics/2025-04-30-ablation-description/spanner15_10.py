from fnmatch import fnmatch
# Assume Heuristic base class is available from 'heuristics.heuristic_base'
# from heuristics.heuristic_base import Heuristic

# Helper function to get parts of a fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match facts
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS for shortest path
def bfs_shortest_path(start_loc, end_loc, links, all_locations):
    """
    Computes the shortest path distance between two locations using BFS.

    Args:
        start_loc: The starting location.
        end_loc: The target location.
        links: A set of tuples representing bidirectional links between locations (e.g., {('locA', 'locB')}).
        all_locations: A set of all known locations in the domain.

    Returns:
        The shortest distance (number of walk actions) or float('inf') if unreachable.
    """
    if start_loc == end_loc:
        return 0
    
    # Build adjacency list from links
    adj = {}
    for l1, l2 in links:
        adj.setdefault(l1, set()).add(l2)
        adj.setdefault(l2, set()).add(l1) # Links are bidirectional

    queue = [(start_loc, 0)]
    visited = {start_loc}
    
    while queue:
        current_loc, dist = queue.pop(0)
        
        if current_loc == end_loc:
            return dist
        
        # Get neighbors from adjacency list
        neighbors = adj.get(current_loc, set())

        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
                
    # If end_loc is in all_locations but unreachable from start_loc
    # or if end_loc is not a known location
    return float('inf')


# Precompute all pairs shortest paths
def precompute_shortest_paths(links, all_locations):
    """
    Precomputes shortest path distances between all pairs of locations.

    Args:
        links: A set of tuples representing bidirectional links.
        all_locations: A set of all known locations.

    Returns:
        A dictionary where keys are (start_loc, end_loc) tuples and values are distances.
    """
    paths = {}
    for start_loc in all_locations:
        for end_loc in all_locations:
            paths[(start_loc, end_loc)] = bfs_shortest_path(start_loc, end_loc, links, all_locations)
    return paths


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the cost to tighten all goal nuts. It sums the number
    of required actions (tighten and pickup) and adds the estimated travel cost
    to reach the nearest location where a necessary action (picking up a spanner
    or tightening a nut) can be performed.

    # Assumptions
    - Links between locations are bidirectional.
    - The problem is solvable, meaning all required locations (loose nuts, usable spanners)
      are reachable from the man's initial location.
    - There is only one man object in the domain.
    - Objects involved in 'usable', 'loose', 'tightened', 'carrying' predicates are spanners or nuts.
    - The first argument of an 'at' predicate that is not a spanner or nut is the man.

    # Heuristic Initialization
    - Extracts all unique locations mentioned in 'link' and initial 'at' facts.
    - Builds the location graph based on 'link' facts.
    - Precomputes shortest path distances between all pairs of extracted locations using BFS.
    - Identifies the set of nuts that must be tightened according to the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Identify all loose nuts in the current state and their locations.
    3. Identify all usable spanners at locations in the current state and their locations.
    4. Determine if the man is currently carrying a usable spanner.
    5. Filter the loose nuts to include only those that are part of the problem's goal.
    6. If there are no loose goal nuts, the heuristic value is 0 (goal state reached).
    7. Calculate the base cost:
       - Add 1 for each loose goal nut (representing the 'tighten_nut' action).
       - Calculate the number of 'pickup_spanner' actions needed: This is the number of loose goal nuts minus 1 if the man is already carrying a usable spanner (since the first spanner is already held), minimum 0. Add this count to the base cost.
    8. Identify the set of "required locations":
       - This set includes the location of every loose goal nut.
       - If pickup actions are needed (i.e., the number of loose goal nuts is greater than the number of usable spanners currently held), add the locations of all usable spanners found on the ground to this set.
    9. Calculate the travel cost:
       - Find the minimum shortest path distance from the man's current location to any location in the set of required locations.
       - If no required locations are reachable (minimum distance is infinity), the problem is likely unsolvable from this state, return infinity.
    10. The total heuristic value is the base cost (tighten + pickup actions) plus the travel cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and building the location graph."""
        # Access initial_state from task to get all locations mentioned in 'at' facts
        # Assume task object has initial_state attribute
        self.initial_state = task.initial_state # Needed for location extraction
        self.goals = task.goals
        self.static = task.static

        # Extract all unique locations and links
        self.all_locations = set()
        self.links = set()

        # Collect locations from static links
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                self.links.add((l1, l2))
                self.all_locations.add(l1)
                self.all_locations.add(l2)

        # Collect locations from initial 'at' facts
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                 # The second argument of 'at' is the location
                 loc = parts[2]
                 self.all_locations.add(loc)

        # Precompute shortest paths between all pairs of extracted locations
        self.shortest_paths = precompute_shortest_paths(self.links, self.all_locations)

        # Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'tightened':
                self.goal_nuts.add(parts[1])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions and travel."""
        state = node.state

        # 1. Extract relevant information from the current state
        man_name = None
        man_loc = None
        loose_nuts = {} # {nut: location}
        usable_spanners_at_loc = {} # {spanner: location}
        man_carrying_usable_spanner = False

        # First pass to identify spanners and nuts
        spanners_in_state = set()
        nuts_in_state = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'usable': spanners_in_state.add(parts[1])
            elif parts[0] == 'loose': nuts_in_state.add(parts[1])
            elif parts[0] == 'tightened': nuts_in_state.add(parts[1])
            elif parts[0] == 'carrying': spanners_in_state.add(parts[2]) # Spanner being carried

        # Second pass to get locations and man's status
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                # Identify the man: a locatable that is not a spanner or nut
                if obj not in spanners_in_state and obj not in nuts_in_state:
                     man_name = obj
                     man_loc = loc

                # Identify loose nuts and their locations
                if obj in nuts_in_state and f'(loose {obj})' in state:
                     loose_nuts[obj] = loc

                # Identify usable spanners at locations
                if obj in spanners_in_state and f'(usable {obj})' in state:
                     usable_spanners_at_loc[obj] = loc

            elif parts[0] == 'carrying':
                 carrier, spanner = parts[1], parts[2]
                 # Assuming the carrier is the man based on domain definition
                 man_name = carrier
                 if f'(usable {spanner})' in state:
                     man_carrying_usable_spanner = True

        # Ensure man_loc was found (should be in initial state for a valid problem)
        if man_loc is None:
             # Cannot compute heuristic without man's location
             return float('inf')


        # 2. Identify loose nuts that are goals
        goal_loose_nuts = {nut: loc for nut, loc in loose_nuts.items() if nut in self.goal_nuts}

        # If all goal nuts are tightened, heuristic is 0
        if not goal_loose_nuts:
            return 0

        # 3. Calculate base cost (tighten actions + pickup actions)
        num_loose_goal_nuts = len(goal_loose_nuts)

        # Number of pickup actions needed is the number of additional spanners required
        # beyond the first one if the man is already carrying one.
        # He needs num_loose_goal_nuts spanners in total.
        # He has (1 if man_carrying_usable_spanner else 0) spanners available without pickup.
        # Pickups needed = max(0, num_loose_goal_nuts - (1 if man_carrying_usable_spanner else 0))
        num_pickups = max(0, num_loose_goal_nuts - (1 if man_carrying_usable_spanner else 0))

        base_cost = num_loose_goal_nuts # Cost for tighten actions (1 per nut)
        base_cost += num_pickups # Cost for pickup actions (1 per spanner needed)

        # 4. Calculate travel cost
        LooseNutLocations = set(goal_loose_nuts.values())
        UsableSpannerLocations = set(usable_spanners_at_loc.values())

        required_locations = set(LooseNutLocations)
        # Add spanner locations only if pickups are needed
        if num_pickups > 0:
             required_locations.update(UsableSpannerLocations)

        travel_cost = float('inf')
        if required_locations:
            min_dist = float('inf')
            for loc in required_locations:
                # Ensure the location is in our precomputed paths graph
                # and the man's current location is also in the graph
                if man_loc in self.all_locations and loc in self.all_locations:
                     min_dist = min(min_dist, self.shortest_paths.get((man_loc, loc), float('inf')))
                else:
                    # A required location or the man's location is not in the known graph, likely unsolvable
                    min_dist = float('inf')
                    break # No need to check other locations if one is unreachable

            travel_cost = min_dist

        # If the nearest required location is unreachable, the problem is unsolvable from this state
        if travel_cost == float('inf'):
            return float('inf')

        # 5. Total heuristic cost
        return base_cost + travel_cost
