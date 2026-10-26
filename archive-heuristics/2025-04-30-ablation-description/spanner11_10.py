from collections import deque
import math

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at bob shed)" -> ["at", "bob", "shed"]
    return fact[1:-1].split()

def build_location_graph(static_facts):
    """Builds an adjacency list representation of the location graph from 'link' facts."""
    graph = {}
    for fact in static_facts:
        parts = get_parts(fact)
        if parts[0] == 'link':
            loc1, loc2 = parts[1], parts[2]
            graph.setdefault(loc1, set()).add(loc2)
            graph.setdefault(loc2, set()).add(loc1) # Links are bidirectional
    return graph

def get_all_locations(static_facts, initial_state):
    """Collects all unique objects that appear as arguments in link or at predicates, assuming they are locations."""
    locations = set()
    for fact in static_facts:
        parts = get_parts(fact)
        if parts[0] == 'link':
            locations.add(parts[1])
            locations.add(parts[2])
        elif parts[0] == 'at':
            if len(parts) == 3: # Expecting (at obj loc)
                locations.add(parts[2])
    for fact in initial_state:
        parts = get_parts(fact)
        if parts[0] == 'at':
            if len(parts) == 3: # Expecting (at obj loc)
                locations.add(parts[2])
    return list(locations)

def compute_all_pairs_shortest_paths(graph, all_locations):
    """Computes shortest path distances between all pairs of locations using BFS."""
    distances = {}
    for start_node in all_locations:
        # Perform BFS from each start node
        queue = deque([(start_node, 0)])
        visited = {start_node}
        distances[(start_node, start_node)] = 0

        while queue:
            current_loc, dist = queue.popleft()

            # Only explore neighbors if the current location has links defined
            if current_loc in graph:
                for neighbor in graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[(start_node, neighbor)] = dist + 1
                        queue.append((neighbor, dist + 1))

    # Fill in unreachable pairs with infinity for all pairs in all_locations
    for loc1 in all_locations:
        for loc2 in all_locations:
            if (loc1, loc2) not in distances:
                distances[(loc1, loc2)] = math.inf

    return distances

# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# If the Heuristic base class is not provided, use a standalone class definition:
class spannerHeuristic: # Change to `class spannerHeuristic(Heuristic):` if inheriting
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the minimum number of actions required to tighten
    all goal nuts. It sums three components: the number of loose goal nuts
    (representing tighten actions), a cost for picking up a spanner if the man
    doesn't have one, and the travel cost to reach the nearest loose goal nut.

    # Assumptions
    - The man needs a usable spanner for each nut he tightens. Once used, a spanner becomes unusable.
    - The man can only carry one spanner at a time (implied by action effects).
    - The cost of walking between linked locations is 1.
    - The cost of picking up a spanner is 1.
    - The cost of tightening a nut is 1.
    - The heuristic provides an admissible estimate by summing lower bounds:
      - Number of tighten actions = number of loose goal nuts.
      - Spanner pickup cost = 1 if man needs a spanner, 0 otherwise.
      - Travel cost = shortest path distance to the nearest loose goal nut location.
      This sum is a lower bound on the total actions needed.
    - The heuristic assumes the man object's name does not start with 'spanner' or 'nut'
      to identify the man object from state facts. A more robust implementation
      would use type information from the PDDL problem instance.

    # Heuristic Initialization
    - Extracts the set of nuts that need to be tightened from the goal conditions.
    - Identifies all potential location objects mentioned in static facts or the initial state.
    - Builds a graph of locations based on `link` predicates from static facts.
    - Computes all-pairs shortest paths between all identified locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all nuts that are currently loose in the state and are part of the goal state.
    2. If there are no loose goal nuts, the goal is reached, and the heuristic is 0.
    3. Find the man's current location by iterating through 'at' facts and identifying the object that is not a spanner or nut (based on naming convention).
    4. Determine if the man is currently carrying a usable spanner by checking 'carrying' and 'usable' facts.
    5. Initialize the heuristic cost with the number of loose goal nuts (this is a lower bound on the number of `tighten_nut` actions).
    6. Calculate the cost associated with obtaining a spanner if the man doesn't have one:
       - If the man is not carrying a usable spanner:
         - Find all locations where a usable spanner is currently located.
         - If no usable spanners exist anywhere, the problem is unsolvable; return infinity.
         - Calculate the shortest distance from the man's current location to the nearest location with a usable spanner.
         - Add this minimum distance plus 1 (for the `pickup_spanner` action) to the total cost.
    7. Find the locations of all loose goal nuts in the current state.
    8. Calculate the shortest distance from the man's current location to the nearest location containing a loose goal nut.
    9. Add this minimum distance to the total cost (this is a lower bound on the travel needed to reach the first nut that needs tightening).
    10. Return the total calculated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Need initial state to find all locations

        # Extract goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == 'tightened'}

        # Identify all locations mentioned in static facts or initial state
        all_locations = get_all_locations(self.static, self.initial_state)

        # Build location graph from static facts
        self.location_graph = build_location_graph(self.static)

        # Compute all-pairs shortest paths
        self.shortest_paths = compute_all_pairs_shortest_paths(self.location_graph, all_locations)


    def __call__(self, node):
        """Estimate the minimum cost to tighten all remaining goal nuts."""
        state = node.state

        # 1. Identify loose goal nuts in the current state
        loose_goal_nuts_in_state = {n for n in self.goal_nuts if f'(loose {n})' in state}
        num_loose_goal_nuts = len(loose_goal_nuts_in_state)

        # 2. If no loose goal nuts, goal is reached
        if num_loose_goal_nuts == 0:
            return 0

        # 3. Find man's current location
        man_name = None
        man_location = None
        # Assuming the man object's name does not start with 'spanner' or 'nut'.
        # A more robust implementation would use type information from the PDDL problem.
        potential_men = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj_name = parts[1]
                if not obj_name.startswith('spanner') and not obj_name.startswith('nut'):
                    potential_men.add(obj_name)
            elif parts[0] == 'carrying':
                 potential_men.add(parts[1])

        if not potential_men:
            # Should not happen in a valid spanner problem
            return math.inf # Cannot find the man object

        # Assuming there's only one man, pick the first one found
        man_name = next(iter(potential_men))

        # Find the man's current location based on the identified man_name
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] == man_name:
                man_location = parts[2]
                break
        if man_location is None:
             # Man is not at any location? Problem state is invalid.
             return math.inf


        # 4. Determine if man is carrying a usable spanner
        man_carrying_usable_spanner = False
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'carrying' and parts[1] == man_name:
                spanner_name = parts[2]
                if f'(usable {spanner_name})' in state:
                    man_carrying_usable_spanner = True
                    break

        # 5. Calculate base cost (tighten actions)
        cost = num_loose_goal_nuts

        # 6. Calculate spanner pickup cost if needed
        if not man_carrying_usable_spanner:
            usable_spanner_locations = set()
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == 'at' and parts[1].startswith('spanner'):
                    spanner_name = parts[1]
                    location = parts[2]
                    if f'(usable {spanner_name})' in state:
                        usable_spanner_locations.add(location)

            if not usable_spanner_locations:
                # No usable spanners available anywhere
                return math.inf

            min_dist_to_spanner = math.inf
            for loc in usable_spanner_locations:
                # Ensure location is in our computed paths
                if (man_location, loc) in self.shortest_paths:
                    dist = self.shortest_paths[(man_location, loc)]
                    min_dist_to_spanner = min(min_dist_to_spanner, dist)
                # else: location might be isolated or not a location in the graph, treat as unreachable

            if min_dist_to_spanner == math.inf:
                # Cannot reach any usable spanner location from man's current location
                return math.inf

            cost += min_dist_to_spanner + 1 # Travel to nearest spanner + pickup action

        # 7. Find the locations of all loose goal nuts
        loose_nut_locations = set()
        for nut in loose_goal_nuts_in_state:
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == 'at' and parts[1] == nut:
                    loose_nut_locations.add(parts[2])
                    break # Found location for this nut

        # This set should not be empty if num_loose_goal_nuts > 0 and state is valid.
        # If it were empty, it would mean a loose goal nut is not 'at' any location, which is invalid.

        # 8. Calculate travel cost to reach the nearest loose goal nut
        min_dist_to_nut = math.inf
        for loc in loose_nut_locations:
             # Ensure location is in our computed paths
             if (man_location, loc) in self.shortest_paths:
                 dist = self.shortest_paths[(man_location, loc)]
                 min_dist_to_nut = min(min_dist_to_nut, dist)
             # else: location might be isolated or not a location in the graph, treat as unreachable

        if min_dist_to_nut == math.inf:
            # Cannot reach any loose goal nut location from man's current location
            return math.inf

        cost += min_dist_to_nut # Travel to the nearest loose goal nut

        return cost
