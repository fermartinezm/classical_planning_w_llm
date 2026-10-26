from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Requirement 1: Ignore surrounding brackets.
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if pattern is longer than fact parts
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """Computes shortest path distances from start_node to all other nodes."""
    distances = {node: float('inf') for node in graph}
    if start_node not in graph:
         # Start node is not in the graph (e.g., object at an isolated location)
         return distances # All distances remain infinity

    distances[start_node] = 0
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()

        # Check if current_node is still in the graph (should be if distances[current_node] is not inf)
        if current_node in graph:
            for neighbor in graph.get(current_node, []):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

def build_distance_map(locations, links):
    """Builds a graph from links and computes all-pairs shortest paths."""
    graph = {loc: [] for loc in locations}
    for l1, l2 in links:
        # Ensure linked locations are actually in the provided locations set
        if l1 in graph and l2 in graph:
            graph[l1].append(l2)
            graph[l2].append(l1) # Links are bidirectional

    distance_map = {}
    for start_loc in locations:
        distance_map[start_loc] = bfs(graph, start_loc)
    return distance_map


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts specified in the goal. It calculates the minimum cost for each untightened goal nut independently and sums these costs. The cost for a single nut is estimated as the minimum actions needed to get the man to the nut's location with a usable spanner and then tighten it.

    # Assumptions
    - The goal is always to tighten a set of nuts.
    - Links between locations are bidirectional and static.
    - Spanners are consumed (become unusable) after one use.
    - There is only one man, assumed to be named 'bob' based on examples.
    - Nuts remain at their initial locations.
    - The problem is solvable, implying enough usable spanners exist somewhere to tighten all goal nuts and all relevant locations are connected.

    # Heuristic Initialization
    - Extracts the set of goal facts, specifically identifying which nuts need to be tightened.
    - Processes static facts to build the location graph based on `link` predicates.
    - Identifies all unique locations mentioned in static `link` facts and initial state `at` facts.
    - Computes all-pairs shortest path distances between all identified locations using BFS, storing them in a distance map.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the single man (assumed 'bob') and his current location.
    2. Identify all nuts that are currently `loose` and are part of the goal conditions. Determine their locations.
    3. Identify all spanners that are currently `usable` and their locations (either `at` a location or `carrying` by the man).
    4. Initialize the total heuristic cost to 0.
    5. For each loose nut `N` at location `L_N` that needs to be tightened (i.e., is a goal):
        a. Calculate the minimum cost to tighten this specific nut from the current state.
        b. Initialize `min_cost_for_nut = float('inf')`.
        c. Consider the option of using a spanner the man is currently carrying:
           - If the man is carrying a spanner `S` and `S` is `usable`:
             - The cost is the distance from the man's current location to `L_N` (walks) + 1 (tighten).
             - Update `min_cost_for_nut` with this value if it's smaller.
        d. Consider the option of picking up an available usable spanner:
           - Iterate through all usable spanners `S` that are `at` a location `L_S`:
             - The cost is the distance from the man's current location to `L_S` (walks) + 1 (pickup) + distance from `L_S` to `L_N` (walks) + 1 (tighten).
             - Update `min_cost_for_nut` with this value if it's smaller.
        e. If `min_cost_for_nut` is still infinity (no usable spanners available or reachable, or nut location unreachable), the state is likely a dead end. Return infinity immediately.
        f. Add `min_cost_for_nut` to the `total_heuristic`.
    6. Return the `total_heuristic`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # Requirement 5: Process static facts in the constructor.
        static_facts = task.static

        # Identify goal nuts from goal facts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'tightened' and len(parts) == 2:
                self.goal_nuts.add(parts[1])

        # Build location graph and compute distances
        locations = set()
        links = []

        # Find all locations and links from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                links.append((l1, l2))
                locations.add(l1)
                locations.add(l2)

        # Add locations mentioned in initial state 'at' facts to ensure all relevant locations are in the graph
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3:
                 # The third argument is the location
                 locations.add(parts[2])

        self.distance_map = build_distance_map(list(locations), links)

        # Assume the man's name is 'bob' based on examples
        # A more robust implementation would find the object of type 'man' from task.objects
        self.man_name = 'bob'


    def get_distance(self, loc1, loc2):
        """Helper to get precomputed distance, handles unreachable locations."""
        if loc1 not in self.distance_map or loc2 not in self.distance_map.get(loc1, {}):
             # This means loc1 or loc2 was not included in the locations set during init,
             # or there is no path between them.
             return float('inf')
        return self.distance_map[loc1][loc2]


    def __call__(self, node):
        """Estimate the minimum cost to tighten all goal nuts."""
        state = node.state

        # Requirement 2: Heuristic is 0 only for goal states.
        # Check if all goal nuts are already tightened.
        all_goal_nuts_tightened = True
        for nut in self.goal_nuts:
            if f'(tightened {nut})' not in state:
                all_goal_nuts_tightened = False
                break
        if all_goal_nuts_tightened:
            return 0

        # Requirement 3: Heuristic value is finite for solvable states.
        # We return infinity if any necessary component is unreachable.

        # 1. Identify the man and his location
        man_location = None
        carried_spanner = None # Store the name of the spanner if carried

        # Find man's location and if he is carrying a spanner
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3 and parts[1] == self.man_name:
                man_location = parts[2]
            elif parts[0] == 'carrying' and len(parts) == 3 and parts[1] == self.man_name:
                 carried_spanner = parts[2]

        if man_location is None:
             # Man location not found, problem state is likely invalid or goal unreachable.
             return float('inf')


        # 2. Identify loose goal nuts and their locations
        loose_goal_nuts_info = {} # {nut_name: location}
        nut_locations = {} # {nut_name: location}

        # First pass to find all nut locations
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('nut'): # Assuming nuts start with 'nut'
                 nut_locations[parts[1]] = parts[2]

        # Second pass to find loose goal nuts and their locations
        for nut_name in self.goal_nuts:
             if f'(loose {nut_name})' in state:
                  if nut_name in nut_locations:
                       loose_goal_nuts_info[nut_name] = nut_locations[nut_name]
                  else:
                       # Loose goal nut exists but its location is unknown. Problematic state.
                       return float('inf')


        # 3. Identify usable spanners and their locations/status
        usable_spanners_at_location = {} # {spanner_name: location}
        man_has_usable_spanner = False
        spanner_locations = {} # {spanner_name: location}

        # First pass to find all spanner locations
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('spanner'): # Assuming spanners start with 'spanner'
                  spanner_locations[parts[1]] = parts[2]

        # Second pass to find usable spanners and check if carried
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'usable' and len(parts) == 2 and parts[1].startswith('spanner'):
                  spanner_name = parts[1]
                  if carried_spanner == spanner_name:
                       man_has_usable_spanner = True
                  elif spanner_name in spanner_locations:
                       usable_spanners_at_location[spanner_name] = spanner_locations[spanner_name]
                  # else: usable spanner exists but its location is unknown or not at a standard location. Problematic state.


        # 4. Initialize total heuristic cost
        total_heuristic = 0

        # 5. Calculate cost for each loose goal nut
        for nut_name, nut_location in loose_goal_nuts_info.items():
            min_cost_for_nut = float('inf')

            # Check if the nut's location is even in our distance map
            if nut_location not in self.distance_map:
                 return float('inf') # Nut location is isolated

            # a. Cost if man is already carrying a usable spanner
            if man_has_usable_spanner:
                dist_man_to_nut = self.get_distance(man_location, nut_location)
                if dist_man_to_nut != float('inf'):
                    cost = dist_man_to_nut + 1 # walk + tighten
                    min_cost_for_nut = min(min_cost_for_nut, cost)

            # b. Cost if man needs to pick up an available usable spanner
            for spanner_name, spanner_location in usable_spanners_at_location.items():
                 # Check if spanner location is in our distance map
                 if spanner_location not in self.distance_map:
                      continue # Skip this spanner if its location is isolated

                 dist_man_to_spanner = self.get_distance(man_location, spanner_location)
                 dist_spanner_to_nut = self.get_distance(spanner_location, nut_location)

                 if dist_man_to_spanner != float('inf') and dist_spanner_to_nut != float('inf'):
                      cost = dist_man_to_spanner + 1 + dist_spanner_to_nut + 1 # walk1 + pickup + walk2 + tighten
                      min_cost_for_nut = min(min_cost_for_nut, cost)

            # e. Handle unreachable nuts or lack of spanners
            if min_cost_for_nut == float('inf'):
                 # This nut is unreachable or no usable spanner is reachable.
                 # Return infinity immediately if any goal nut cannot be tightened.
                 return float('inf')

            # f. Add cost for this nut to total
            total_heuristic += min_cost_for_nut

        # Return the total estimated cost
        return total_heuristic
