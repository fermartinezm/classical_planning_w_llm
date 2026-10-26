from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

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

def bfs(graph, start_node):
    """
    Performs a Breadth-First Search to find shortest distances from a start node
    in an unweighted graph.

    Args:
        graph: Adjacency list representation of the graph {node: [neighbor1, ...]}
        start_node: The node to start the BFS from.

    Returns:
        A dictionary mapping each reachable node to its distance from the start_node.
        Nodes not reachable will not be in the dictionary or have distance infinity.
    """
    distances = {node: float('inf') for node in graph}
    if start_node in graph: # Ensure start_node is in the graph
        distances[start_node] = 0
        queue = deque([start_node])
        while queue:
            current_node = queue.popleft()
            if current_node not in graph: continue # Should not happen if graph is built correctly from links
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It sums the number of required tighten actions, the number of required spanner
    pickup actions, and an estimate of the travel cost.

    # Heuristic Initialization
    - Precomputes shortest path distances between all locations using BFS based on
      the 'link' predicates.
    - Identifies all potential nut and spanner objects from the initial state and goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Identify all nuts that are currently 'loose'.
    3. Count the number of usable spanners the man is currently 'carrying'.
    4. Identify all usable spanners that are currently 'at' a location (on the ground).
    5. Calculate the base cost:
       - Number of loose nuts (each needs a 'tighten_nut' action).
       - Number of spanners the man needs to pick up from the ground
         (max(0, num_loose_nuts - num_carried_usable_spanners)). Each needs a 'pickup_spanner' action.
    6. Estimate travel cost:
       - The man needs to visit the location of each loose nut.
       - The man needs to visit the location of each spanner he must pick up from the ground.
       - Estimate travel cost as the sum of shortest path distances from the man's
         current location to *each* required destination location (nut locations
         and locations of needed ground spanners). This overestimates travel but
         provides a simple, computable gradient.
    7. Sum the base cost and the estimated travel cost.
    8. If the goal is unreachable (e.g., not enough usable spanners exist in total),
       return infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances and identifying objects.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Precompute distances between all locations using BFS
        locations = set()
        graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                graph.setdefault(l1, []).append(l2)
                graph.setdefault(l2, []).append(l1) # Links are bidirectional

        self.distances = {}
        # Ensure all locations mentioned in initial state or goals are included,
        # even if they have no links (though this is unlikely in valid problems).
        all_mentioned_locations = set(locations)
        for fact in initial_state:
             if match(fact, 'at', '*', '*'):
                  all_mentioned_locations.add(get_parts(fact)[2])
        for goal in self.goals:
             if match(goal, 'at', '*', '*'):
                  all_mentioned_locations.add(get_parts(goal)[2])

        # Add any mentioned locations that weren't in links to the graph nodes
        for loc in all_mentioned_locations:
             graph.setdefault(loc, [])

        for loc in graph: # Iterate through all identified locations
            self.distances[loc] = bfs(graph, loc)

        # Identify all potential nut and spanner objects
        self.all_nuts = set()
        self.all_spanners = set()
        self.man_obj = None # Assuming one man object

        # Identify man, nuts, and spanners from initial state and goals
        potential_locatables = set()
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                  obj = parts[1]
                  potential_locatables.add(obj)
             elif parts[0] == 'loose':
                  self.all_nuts.add(parts[1])
             elif parts[0] == 'usable':
                  self.all_spanners.add(parts[1])
             elif parts[0] == 'carrying':
                  man, spanner = parts[1], parts[2]
                  self.man_obj = man # Found the man object
                  self.all_spanners.add(spanner)

        for goal in self.goals:
             if match(goal, 'tightened', '*'):
                  self.all_nuts.add(get_parts(goal)[1])

        # If man_obj wasn't found via 'carrying', find the locatable that isn't a nut or spanner
        if self.man_obj is None:
             for obj in potential_locatables:
                  if obj not in self.all_nuts and obj not in self.all_spanners:
                       self.man_obj = obj
                       break

        # If man_obj is still None, problem definition is unexpected
        if self.man_obj is None:
             print("Warning: Could not identify the man object in the initial state.")
             # Heuristic might fail or return inf later if man_loc cannot be found.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Find Man's Location
        man_loc = None
        if self.man_obj:
            for fact in state:
                if match(fact, 'at', self.man_obj, '*'):
                    man_loc = get_parts(fact)[2]
                    break

        if man_loc is None:
             # Man is not at any location (should not happen in this domain)
             return float('inf')

        # 2. Find Loose Nuts and their Locations
        loose_nuts = set()
        nut_locations = {} # {nut: location}
        for nut in self.all_nuts:
             if '(loose ' + nut + ')' in state:
                  loose_nuts.add(nut)
                  # Find location of this loose nut
                  for fact in state:
                       if match(fact, 'at', nut, '*'):
                            nut_locations[nut] = get_parts(fact)[2]
                            break # Found location for this nut
                  # If nut location not found, something is wrong
                  if nut not in nut_locations:
                       return float('inf')


        num_nuts = len(loose_nuts)
        if num_nuts == 0:
            return 0 # Goal reached

        # 3. Find Carried Usable Spanners
        carried_usable_count = 0
        carried_spanners = set()
        if self.man_obj:
            for fact in state:
                 if match(fact, 'carrying', self.man_obj, '*'):
                      spanner = get_parts(fact)[2]
                      carried_spanners.add(spanner)
                      # Check if this carried spanner is usable in the current state
                      if '(usable ' + spanner + ')' in state:
                           carried_usable_count += 1

        # 4. Find Usable Spanners on the Ground and their Locations
        ground_usable_spanners = {} # {spanner: location}
        for spanner in self.all_spanners:
             # Check if spanner is usable and not carried
             if '(usable ' + spanner + ')' in state and spanner not in carried_spanners:
                  # Find location of this usable ground spanner
                  for fact in state:
                       if match(fact, 'at', spanner, '*'):
                            ground_usable_spanners[spanner] = get_parts(fact)[2]
                            break # Found location for this spanner
                  # If usable ground spanner location not found, something is wrong
                  if spanner not in ground_usable_spanners:
                       # This spanner is usable but neither carried nor on the ground? Impossible state.
                       return float('inf')


        # Calculate base cost: tighten actions + pickup actions
        h = num_nuts # Cost for tighten actions

        total_usable_spanners_available = carried_usable_count + len(ground_usable_spanners)
        if total_usable_spanners_available < num_nuts:
             # Not enough usable spanners in the entire state to tighten all nuts
             return float('inf')

        needed_pickups = max(0, num_nuts - carried_usable_count)
        h += needed_pickups # Cost for pickup actions

        # Calculate travel cost
        target_locations = set()

        # Add nut locations to targets
        for nut_loc in nut_locations.values():
             target_locations.add(nut_loc)

        # Add locations of needed ground spanners to targets
        if needed_pickups > 0:
             ground_spanner_distances = [] # List of (distance, spanner_loc)
             for spanner, spanner_loc in ground_usable_spanners.items():
                  if man_loc in self.distances and spanner_loc in self.distances[man_loc]:
                       dist = self.distances[man_loc][spanner_loc]
                       ground_spanner_distances.append((dist, spanner_loc))
                  else:
                       # Impossible path from man to a usable ground spanner
                       return float('inf')

             # Sort by distance and take the locations of the closest 'needed_pickups' spanners
             ground_spanner_distances.sort()
             # Take min(needed_pickups, actual_available_ground_spanners) locations
             for i in range(min(needed_pickups, len(ground_spanner_distances))):
                  target_locations.add(ground_spanner_distances[i][1])


        # Calculate travel cost component: Sum of distances from man_loc to each target location
        travel_cost = 0
        for target_loc in target_locations:
             if man_loc in self.distances and target_loc in self.distances[man_loc]:
                  travel_cost += self.distances[man_loc][target_loc]
             else:
                  # Impossible path from man to a target location
                  return float('inf')

        h += travel_cost

        return h

