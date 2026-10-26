import heapq
import logging
from collections import deque

from heuristics.heuristic_base import Heuristic
from task import Operator, Task

# Helper function to parse facts
def parse_fact(fact_string):
    """Parses a PDDL fact string into predicate and arguments."""
    # Remove leading/trailing parentheses and split by spaces
    parts = fact_string.strip('()').split()
    if not parts:
        return None, [] # Handle empty string case
    return parts[0], parts[1:]

# BFS function
def bfs(graph, start):
    """Performs BFS to find shortest paths from a start node in a graph."""
    distances = {node: float('inf') for node in graph}
    if start not in graph:
         # Start node is not in the graph (e.g., isolated location)
         # Distances to all other nodes remain infinity, except to itself (0)
         if start in distances:
              distances[start] = 0
         return distances

    distances[start] = 0
    queue = deque([start])
    while queue:
        current = queue.popleft()
        # Check if current node exists in graph keys (handles isolated nodes added to graph structure)
        if current in graph:
            for neighbor in graph[current]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
    return distances


class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    Estimates the cost to reach the goal by summing the minimum actions
    required for each package that is not yet at its goal location.
    It calculates the number of drive actions needed using precomputed
    shortest paths on the road network and adds 1 for each required
    pick-up and drop action.

    Assumptions:
    - The heuristic assumes that a suitable vehicle is always available
      at the required location with sufficient capacity whenever a package
      needs to be picked up from a location (when the package is not
      already in a vehicle). This is a significant relaxation.
    - The heuristic only considers packages explicitly listed in the goal
      state with an '(at ?p ?l)' predicate. Goals involving vehicle locations
      or capacities are ignored.
    - The road network defined by '(road ?l1 ?l2)' facts is used to compute
      shortest path distances (number of drive actions). Roads are assumed
      to be bidirectional if '(road l1 l2)' and '(road l2 l1)' are present,
      which is handled by building a bidirectional graph.
    - The heuristic does not consider vehicle capacity constraints or
      the location/availability of specific vehicles when a package is
      at a location (not in a vehicle).

    Heuristic Initialization:
    In the constructor, the heuristic performs the following steps:
    1. Parses the static facts from the task description.
    2. Builds a graph representation of the road network using '(road ?l1 ?l2)'
       facts. It identifies all unique locations mentioned in the static facts,
       initial state, and goal state to ensure all relevant locations are
       included as nodes in the graph structure.
    3. Computes all-pairs shortest paths (number of drive actions) between
       all identified locations using Breadth-First Search (BFS). These
       distances are stored in a dictionary for quick lookup during heuristic
       computation.
    4. Identifies the goal location for each package by parsing the '(at ?p ?l)'
       facts in the task's goal state. It assumes objects starting with 'p'
       are packages based on common PDDL naming conventions for this domain.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Extract the current location of all locatables ('(at ?x ?l)') and
       which packages are currently inside which vehicles ('(in ?p ?v)')
       from the state facts.
    2. Initialize the total heuristic value `h` to 0.
    3. Initialize a flag `unreachable` to False.
    4. Iterate through each package `p` that has an explicit goal location
       `goal_l` defined in the task's goal state (`self.package_goals`).
    5. For the current package `p`:
       a. Check if the fact '(at p goal_l)' is present in the current state.
          If yes, the package is already at its final destination and not
          in a vehicle; its contribution to the heuristic is 0. Continue
          to the next package.
       b. If '(at p goal_l)' is not in the state, check if the package `p`
          is currently inside a vehicle `v` (i.e., '(in p v)' is in the state).
          i. If `p` is in vehicle `v`: Find the current location `current_l`
             of vehicle `v` from the state ('(at v current_l)'). If the vehicle
             has no location in the state (indicates an invalid state), set
             `unreachable` to True and break the loop.
             - If `current_l` is the same as `goal_l`, the package is in the
               vehicle at the goal location. It only needs one 'drop' action.
               Add 1 to `h`.
             - If `current_l` is different from `goal_l`, the vehicle needs
               to drive from `current_l` to `goal_l`, and then the package
               needs to be dropped. Look up the precomputed distance
               `Distance(current_l, goal_l)`. If the goal location is unreachable
               from the vehicle's current location (distance is infinity), set
               `unreachable` to True and break the loop. Otherwise, add the
               distance (number of drive actions) plus 1 (for the drop action)
               to `h`.
          ii. If `p` is not in a vehicle (meaning it must be at a location
              '(at p current_l)' where `current_l != goal_l`): The package
              needs to be picked up, transported, and dropped. Assuming a
              vehicle is available instantly at `current_l` with sufficient
              capacity, this requires one 'pick-up' action, driving from
              `current_l` to `goal_l`, and one 'drop' action. Look up the
              precomputed distance `Distance(current_l, goal_l)`. If the goal
              location is unreachable from the package's current location
              (distance is infinity), set `unreachable` to True and break
              the loop. Otherwise, add 1 (pickup) + distance (drive actions)
              + 1 (drop) to `h`.
          iii. If `p` is neither in a vehicle nor at a location (indicates
               an invalid state), set `unreachable` to True and break the loop.
    6. If the `unreachable` flag is True after checking all packages, return
       `float('inf')`.
    7. Otherwise, return the total heuristic value `h`. The heuristic is 0
       if and only if all package goals `(at p l)` are satisfied in the state.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.package_goals = {}
        self.road_graph = {}
        self._flat_distances = {}

        # Collect all locations mentioned in the problem
        all_locations = set()

        # Parse static facts
        for fact in task.static:
            pred, args = parse_fact(fact)
            if pred == 'road' and len(args) == 2:
                l1, l2 = args
                all_locations.add(l1)
                all_locations.add(l2)
                self.road_graph.setdefault(l1, []).append(l2)
                self.road_graph.setdefault(l2, []).append(l1) # Assuming bidirectional roads

        # Add locations from initial state and goals to ensure they are in the graph nodes
        for fact in task.initial_state:
             pred, args = parse_fact(fact)
             if pred == 'at' and len(args) == 2:
                  # args[0] is locatable, args[1] is location
                  all_locations.add(args[1])
        for fact in task.goals:
             pred, args = parse_fact(fact)
             if pred == 'at' and len(args) == 2:
                  # args[0] is locatable, args[1] is location
                  all_locations.add(args[1])

        # Ensure all collected locations are nodes in the graph structure, even if isolated
        # This allows BFS to correctly report infinity distance to/from isolated nodes
        for loc in all_locations:
            self.road_graph.setdefault(loc, [])

        # Compute all-pairs shortest paths
        for start_loc in self.road_graph:
            distances_from_start = bfs(self.road_graph, start_loc)
            for end_loc, dist in distances_from_start.items():
                 self._flat_distances[(start_loc, end_loc)] = dist

        # Identify package goals
        for goal_fact in task.goals:
            pred, args = parse_fact(goal_fact)
            if pred == 'at' and len(args) == 2:
                # Assuming goal facts like (at package location)
                item_name, loc_name = args
                # Assume objects starting with 'p' are packages based on domain examples
                if item_name.startswith('p'):
                     self.package_goals[item_name] = loc_name
                # Ignore other goal types like (at vehicle location) or (in package vehicle)

        # logging.info(f"Road Graph: {self.road_graph}")
        # logging.info(f"Distances: {self._flat_distances}")
        # logging.info(f"Package Goals: {self.package_goals}")


    def __call__(self, node):
        state = node.state
        current_location = {} # object -> location
        package_in_vehicle = {} # package -> vehicle
        # vehicle_capacity = {} # Not needed for this simple heuristic

        # Extract dynamic information from the state
        for fact in state:
            pred, args = parse_fact(fact)
            if pred == 'at' and len(args) == 2:
                current_location[args[0]] = args[1]
            elif pred == 'in' and len(args) == 2:
                package_in_vehicle[args[0]] = args[1]
            # elif pred == 'capacity' and len(args) == 2:
            #     vehicle_capacity[args[0]] = args[1] # Not needed

        h = 0
        unreachable = False

        # Calculate cost for each package that has a goal location
        for package, goal_l in self.package_goals.items():
            # Check if package is already at its final goal location (and not in a vehicle)
            if f'(at {package} {goal_l})' in state:
                # Package is done, contributes 0
                continue

            # Check if package is in a vehicle
            if package in package_in_vehicle:
                vehicle = package_in_vehicle[package]
                # Find vehicle's current location
                current_l = current_location.get(vehicle)
                if current_l is None:
                    # Vehicle containing package has no location - indicates invalid state
                    logging.warning(f"Vehicle {vehicle} containing package {package} has no location in state.")
                    unreachable = True
                    break # Cannot compute finite heuristic

                if current_l == goal_l:
                    # Package is in vehicle at goal location. Needs 1 drop action.
                    h += 1
                else:
                    # Package is in vehicle, vehicle is not at goal location. Needs drive + drop.
                    dist = self._flat_distances.get((current_l, goal_l), float('inf'))
                    if dist == float('inf'):
                        logging.warning(f"Vehicle at {current_l} cannot reach goal {goal_l} for package {package}.")
                        unreachable = True
                        break # Cannot compute finite heuristic
                    h += dist + 1 # drive actions + drop action

            # Check if package is at a location (and not in a vehicle, already checked above)
            elif package in current_location:
                 current_l = current_location[package]
                 # If we reached here, the package is at current_l and current_l != goal_l
                 # Needs pickup + drive + drop. Assumes vehicle availability.
                 dist = self._flat_distances.get((current_l, goal_l), float('inf'))
                 if dist == float('inf'):
                     logging.warning(f"Package at {current_l} cannot reach goal {goal_l}.")
                     unreachable = True
                     break # Cannot compute finite heuristic
                 h += 1 + dist + 1 # pickup + drive actions + drop action
            else:
                # Package is neither at a location nor in a vehicle - indicates invalid state
                logging.warning(f"Package {package} is neither at a location nor in a vehicle.")
                unreachable = True
                break # Cannot compute finite heuristic

        if unreachable:
            return float('inf')

        # Heuristic is 0 only if all package goals (at p l) are met.
        # This is true if the loop finishes without adding cost for any package.
        # If there are no package goals, h remains 0.
        return h

