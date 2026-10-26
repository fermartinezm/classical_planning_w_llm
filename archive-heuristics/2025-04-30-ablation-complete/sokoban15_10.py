from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input gracefully, although problem description implies valid facts
        # Returning empty list or raising error might be options.
        # For this context, assuming valid input format.
        return fact[1:-1].split()
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing two components for each box that is not yet at its goal location:
    1. The shortest path distance (graph distance) from the box's current location to its goal location. This represents the minimum number of push actions required for the box itself.
    2. The shortest path distance (graph distance) from the robot's current location to the box's current location. This represents the cost for the robot to reach the box to start pushing it.

    The total heuristic value is the sum of these costs over all misplaced boxes. This heuristic is non-admissible as the robot's movement cost might be counted multiple times if it needs to move towards different boxes.

    # Assumptions
    - The grid structure and connectivity are fully defined by the `adjacent` static facts.
    - All locations mentioned in the problem (initial state, goals, adjacent facts) are part of the connected graph defined by `adjacent` facts, or are otherwise unreachable (resulting in infinite heuristic).
    - The cost of a 'move' action is 1.
    - The cost of a 'push' action is 1.

    # Heuristic Initialization
    - Extracts the goal locations for each box from the task definition.
    - Builds an undirected graph representing the locations and their adjacencies based on the `adjacent` static facts.
    - Computes all-pairs shortest paths (APSP) on this graph using Breadth-First Search (BFS) starting from every location. These distances are stored for efficient lookup during heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot.
    2. Identify the current location of each box.
    3. Initialize the total heuristic cost to 0.
    4. For each box specified in the goal conditions:
        a. Determine the box's current location and its goal location.
        b. If the box is already at its goal location, add 0 cost for this box and proceed to the next box.
        c. If the box is not at its goal:
            i. Calculate the shortest path distance (graph distance) from the box's current location to its goal location using the pre-computed distances. If no path exists, the state is likely unsolvable, return infinity.
            ii. Add this box-to-goal distance to the total heuristic cost.
            iii. Calculate the shortest path distance (graph distance) from the robot's current location to the box's current location using the pre-computed distances. If no path exists, the state is likely unsolvable, return infinity.
            iv. Add this robot-to-box distance to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        location graph, and computing all-pairs shortest paths.
        """
        # Call the base class constructor if it exists and needs the task
        # super().__init__(task)

        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Build the adjacency graph and collect all locations
        self.graph = {}
        all_locations = set()
        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                all_locations.add(loc1)
                all_locations.add(loc2)
                self.graph.setdefault(loc1, []).append(loc2)
                self.graph.setdefault(loc2, []).append(loc1) # Adjacency is symmetric

        self.locations = list(all_locations) # Store list of all locations
        self.distances = {} # Store all-pairs shortest paths

        # Compute all-pairs shortest paths using BFS
        for start_loc in self.locations:
            self._bfs(start_loc)

    def _bfs(self, start_loc):
        """Performs BFS from start_loc to find distances to all reachable locations."""
        queue = deque([(start_loc, 0)]) # Use deque for efficient pop(0)
        visited = {start_loc}
        
        # Store distances from this start_loc
        self.distances[(start_loc, start_loc)] = 0 # Distance to self is 0

        while queue:
            current_loc, current_dist = queue.popleft() # Use popleft() for deque

            # Get neighbors from the graph, handling locations not in graph (shouldn't happen if all locations are in adjacent facts)
            neighbors = self.graph.get(current_loc, [])

            for neighbor_loc in neighbors:
                if neighbor_loc not in visited:
                    visited.add(neighbor_loc)
                    self.distances[(start_loc, neighbor_loc)] = current_dist + 1 # Store distance
                    queue.append((neighbor_loc, current_dist + 1))

        # Note: If a location is unreachable from start_loc, it won't be in self.distances
        # when accessed with (start_loc, unreachable_loc). The .get() method in __call__
        # will handle this by returning float('inf').


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break
        # If robot location isn't found, something is wrong with the state representation or domain.
        # assert robot_loc is not None, "Robot location not found in state"
        # If robot_loc is None, the state is invalid/unreachable, return infinity.
        if robot_loc is None:
             return float('inf')


        # Find box locations
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                predicate, box, loc = get_parts(fact)
                if box in self.goal_locations: # Only track boxes relevant to the goal
                     box_locations[box] = loc

        total_cost = 0  # Initialize action cost counter.

        # Check if all goal boxes are present in the state
        # This is a robustness check for potentially malformed states
        if len(box_locations) < len(self.goal_locations):
             # Some goal boxes are not found in the state facts
             return float('inf')


        for box, goal_location in self.goal_locations.items():
            current_location = box_locations.get(box)

            # This check is redundant if len(box_locations) == len(self.goal_locations)
            # and all keys in goal_locations are present in box_locations.
            # Keeping it for safety if goal_locations might contain boxes not in initial state.
            if current_location is None:
                 return float('inf')

            # If the box is already at its goal, no cost for this box
            if current_location == goal_location:
                continue

            # Calculate box-to-goal distance
            box_dist = self.distances.get((current_location, goal_location), float('inf'))
            if box_dist == float('inf'):
                # Box cannot reach its goal from its current position
                return float('inf')
            total_cost += box_dist

            # Calculate robot-to-box distance
            # The robot needs to reach the box's location (or adjacent to it)
            # Using distance to box location as a simple proxy
            robot_dist = self.distances.get((robot_loc, current_location), float('inf'))
            if robot_dist == float('inf'):
                # Robot cannot reach the box
                return float('inf')
            total_cost += robot_dist

        return total_cost
