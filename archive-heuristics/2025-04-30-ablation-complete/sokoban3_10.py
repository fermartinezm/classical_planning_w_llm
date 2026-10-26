from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def bfs_distance(start, end, graph):
    """
    Calculates the shortest path distance between two locations in the grid graph
    using Breadth-First Search.
    """
    if start == end:
        return 0
    if start not in graph or end not in graph:
         return float('inf') # Cannot reach if not in graph

    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        current_loc, dist = queue.popleft()
        if current_loc == end:
            return dist
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
    return float('inf') # Goal not reachable


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the total minimum number of box movements required
    to get all boxes to their goal locations. It calculates the shortest path
    distance for each box from its current location to its goal location on
    the grid graph, and sums these distances.

    # Assumptions
    - The grid structure is defined by `adjacent` facts in the static information.
    - Goals are specified as `(at box location)` facts.
    - The heuristic ignores the robot's position and the potential for boxes
      or walls to block paths, making it non-admissible.
    - Unreachable goals (based on the static adjacency graph) result in an
      infinite heuristic value.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Builds an undirected adjacency graph of locations based on the `adjacent`
      facts provided in the static information.

    # Step-By-Step Thinking for Computing Heuristic
    1. Get the current state of the world.
    2. Identify the current location of each box by parsing the `(at ?b ?l)` facts in the state.
    3. Initialize the total heuristic cost to 0.
    4. For each box that has a specified goal location:
       a. Get the box's current location.
       b. Get the box's goal location.
       c. If the box is not already at its goal location:
          i. Calculate the shortest path distance between the box's current location
             and its goal location using Breadth-First Search (BFS) on the
             pre-computed location graph.
          ii. If the goal is unreachable from the current location, return infinity
              immediately (indicating a likely unsolvable state).
          iii. Add the calculated distance to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the location graph from static facts.
        """
        # Store goal locations for each box.
        self.box_goals = {}
        for goal in task.goals:
            parts = get_parts(goal)
            # Assuming goals are always (at box_name location)
            if parts[0] == 'at':
                box_name = parts[1]
                goal_location = parts[2]
                self.box_goals[box_name] = goal_location
            # Ignore other types of goals if any

        # Build the adjacency graph from static facts.
        self.graph = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1:]
                if loc1 not in self.graph:
                    self.graph[loc1] = set()
                if loc2 not in self.graph:
                    self.graph[loc2] = set()
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1) # Adjacency is symmetric

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find current box locations.
        box_locations = {}
        # Robot location is not strictly needed for this simple heuristic,
        # but we can parse it anyway.
        # robot_location = None # Not used in this heuristic
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'at':
                # Fact is (at box_name location)
                box_name = parts[1]
                location = parts[2]
                box_locations[box_name] = location
            # elif predicate == 'at-robot':
            #     robot_location = parts[1] # Not used in this heuristic

        total_cost = 0  # Initialize action cost counter.

        # Calculate sum of distances for each box to its goal.
        for box_name, goal_location in self.box_goals.items():
            # Ensure the box exists in the current state (should always be true in valid states)
            if box_name in box_locations:
                current_location = box_locations[box_name]

                # If box is already at goal, cost is 0 for this box.
                if current_location != goal_location:
                    # Calculate shortest path distance for the box.
                    box_dist = bfs_distance(current_location, goal_location, self.graph)

                    # If goal is unreachable for this box, the state is likely unsolvable.
                    if box_dist == float('inf'):
                        return float('inf')

                    total_cost += box_dist
            # else:
                # This case should ideally not happen in a valid state representation
                # where all objects are accounted for. If it could, we might return inf.
                # For now, assume box_name is always in box_locations if it's in box_goals.


        return total_cost
