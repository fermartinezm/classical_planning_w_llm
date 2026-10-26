from collections import deque
from heuristics.heuristic_base import Heuristic
import math

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def bfs(start_loc, end_loc, graph):
    """
    Compute the shortest path distance between two locations on the grid graph.
    Ignores dynamic obstacles (robot, other boxes).
    """
    if start_loc == end_loc:
        return 0

    queue = deque([(start_loc, 0)])
    visited = {start_loc}
    distance = {start_loc: 0}

    while queue:
        current_loc, current_dist = queue.popleft()

        if current_loc == end_loc:
            return current_dist

        # Check if current_loc exists in the graph (it should if it's a valid location)
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current_dist + 1))
                    distance[neighbor] = current_dist + 1

    # If end_loc is not reachable from start_loc
    return math.inf

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost by summing the shortest path distances
    from each box's current location to its goal location. The distance is
    calculated on the static grid defined by 'adjacent' facts, ignoring
    the robot's position and other boxes as obstacles.

    # Assumptions
    - The grid structure is defined by 'adjacent' facts.
    - Distance is the number of steps in the shortest path on this static grid.
    - The cost of moving a box is primarily the number of grid cells it must traverse.
    - The heuristic ignores the robot's movement cost and the requirement for clear
      spaces and specific robot positioning for pushing.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Builds an undirected graph representing the grid connectivity based on
      'adjacent' facts from the static information.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Identify the current location of each box in the state.
    3. For each box that has a specific goal location defined in the task:
       a. Get the box's current location.
       b. Get the box's goal location.
       c. If the box is already at its goal location, add 0 to the total cost.
       d. If the box is not at its goal location, calculate the shortest path
          distance between the box's current location and its goal location
          using BFS on the precomputed static grid graph.
       e. Add this distance to the total heuristic cost. If the goal is unreachable
          on the static grid, adding infinity ensures this state is highly penalized.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the static grid graph.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                obj, location = args
                # Assuming objects starting with 'box' are the ones we care about
                if obj.startswith('box'):
                     self.goal_locations[obj] = location

        # Build the static grid graph from adjacent facts.
        self.graph = {}
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "adjacent":
                loc1, loc2, direction = args
                # Add bidirectional edges
                if loc1 not in self.graph:
                    self.graph[loc1] = set()
                if loc2 not in self.graph:
                    self.graph[loc2] = set()
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find current locations of all objects (specifically boxes).
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                obj, location = args
                current_locations[obj] = location
            # We don't need the robot location for this specific heuristic calculation
            # elif predicate == "at-robot":
            #      robot_location = args[0]


        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each box that needs to reach a specific goal location.
        for box, goal_location in self.goal_locations.items():
            # Check if the box exists in the current state and has a location.
            # It should exist in any valid state of a solvable problem.
            if box in current_locations:
                current_location = current_locations[box]

                # If the box is already at its goal, cost is 0 for this box.
                if current_location == goal_location:
                    continue

                # Calculate shortest path distance from current location to goal location.
                # This BFS uses the static graph, ignoring dynamic obstacles.
                distance = bfs(current_location, goal_location, self.graph)

                # If the goal is unreachable for this box on the static grid,
                # this state is likely a dead end or part of an unsolvable problem.
                # Return infinity to prune this branch.
                if distance == math.inf:
                    return math.inf

                total_cost += distance

        # If there are boxes in the state that are not mentioned in the goal
        # (e.g., extra boxes), they don't contribute to this heuristic.
        # If there are goals for boxes not in the state, this is an invalid problem,
        # but the code handles it by skipping the loop for that box.

        return total_cost
