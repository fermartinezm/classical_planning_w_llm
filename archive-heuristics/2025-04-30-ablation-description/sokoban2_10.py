from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# BFS function for shortest path on the graph
def bfs_distance(graph, start, goal):
    """Find the shortest path distance between start and goal in the graph."""
    if start == goal:
        return 0
    # Handle cases where start or goal might not be in the graph (e.g., invalid state)
    if start not in graph or goal not in graph:
         return float('inf')

    queue = deque([(start, 0)]) # (location, distance)
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc in graph: # Ensure the location exists in the graph
            for neighbor in graph[current_loc]:
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return float('inf') # Goal is unreachable

# BFS function for shortest path from a start to any of multiple goals
def bfs_distance_to_any(graph, start, goals):
    """Find the shortest path distance from start to any location in the goals set."""
    if not goals: # No goals to reach
        return 0
    if start in goals:
        return 0
    if start not in graph:
        return float('inf') # Start location not in the connected graph

    queue = deque([(start, 0)]) # (location, distance)
    visited = {start}

    while queue:
        current_loc, dist = queue.popleft()

        if current_loc in graph: # Ensure the location exists in the graph
            for neighbor in graph[current_loc]:
                if neighbor in goals:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return float('inf') # No goal location is reachable

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost by summing the shortest path distances
    for each box from its current location to its goal location, and adding
    the shortest path distance from the robot to the nearest box that needs
    to be moved. Distances are calculated on the grid graph defined by the
    'adjacent' facts.

    # Assumptions
    - The goal specifies the target location for each box.
    - The grid structure is defined by 'adjacent' facts, forming an undirected graph.
    - The heuristic assumes that moving boxes closer to their goals and moving
      the robot closer to a box that needs moving are steps towards the solution.
      It uses shortest path distances on the grid graph as estimates, which
      ignores complex interactions like deadlocks, required robot positioning
      relative to the box for pushes, and dynamic obstacles (other boxes, robot)
      blocking paths. It is not strictly admissible but is intended to be
      informative for greedy best-first search.
    - The goal state is defined solely by the locations of the boxes. If other
      goal conditions exist (e.g., robot position), this heuristic might return
      0 when the state is not the true goal. Based on the examples, this
      assumption seems reasonable for this domain.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task goals.
    - Builds an undirected graph representing the grid connectivity based on
      the 'adjacent' facts in the static information.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is the goal state by verifying if all goal facts
       are present in the state. If it is, the heuristic value is 0.
    2. If not the goal state, find the current location of the robot from the state facts.
       If the robot's location is not found, the state is considered invalid or
       unreachable, and the heuristic returns infinity.
    3. Find the current location of each box that has a corresponding goal location
       defined in the task. Identify the set of locations occupied by boxes that
       are not yet at their respective goal locations.
    4. Calculate the sum of shortest path distances for each box that is not yet
       at its goal location. The distance is calculated using Breadth-First Search (BFS)
       on the pre-built grid graph from the box's current location to its goal location.
       If the goal location for any box is unreachable from its current location
       on the graph, the state is considered unsolvable, and the heuristic returns infinity.
    5. Calculate the shortest path distance from the robot's current location to the
       nearest location in the set of locations identified in step 3 (i.e., the location
       of the nearest box that needs moving). This distance is also calculated using
       BFS on the grid graph. If the robot cannot reach any box that needs moving,
       the state is considered unsolvable, and the heuristic returns infinity. If
       the set of boxes needing movement is empty (meaning all boxes are at their
       goals, but the state is not the overall goal state - this case is handled
       by step 1 under the assumption that goals are only box locations), this
       robot distance component is 0.
    6. The total heuristic value is the sum of the total box distances (from step 4)
       and the robot-to-nearest-box distance (from step 5).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the grid graph."""
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                box, location = parts[1:]
                self.goal_locations[box] = location

        # Build the adjacency list graph from static 'adjacent' facts.
        self.adj_list = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent":
                l1, l2, _ = parts[1:] # Ignore direction
                # Add bidirectional edges assuming adjacency is symmetric
                if l1 not in self.adj_list:
                    self.adj_list[l1] = set()
                if l2 not in self.adj_list:
                    self.adj_list[l2] = set()
                self.adj_list[l1].add(l2)
                self.adj_list[l2].add(l1)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Check if goal state
        # The goal is reached if all goal facts are present in the current state.
        if self.goals.issubset(state):
             return 0

        # 2. Find robot location
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_loc = parts[1]
                break
        # If robot location is not found, the state is invalid/unreachable
        if robot_loc is None:
             return float('inf')

        # 3. Find current box locations and identify boxes not at goals
        current_box_locations = {}
        boxes_not_at_goal_locs = set()
        for fact in state:
            parts = get_parts(fact)
            # Only consider facts about boxes that are part of the goal
            if parts[0] == "at" and parts[1] in self.goal_locations:
                box, loc = parts[1:]
                current_box_locations[box] = loc
                # Check if the box is not at its goal location
                if loc != self.goal_locations[box]:
                    boxes_not_at_goal_locs.add(loc)

        # 4. Calculate box_distances_sum
        box_distances_sum = 0
        for box, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box)
            # If a box with a goal isn't found in the state, it's likely an invalid state
            if current_loc is None:
                 return float('inf')

            # If the box is not at its goal, calculate distance
            if current_loc != goal_loc:
                # Calculate shortest distance from current box location to its goal.
                dist = bfs_distance(self.adj_list, current_loc, goal_loc)

                if dist == float('inf'):
                     # If any box goal is unreachable, the state is unsolvable.
                     return float('inf')

                box_distances_sum += dist

        # 5. & 6. Calculate robot_to_nearest_box_distance
        # Find the shortest path distance from the robot to any location
        # occupied by a box that is not at its goal.
        # If boxes_not_at_goal_locs is empty, it means all boxes are at their goals.
        # In this case, robot_to_nearest_box_distance should be 0.
        # The bfs_distance_to_any function handles the empty goals set case by returning 0.
        robot_to_nearest_box_distance = bfs_distance_to_any(
            self.adj_list, robot_loc, boxes_not_at_goal_locs
        )

        if robot_to_nearest_box_distance == float('inf'):
             # If the robot cannot reach any box that needs moving, the state is unsolvable.
             return float('inf')

        # 7. Return sum
        # The sum of box distances is a lower bound on box pushes.
        # The robot needs to reach a box to push it.
        # This sum is a simple combination that encourages moving boxes towards goals
        # and moving the robot towards boxes that need moving.
        return box_distances_sum + robot_to_nearest_box_distance
