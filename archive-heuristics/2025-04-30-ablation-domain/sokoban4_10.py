from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to extract parts from a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

# BFS function to compute distances
def bfs_distances(start_node, graph):
    """Computes shortest path distances from start_node to all reachable nodes in the graph."""
    distances = {start_node: 0}
    queue = deque([start_node])
    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]
        # Ensure current_node is a key in the graph before accessing neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
    return distances

# The heuristic class
class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing the shortest
    path distances for each box to its specific goal location and adding the shortest
    path distance from the robot to the nearest box that still needs to be moved.
    It uses graph distances computed on the grid defined by 'adjacent' predicates.
    States where a box goal is unreachable or the robot cannot reach any box needing
    a move are assigned an infinite heuristic value, indicating a likely unsolvable state.

    # Assumptions
    - The environment is a grid defined by 'adjacent' predicates, forming a connected graph (or multiple components).
    - Each box has a unique goal location specified in the task goals.
    - The robot can only push one box at a time.
    - The 'adjacent' predicates define symmetric connections (if A is adjacent to B, B is adjacent to A), and both directions are listed in the static facts.
    - The initial state and all states reachable from a solvable initial state will have robot location and box locations present in the state facts.

    # Heuristic Initialization
    1. Parse the task goals to create a mapping from each box object to its target location string.
    2. Build an undirected graph representation of the grid connectivity based on the 'adjacent' predicates provided in the static facts. The graph stores adjacent locations for each location.
    3. For each unique goal location specified in the task goals, precompute the shortest path distances from that goal location to all other reachable locations in the grid using Breadth-First Search (BFS). Store these distances in a dictionary where keys are goal locations and values are dictionaries mapping reachable locations to their distances from the goal.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and all boxes by examining the state facts. Store box locations in a dictionary mapping box names to their locations.
    2. Create a list of 'boxes needing move' by identifying boxes whose current location is not their designated goal location.
    3. If the list of 'boxes needing move' is empty, the state is a goal state, and the heuristic value is 0.
    4. If there are boxes needing move, calculate the sum of shortest path distances for each of these boxes from its current location to its goal location.
       - For each box needing move, retrieve its goal location from the precomputed map.
       - Look up the precomputed distance from the goal location to the box's current location using the distances stored during initialization (`self.goal_distances[goal_loc][current_loc]`).
       - If the goal location was not in the graph (meaning it's isolated) or if the box's current location is not reachable from the goal location (based on precomputed distances), the state is likely unsolvable. Return `float('inf')` immediately.
       - Otherwise, add the distance to a running total for box distances (`total_box_distance`).
    5. Calculate the shortest path distance from the robot's current location to the location of the *nearest* box among those needing to be moved.
       - First, ensure the robot's location was found in the state and exists as a node in the grid graph. If not, return `float('inf')`.
       - Compute shortest path distances from the robot's current location to all other reachable locations using BFS (`robot_distances = bfs_distances(robot_loc, self.graph)`).
       - Iterate through the locations of the boxes needing move and find the minimum distance from the robot's location to any of these box locations using the distances computed in the previous step (`min_robot_box_dist`).
       - If the robot cannot reach any of the boxes needing move (i.e., the minimum distance remains `float('inf')`), the state is likely unsolvable. Return `float('inf')`.
    6. If the state is not deemed unsolvable in steps 4 or 5, the total heuristic value is the sum of the box-goal distances (calculated in step 4) plus the robot-to-nearest-box distance (calculated in step 5). This sum serves as an estimate of the total movement effort required.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the grid graph.
        """
        self.task = task

        # 1. Extract goal locations for each box
        self.box_goals = {}
        for goal in task.goals:
            # Goal is typically (at box loc)
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                box, location = parts[1], parts[2]
                self.box_goals[box] = location

        # 2. Build the adjacency graph from static facts
        self.graph = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                self.graph[loc1].append(loc2)
                # Assuming adjacency is symmetric and both directions are listed.

        # 3. Precompute shortest path distances from each goal location
        self.goal_distances = {}
        unique_goal_locations = set(self.box_goals.values())
        for goal_loc in unique_goal_locations:
            # Only compute distances if the goal location is a valid node in the graph
            if goal_loc in self.graph:
                 self.goal_distances[goal_loc] = bfs_distances(goal_loc, self.graph)
            # If a goal location is not in the graph, it implies it's an isolated location.
            # Any box needing to reach this goal (unless starting there) makes the state unsolvable.
            # This case is handled in __call__.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        total_box_distance = 0

        # Find current location of robot and boxes
        robot_loc = None
        box_locations = {}
        boxes_needing_move = [] # List of box names that are not at their goals

        # Iterate through the state facts to find locations
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'at-robot' and len(parts) == 2:
                 robot_loc = parts[1]
            elif parts[0] == 'at' and len(parts) == 3:
                 box, loc = parts[1], parts[2]
                 if box in self.box_goals:
                     box_locations[box] = loc
                     if loc != self.box_goals[box]:
                         boxes_needing_move.append(box)

        # If all boxes are at their goals, the heuristic is 0.
        if not boxes_needing_move:
             return 0

        # --- Calculate sum of box-goal distances ---
        # Check reachability and sum distances for boxes not at their goals
        for box in boxes_needing_move:
            current_loc = box_locations.get(box)
            goal_loc = self.box_goals[box]

            # Check if goal_loc was processed during initialization (i.e., is in the graph keys)
            # and if the current location is reachable from the goal.
            if goal_loc not in self.goal_distances or current_loc not in self.goal_distances[goal_loc]:
                 # Box goal is unreachable from its current location or goal is isolated.
                 return float('inf') # State is likely unsolvable

            # Add the shortest path distance from the box's current location to its goal
            total_box_distance += self.goal_distances[goal_loc][current_loc]

        # --- Calculate robot distance to nearest box needing move ---
        min_robot_box_dist = float('inf')

        # Ensure robot_loc was found and is a valid node in the graph
        if robot_loc is None or robot_loc not in self.graph:
             # Robot location not found or not in the grid graph. Problem likely unsolvable.
             return float('inf')

        # Compute distances from the robot's current location
        robot_distances = bfs_distances(robot_loc, self.graph)

        # Find the minimum distance from the robot to any box needing move
        robot_can_reach_any_box = False
        for box in boxes_needing_move:
            box_loc = box_locations.get(box)
            if box_loc and box_loc in robot_distances:
                min_robot_box_dist = min(min_robot_box_dist, robot_distances[box_loc])
                robot_can_reach_any_box = True

        # If the robot cannot reach any box that needs moving, the state is likely unsolvable.
        if not robot_can_reach_any_box:
             return float('inf')

        # --- Combine distances ---
        # The heuristic is the sum of box-goal distances plus the robot-to-nearest-box distance.
        return total_box_distance + min_robot_box_dist
