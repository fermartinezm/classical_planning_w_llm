from collections import deque
# from fnmatch import fnmatch # Not used in this heuristic
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    fact = fact.strip()
    if fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    # Fallback for facts without parentheses, though PDDL facts usually have them
    return fact.split()

def build_adjacency_graph(static_facts):
    """Builds an undirected graph from 'adjacent' static facts."""
    graph = {}
    for fact in static_facts:
        parts = get_parts(fact)
        if parts and parts[0] == 'adjacent':
            # (adjacent loc1 loc2 direction)
            l1, l2 = parts[1], parts[2]
            graph.setdefault(l1, set()).add(l2)
            graph.setdefault(l2, set()).add(l1) # Add reverse edge for undirected graph
    return graph

def bfs(graph, start_node):
    """Performs BFS to find shortest distances from start_node to all reachable nodes."""
    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        for neighbor in graph.get(current_node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = current_dist + 1
                queue.append(neighbor)
    return distances

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing the
    shortest path distances for each box to its goal location and adding the
    shortest path distance for the robot to reach a location adjacent to any
    box that is not yet at its goal.

    # Assumptions
    - The grid/graph defined by 'adjacent' facts is connected (or relevant parts are).
    - Shortest path distances on this graph are a reasonable proxy for movement costs.
    - The heuristic ignores complex state constraints like 'clear' preconditions
      for robot movement or box pushing, and potential deadlocks.
    - The cost of a 'move' action is 1, and the cost of a 'push' action is 1.
      A 'push' action moves the box one step and the robot one step.
      The heuristic approximates the cost as box moves (pushes) + robot moves to get into position.

    # Heuristic Initialization
    - Parses the goal conditions to map each box to its target location.
    - Builds an undirected graph of locations based on 'adjacent' facts.
    - Pre-calculates shortest path distances between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes from the state.
    2. Identify which boxes are not yet at their goal locations.
    3. If all boxes are at their goals, the heuristic is 0.
    4. For each box not at its goal:
       - Calculate the shortest path distance from the box's current location to its goal location using the pre-calculated distances. Sum these distances. This estimates the minimum number of 'push' actions required in total for all boxes.
    5. Calculate the minimum shortest path distance from the robot's current location to *any* location that is adjacent to *any* box not at its goal. This estimates the minimum number of 'move' actions required for the robot to get into a position from which it *could* potentially push a box.
    6. The total heuristic value is the sum of the total box-to-goal distance (step 4) and the minimum robot-to-box-adjacent distance (step 5).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the distance matrix for locations.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse goals to get box-goal mapping
        self.box_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'at':
                # Goal is (at ?b ?l)
                box, location = parts[1], parts[2]
                self.box_goals[box] = location

        # 2. Build adjacency graph from static facts
        self.adj_graph = build_adjacency_graph(static_facts)

        # 3. Pre-calculate all-pairs shortest paths
        # Collect all locations mentioned in the graph
        all_locations = set(self.adj_graph.keys())
        # Also include locations mentioned in goals but maybe not in adjacent facts (unlikely but safe)
        all_locations.update(self.box_goals.values())

        self.dist_matrix = {}
        for loc in all_locations:
            # Only run BFS if the location is actually in the graph (has neighbors)
            if loc in self.adj_graph:
                 self.dist_matrix[loc] = bfs(self.adj_graph, loc)
            else:
                 # Location is isolated, cannot reach or be reached from it
                 self.dist_matrix[loc] = {} # Empty dict indicates no reachable nodes

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # 1. Identify robot and box locations
        robot_loc = None
        box_locations = {}
        # We only care about boxes that have a goal location
        relevant_boxes = set(self.box_goals.keys())

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            if parts[0] == 'at-robot':
                robot_loc = parts[1]
            elif parts[0] == 'at' and parts[1] in relevant_boxes:
                 box, loc = parts[1], parts[2]
                 box_locations[box] = loc

        # Basic validation: Ensure robot and all relevant boxes are located
        # If not, this state is likely invalid or unreachable in a solvable problem
        if robot_loc is None or len(box_locations) != len(relevant_boxes):
             # print("Debug: Could not find robot or all relevant boxes in state.")
             return float('inf')

        # 2. Identify active boxes (not at goal)
        active_boxes = {
            box for box in relevant_boxes
            if box_locations.get(box) != self.box_goals[box]
        }

        # 3. Goal check
        if not active_boxes:
            return 0 # All boxes are at their goals

        # 4. Calculate total box-to-goal distance
        total_box_distance = 0
        for box in active_boxes:
            box_loc = box_locations[box]
            goal_loc = self.box_goals[box]

            # Get distance from pre-calculated matrix
            # dist_matrix[start_loc] gives distances FROM start_loc
            dist_from_box = self.dist_matrix.get(box_loc)
            if dist_from_box is None:
                 # Box is at an isolated location not in the graph
                 # print(f"Debug: Box at isolated location {box_loc}")
                 return float('inf') # Unreachable goal

            dist = dist_from_box.get(goal_loc)
            if dist is None:
                 # Goal is unreachable from box location
                 # print(f"Debug: Goal {goal_loc} unreachable from box at {box_loc}")
                 return float('inf') # Unreachable goal

            total_box_distance += dist

        # 5. Calculate minimum robot-to-box-adjacent distance
        min_robot_to_any_box_adjacent = float('inf')
        robot_dist_map = self.dist_matrix.get(robot_loc)

        if robot_dist_map is None:
             # Robot is at an isolated location
             # print(f"Debug: Robot at isolated location {robot_loc}")
             return float('inf') # Cannot reach anything

        for box in active_boxes:
            box_loc = box_locations[box]
            # Iterate through locations adjacent to the box
            for adj_loc in self.adj_graph.get(box_loc, set()):
                 # Get distance from robot_loc to adj_loc
                 dist = robot_dist_map.get(adj_loc)
                 if dist is not None: # Ensure adj_loc is reachable from robot_loc
                     min_robot_to_any_box_adjacent = min(min_robot_to_any_box_adjacent, dist)

        # If min_robot_to_any_box_adjacent is still infinity, it means the robot
        # cannot reach any location adjacent to any active box.
        if min_robot_to_any_box_adjacent == float('inf'):
             # print("Debug: Robot cannot reach any location adjacent to active boxes.")
             return float('inf')

        # 6. Total heuristic value
        return total_box_distance + min_robot_to_any_box_adjacent
