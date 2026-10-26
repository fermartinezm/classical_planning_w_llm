from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import math # For float('inf')

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The original example implementation relies on zip and all, which works
    # if the number of parts matches the number of args for the predicates used.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest
    path distances for each box to its goal location, plus the shortest path
    distance for the robot to reach the nearest box that needs to be moved.
    The distances are calculated on the grid graph defined by 'adjacent' facts,
    ignoring the robot and other boxes as obstacles (relaxed problem).

    # Assumptions
    - The grid structure is defined by 'adjacent' facts, forming an undirected graph.
    - Movement is possible between adjacent locations for both the robot and boxes (in the relaxed problem).
    - The shortest path distance on the grid is a reasonable estimate of
      the minimum number of moves/pushes required in a relaxed problem.
    - The cost of moving a box is related to its distance to the goal.
    - The robot must be near a box to push it.

    # Heuristic Initialization
    - Build the grid graph from 'adjacent' facts found in the static information.
      The graph nodes are locations, and edges connect adjacent locations.
    - Store the goal locations for each box that appears in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current location of the robot from the state facts.
    2. Identify the current location of each box that is a goal object from the state facts.
    3. Check if all goal boxes are already at their respective goal locations. If yes, the heuristic is 0.
    4. Initialize the total heuristic value to 0.
    5. Create a list of boxes that are not currently at their goal locations.
    6. For each box in the list of boxes to move:
       a. Get the box's current location and its goal location.
       b. Calculate the shortest path distance between the box's current location
          and its goal location using Breadth-First Search (BFS) on the pre-built
          grid graph. This BFS ignores dynamic obstacles (robot, other boxes).
       c. If no path exists (distance is -1), the state is likely unsolvable; return infinity.
       d. Add this distance to the total heuristic value. This represents the minimum
          number of "pushes" needed for this box in a relaxed setting.
    7. Calculate the shortest path distance from the robot's current location
       to the location of the *nearest* box in the list of boxes to move.
       Use BFS on the grid graph for this calculation.
    8. If the robot cannot reach any box that needs moving (minimum distance is infinity or -1),
       the state is likely unsolvable; return infinity.
    9. Add the calculated minimum robot distance to the total heuristic value. This estimates
       the initial effort for the robot to engage with the task.
    10. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - The grid graph from 'adjacent' facts.
        - Goal locations for each box.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the adjacency list graph from static 'adjacent' facts.
        self.graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "adjacent":
                loc1, loc2, direction = parts[1:]
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                if loc2 not in self.graph:
                    self.graph[loc2] = []
                # Add edges in both directions since adjacency seems symmetric
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1)

        # Store goal locations for each box that is mentioned in the goal.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

    def _bfs_distance(self, start_node, end_node):
        """
        Calculate the shortest path distance between two nodes in the grid graph
        using Breadth-First Search. Returns -1 if no path exists.
        """
        if start_node == end_node:
            return 0
        # Ensure nodes exist in the graph
        if start_node not in self.graph or end_node not in self.graph:
             return -1 # Should not happen in valid problems, but safe check

        queue = deque([(start_node, 0)])
        visited = {start_node}

        while queue:
            current_node, distance = queue.popleft()

            if current_node == end_node:
                return distance

            # Ensure current_node is still valid in graph (should be)
            if current_node in self.graph:
                for neighbor in self.graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))

        return -1 # No path found

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find current locations of robot and boxes
        robot_location = None
        box_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_location = parts[1]
            elif parts[0] == "at":
                 # Check if this 'at' fact is for a box that is a goal object
                 box = parts[1]
                 if box in self.goal_locations:
                    location = parts[2]
                    box_locations[box] = location

        # Check if goal is reached (all goal boxes are at their goal locations)
        all_boxes_at_goal = True
        for box, goal_loc in self.goal_locations.items():
            # If a goal box is not found in the current state or is not at its goal location
            if box not in box_locations or box_locations[box] != goal_loc:
                all_boxes_at_goal = False
                break

        if all_boxes_at_goal:
            return 0

        total_box_distance = 0
        boxes_to_move = []

        # Calculate sum of box-to-goal distances for boxes not at goal
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box)
            # Only consider boxes that are in the state and not at their goal
            if current_loc and current_loc != goal_loc:
                dist = self._bfs_distance(current_loc, goal_loc)
                if dist == -1:
                    # Box cannot reach its goal location
                    return math.inf
                total_box_distance += dist
                boxes_to_move.append(box) # Keep track of boxes that need moving

        # If there are no boxes to move but the goal check failed, this implies
        # a goal box is missing from the state or similar inconsistency.
        # Assuming valid states, boxes_to_move will be non-empty if not at goal.

        # Calculate robot distance to the nearest box that needs moving
        min_robot_distance = math.inf
        if robot_location and boxes_to_move:
            for box in boxes_to_move:
                box_loc = box_locations[box]
                dist = self._bfs_distance(robot_location, box_loc)
                if dist != -1: # Only consider reachable box locations
                    min_robot_distance = min(min_robot_distance, dist)

            # If robot cannot reach any box that needs moving, state is unsolvable
            if min_robot_distance == math.inf:
                 return math.inf
        elif boxes_to_move:
             # This case should ideally not be reached in a valid problem state
             # where boxes need moving but the robot location is unknown or
             # the robot cannot reach any box location. Treat as unsolvable.
             return math.inf
        # If boxes_to_move is empty, min_robot_distance remains inf, but we already returned 0.

        # The heuristic is the sum of total box movement effort and the robot's effort
        # to get to the nearest box.
        return total_box_distance + min_robot_distance
