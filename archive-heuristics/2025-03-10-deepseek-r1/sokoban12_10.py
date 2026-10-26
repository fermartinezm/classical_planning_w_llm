from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict


class Sokoban12Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to solve a Sokoban problem by considering:
    1. The minimal number of pushes required for each box to reach its goal.
    2. The minimal number of moves required for the robot to reach the nearest box that needs pushing.

    # Assumptions
    - Each box must be pushed along a path defined by the adjacency relations in the correct direction.
    - The robot can move freely between adjacent locations as per the adjacency relations.
    - The heuristic does not account for boxes blocking each other or the robot's path, which is handled by the search.

    # Heuristic Initialization
    - Extracts adjacency relations from static facts to build a directed graph of possible movements.
    - Determines the goal location for each box from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Extract Current State Information**:
       - Identify the robot's current location.
       - Identify the current location of each box.
    2. **Box Push Calculation**:
       - For each box not at its goal, compute the shortest path (number of pushes) to its goal using BFS on the directed adjacency graph.
    3. **Robot Movement Calculation**:
       - For each box not at its goal, compute the shortest path for the robot to reach the box's current location.
       - Track the minimal robot movement steps required to get adjacent to a box (distance - 1).
    4. **Combine Results**:
       - Sum the total pushes needed for all boxes.
       - Add the minimal robot movement steps to the sum to get the heuristic value.
    """

    def __init__(self, task):
        self.task = task
        self.adjacency = defaultdict(list)
        self.box_goals = {}

        # Build adjacency graph from static facts
        for fact in task.static:
            parts = fact.strip('()').split()
            if parts[0] == 'adjacent':
                from_loc = parts[1]
                to_loc = parts[2]
                self.adjacency[from_loc].append(to_loc)

        # Extract goal locations for each box
        for goal in task.goals:
            parts = goal.strip('()').split()
            if parts[0] == 'at' and parts[1].startswith('box'):
                box = parts[1]
                loc = parts[2]
                self.box_goals[box] = loc

    def _bfs(self, start, goal):
        """Helper function to compute shortest path in directed adjacency graph using BFS."""
        if start == goal:
            return 0
        visited = set()
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            if node == goal:
                return dist
            if node in visited:
                continue
            visited.add(node)
            for neighbor in self.adjacency.get(node, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return float('inf')  # No path found

    def __call__(self, node):
        state = node.state
        robot_loc = None
        box_locs = {}

        # Extract robot's current location
        for fact in state:
            if fnmatch(fact, '(at-robot *)'):
                parts = fact.strip('()').split()
                robot_loc = parts[1]
                break

        # Extract current locations of all boxes
        for fact in state:
            if fnmatch(fact, '(at box* *'):
                parts = fact.strip('()').split()
                box = parts[1]
                loc = parts[2]
                box_locs[box] = loc

        sum_pushes = 0
        min_robot_steps = float('inf')

        for box, goal_loc in self.box_goals.items():
            current_loc = box_locs.get(box)
            if not current_loc or current_loc == goal_loc:
                continue  # Skip if box is already at goal

            # Calculate minimal pushes for this box
            pushes = self._bfs(current_loc, goal_loc)
            if pushes == float('inf'):
                return float('inf')  # Unsolvable state
            sum_pushes += pushes

            # Calculate robot steps to reach this box
            robot_dist = self._bfs(robot_loc, current_loc)
            if robot_dist == float('inf'):
                return float('inf')
            robot_steps = max(0, robot_dist - 1)  # Steps to get adjacent
            if robot_steps < min_robot_steps:
                min_robot_steps = robot_steps

        if sum_pushes == 0:
            return 0  # All boxes are at goals

        if min_robot_steps == float('inf'):
            return sum_pushes  # Edge case, no boxes need pushing

        return sum_pushes + min_robot_steps
