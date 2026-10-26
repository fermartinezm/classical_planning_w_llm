from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move each box to its target location. It considers both the robot's movement to reach each box and the box's movement to its target.

    # Assumptions:
    - Each box must be moved to a specific target location.
    - The robot can move to adjacent clear locations.
    - Boxes can be pushed into adjacent clear locations.
    - The heuristic does not account for blockages by other boxes.

    # Heuristic Initialization
    - Extract the target location for each box from the goal state.
    - Build an adjacency graph from static facts to compute distances.

    # Step-by-Step Thinking for Computing Heuristic
    1. For each box, check if it is already at its target location. If so, no actions are needed for that box.
    2. For each box not at its target:
       a. Compute the shortest path for the robot to reach the box's current location.
       b. Compute the shortest path from the box's current location to its target location.
       c. Sum these distances and add one action for pushing the box.
    3. Sum the values for all boxes to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'at' and parts[1].startswith('box'):
                box = parts[1]
                loc = parts[2]
                self.goal_locations[box] = loc

        # Build adjacency graph from static facts
        self.adjacency = {}
        for fact in self.static:
            if fact.startswith('(adjacent'):
                parts = fact[1:-1].split()
                loc1, loc2, dir = parts[1], parts[2], parts[3]
                if loc1 not in self.adjacency:
                    self.adjacency[loc1] = []
                self.adjacency[loc1].append(loc2)
                if loc2 not in self.adjacency:
                    self.adjacency[loc2] = []
                self.adjacency[loc2].append(loc1)

    def __call__(self, node):
        """Estimate the minimum number of actions to reach the goal state."""
        state = node.state
        current_boxes = {}
        robot_location = None

        # Extract current locations of boxes and robot
        for fact in state:
            if fact.startswith('(at-robot'):
                robot_location = fact[1:-1].split()[1]
            elif fact.startswith('(at'):
                parts = fact[1:-1].split()
                if parts[1].startswith('box'):
                    box = parts[1]
                    loc = parts[2]
                    current_boxes[box] = loc

        # If robot location is not found, assume it's at some location (should not happen in valid state)
        if robot_location is None:
            return 0

        total_cost = 0

        # For each box, calculate the required actions
        for box, target in self.goal_locations.items():
            current_loc = current_boxes.get(box, None)
            if current_loc is None:
                continue  # Box not present in state (should not happen in valid state)

            if current_loc == target:
                continue  # Box is already at target

            # Compute distance from robot to current_loc
            robot_to_box = self.bfs(robot_location, current_loc)
            if robot_to_box is None:
                continue  # Should not happen in valid state

            # Compute distance from current_loc to target
            box_to_target = self.bfs(current_loc, target)
            if box_to_target is None:
                continue  # Should not happen in valid state

            # Total actions: robot moves to box (distance), then pushes box to target (distance + 1)
            total_cost += robot_to_box + box_to_target + 1

        return total_cost

    def bfs(self, start, end):
        """Compute the shortest path distance using BFS."""
        visited = set()
        queue = deque([(start, 0)])

        while queue:
            current, dist = queue.popleft()
            if current == end:
                return dist
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return None  # No path found (should not happen in valid Sokoban state)
