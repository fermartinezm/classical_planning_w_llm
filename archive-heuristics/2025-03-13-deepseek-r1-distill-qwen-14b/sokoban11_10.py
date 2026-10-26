from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to push all boxes to their target locations.

    # Assumptions:
    - The robot can push a box by moving into its position if the next cell is clear.
    - Each box must be pushed individually, requiring the robot to move to its position.
    - The heuristic considers both the distance each box needs to move and the robot's movement to reach the farthest box.

    # Heuristic Initialization
    - Extract the goal locations for each box from the task's goals.
    - Identify static facts, particularly the adjacency relationships to compute distances.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box, determine its current position and target position.
    2. Calculate the Manhattan distance from the box's current position to its target.
    3. Sum these distances for all boxes that are not already at their target.
    4. Determine the box that is farthest from the robot's current position.
    5. Add the Manhattan distance from the robot to this farthest box to the total.
    6. The sum represents the estimated number of actions needed.
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

        # Precompute adjacency for efficient distance calculation
        self.adjacent = {}
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'adjacent':
                l1, l2, dir = parts[1], parts[2], parts[3]
                if l1 not in self.adjacent:
                    self.adjacent[l1] = []
                self.adjacent[l1].append(l2)
                if l2 not in self.adjacent:
                    self.adjacent[l2] = []
                self.adjacent[l2].append(l1)

    def manhattan_distance(self, loc1, loc2):
        """Compute Manhattan distance between two locations."""
        # Assuming locations are in grid format like loc_1_1, loc_2_3, etc.
        x1, y1 = int(loc1.split('_')[1]), int(loc1.split('_')[2])
        x2, y2 = int(loc2.split('_')[1]), int(loc2.split('_')[2])
        return abs(x1 - x2) + abs(y1 - y2)

    def __call__(self, node):
        """Estimate the minimum cost to push all boxes to their target locations."""
        state = node.state

        # Extract current positions of boxes and robot
        current_box_positions = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at' and parts[1].startswith('box'):
                box = parts[1]
                loc = parts[2]
                current_box_positions[box] = loc

        robot_pos = None
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at-robot':
                robot_pos = parts[1]

        if not robot_pos:
            # Robot position not found, which shouldn't happen in valid states
            return 0

        total_distance = 0
        max_robot_distance = 0
        farthest_box = None

        for box in self.goal_locations:
            goal = self.goal_locations[box]
            current = current_box_positions.get(box, None)
            if current is None:
                # Box not present, perhaps already delivered
                continue
            if current == goal:
                continue  # No need to move
            # Compute Manhattan distance
            distance = self.manhattan_distance(current, goal)
            total_distance += distance
            # Compute distance from robot to this box
            robot_to_box = self.manhattan_distance(robot_pos, current)
            if robot_to_box > max_robot_distance:
                max_robot_distance = robot_to_box
                farthest_box = box

        # If there are boxes to move, add the max robot distance
        if farthest_box is not None:
            total_distance += max_robot_distance

        return total_distance
