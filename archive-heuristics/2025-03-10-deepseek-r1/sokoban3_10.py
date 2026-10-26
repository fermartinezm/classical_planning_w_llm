from collections import deque
from heuristics.heuristic_base import Heuristic

class sokoban3Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to move all boxes to their goal positions. It calculates the sum of the robot's distance to each box plus the box's distance to its goal, using precomputed shortest paths in the maze.

    # Assumptions
    - The maze layout is static and defined by the 'adjacent' predicates.
    - Each box's movement requires the robot to first reach the box and then push it to the goal.
    - The shortest path between locations (ignoring other boxes) is used for estimates.

    # Heuristic Initialization
    - Extract goal locations for each box from the task's goals.
    - Build an adjacency graph from static 'adjacent' facts.
    - Precompute shortest paths between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not at its goal:
        a. Compute the robot's current distance to the box's location.
        b. Compute the box's shortest path distance to its goal.
        c. Sum these distances for all boxes to get the heuristic value.
    2. If all boxes are at their goals, the heuristic returns 0.
    """

    def __init__(self, task):
        """Initialize the heuristic with static data and precompute distances."""
        self.box_goals = {}
        for goal in task.goals:
            parts = goal.strip('()').split()
            if parts[0] == 'at' and len(parts) == 3:
                box = parts[1]
                loc = parts[2]
                self.box_goals[box] = loc

        self.adjacency = {}
        for fact in task.static:
            parts = fact.strip('()').split()
            if parts[0] == 'adjacent':
                l1, l2 = parts[1], parts[2]
                if l1 not in self.adjacency:
                    self.adjacency[l1] = []
                self.adjacency[l1].append(l2)
                if l2 not in self.adjacency:
                    self.adjacency[l2] = []
                self.adjacency[l2].append(l1)

        self.distances = {}
        for loc in self.adjacency:
            self.distances[loc] = {}
            queue = deque([loc])
            self.distances[loc][loc] = 0
            while queue:
                current = queue.popleft()
                for neighbor in self.adjacency.get(current, []):
                    if neighbor not in self.distances[loc]:
                        self.distances[loc][neighbor] = self.distances[loc][current] + 1
                        queue.append(neighbor)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        current_robot = None
        current_boxes = {}

        for fact in node.state:
            parts = fact.strip('()').split()
            if parts[0] == 'at-robot':
                current_robot = parts[1]
            elif parts[0] == 'at' and parts[1] in self.box_goals:
                current_boxes[parts[1]] = parts[2]

        if not current_robot:
            return float('inf')

        total = 0
        for box, goal_loc in self.box_goals.items():
            current_loc = current_boxes.get(box)
            if current_loc == goal_loc:
                continue

            robot_dist = self.distances.get(current_robot, {}).get(current_loc, float('inf'))
            box_dist = self.distances.get(current_loc, {}).get(goal_loc, float('inf'))
            total += robot_dist + box_dist

        return total
