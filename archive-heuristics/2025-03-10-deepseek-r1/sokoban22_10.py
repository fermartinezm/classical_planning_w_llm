from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


class sokoban22Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    Estimates the number of actions needed to push all boxes to their goals by summing the minimal pushes for each box and the robot's distance to the nearest box.

    # Assumptions
    - Each box's minimal pushes are the shortest path from current to goal location.
    - The robot's movement to the nearest box is added to the total estimate.
    - Static adjacency facts define the movement graph for shortest path calculations.

    # Heuristic Initialization
    1. Extract box goals from the task's goal conditions.
    2. Build an adjacency graph from static 'adjacent' facts.
    3. Precompute shortest paths between all locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not at its goal, add the precomputed shortest path distance to its goal.
    2. Compute the robot's shortest path to the nearest box needing movement.
    3. Sum these values to get the heuristic estimate.
    """

    def __init__(self, task):
        self.box_goals = {}
        # Extract box goals from task's goals
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                box = parts[1]
                location = parts[2]
                self.box_goals[box] = location

        # Build adjacency graph from static 'adjacent' facts
        self.adjacency = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                from_loc, to_loc = parts[1], parts[2]
                self.adjacency.setdefault(from_loc, set()).add(to_loc)
                self.adjacency.setdefault(to_loc, set()).add(from_loc)  # Bidirectional

        # Precompute shortest paths between all locations using BFS
        self.distance = {}
        all_locations = set(self.adjacency.keys())
        for loc in all_locations:
            self.distance[loc] = self.bfs(loc)

    def bfs(self, start):
        """Compute shortest paths from start to all reachable locations using BFS."""
        distances = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        state = node.state
        current_boxes = {}
        robot_loc = None

        # Extract current state information
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.box_goals:
                    current_boxes[obj] = loc
            elif parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]

        # Check if all boxes are at their goals
        all_goals_reached = True
        total_box_cost = 0
        for box, current in current_boxes.items():
            goal = self.box_goals.get(box)
            if not goal or current != goal:
                all_goals_reached = False
                # Use precomputed distance or a large number if unreachable
                total_box_cost += self.distance.get(current, {}).get(goal, float('inf'))

        if all_goals_reached:
            return 0

        # Find minimal robot distance to any box not at goal
        min_robot_dist = float('inf')
        if robot_loc is not None:
            for box, current in current_boxes.items():
                if current != self.box_goals.get(box):
                    dist = self.distance.get(robot_loc, {}).get(current, float('inf'))
                    if dist < min_robot_dist:
                        min_robot_dist = dist

        # Ensure finite heuristic for solvable states
        min_robot_dist = min_robot_dist if min_robot_dist != float('inf') else 0
        return total_box_cost + min_robot_dist
