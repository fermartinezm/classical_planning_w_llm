from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

class Sokoban7Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to push all boxes to their goal positions. It calculates the shortest path for each box to its goal and the robot's shortest path to each box, adjusting for the need to be adjacent to push.

    # Assumptions:
    - The static adjacency facts define the possible movements between locations.
    - The shortest path between locations is computed using BFS on the adjacency graph.
    - The robot must be adjacent to a box to push it, requiring one less move than the path length.

    # Heuristic Initialization
    - Extract the goal locations for each box from the task's goals.
    - Build an adjacency graph from the static facts to determine possible movements.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box, compute the shortest path from its current location to its goal location using BFS.
    2. For each box, compute the shortest path from the robot's current location to the box's current location using BFS.
    3. Adjust the robot's path length by subtracting one (since the robot needs to be adjacent to push).
    4. Sum the box path lengths and add the minimum adjusted robot path length to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goals and adjacency information."""
        self.goal_locations = {}
        self.adjacency = defaultdict(list)

        # Extract goal locations for each box
        for goal in task.goals:
            parts = self._get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                box = parts[1]
                location = parts[2]
                self.goal_locations[box] = location

        # Extract adjacency relationships from static facts
        for fact in task.static:
            parts = self._get_parts(fact)
            if parts[0] == 'adjacent':
                loc_from = parts[1]
                loc_to = parts[2]
                self.adjacency[loc_from].append(loc_to)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        current_boxes = {}
        robot_location = None

        # Extract current positions of boxes and the robot
        for fact in state:
            parts = self._get_parts(fact)
            if parts[0] == 'at' and parts[1].startswith('box'):
                box = parts[1]
                current_boxes[box] = parts[2]
            elif parts[0] == 'at-robot':
                robot_location = parts[1]

        if not robot_location:
            return float('inf')  # Invalid state if robot location is missing

        sum_pushes = 0
        robot_steps_candidates = []

        for box, current_loc in current_boxes.items():
            goal_loc = self.goal_locations.get(box)
            if not goal_loc or current_loc == goal_loc:
                continue  # Skip boxes at their goal or without a defined goal

            # Calculate shortest path for the box to its goal
            box_distance = self._shortest_path(current_loc, goal_loc)
            if box_distance == float('inf'):
                return float('inf')  # Unreachable goal
            sum_pushes += box_distance

            # Calculate shortest path for the robot to the box's current position
            robot_distance = self._shortest_path(robot_location, current_loc)
            if robot_distance == float('inf'):
                return float('inf')  # Unreachable box
            robot_steps_candidates.append(max(robot_distance - 1, 0))

        if not robot_steps_candidates:
            return 0  # All boxes are at their goals
        else:
            return sum_pushes + min(robot_steps_candidates)

    def _shortest_path(self, start, end):
        """Compute the shortest path between two locations using BFS."""
        if start == end:
            return 0
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
        return float('inf')  # No path found

    @staticmethod
    def _get_parts(fact):
        """Split a PDDL fact into its components."""
        return fact[1:-1].split()
