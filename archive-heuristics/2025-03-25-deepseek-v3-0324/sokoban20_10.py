from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to push all boxes to their goal positions.
    It combines:
    1. The Manhattan distance from each box to its goal position
    2. The Manhattan distance from the robot to each box
    3. A penalty for boxes that are not on goal positions but blocking potential paths

    # Assumptions:
    - Each box has exactly one goal position (standard Sokoban).
    - The grid is rectangular and coordinates follow the pattern loc_X_Y.
    - Only one box exists per location (no stacking).

    # Heuristic Initialization
    - Extract goal positions for boxes from the task goals.
    - Build an adjacency graph from static facts to enable pathfinding.
    - Store clear locations to identify obstacles.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not at its goal:
        a) Calculate Manhattan distance from box to goal (minimum pushes needed).
        b) Find the robot's path to the box (minimum moves needed to reach pushing position).
        c) Add these costs together.
    2. For boxes already at goals, no cost is added.
    3. The total heuristic is the sum of costs for all boxes plus any additional penalties.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal positions for boxes
        self.box_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, box, loc = get_parts(goal)
                self.box_goals[box] = loc

        # Build adjacency graph for pathfinding
        self.adjacency = {}
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                self.adjacency.setdefault(loc1, set()).add(loc2)
                self.adjacency.setdefault(loc2, set()).add(loc1)

    def _manhattan_distance(self, loc1, loc2):
        """Calculate Manhattan distance between two locations."""
        x1, y1 = map(int, loc1.split('_')[1:])
        x2, y2 = map(int, loc2.split('_')[1:])
        return abs(x1 - x2) + abs(y1 - y2)

    def _get_robot_position(self, state):
        """Find the robot's current position from the state."""
        for fact in state:
            if match(fact, "at-robot", "*"):
                return get_parts(fact)[1]
        return None

    def _get_box_positions(self, state):
        """Get current positions of all boxes."""
        boxes = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, box, loc = get_parts(fact)
                boxes[box] = loc
        return boxes

    def _shortest_path_length(self, start, end, obstacles):
        """
        BFS to find shortest path length between start and end.
        obstacles is a set of locations that cannot be passed through.
        """
        if start == end:
            return 0

        visited = {start}
        queue = [(start, 0)]

        while queue:
            current, dist = queue.pop(0)
            for neighbor in self.adjacency.get(current, []):
                if neighbor == end:
                    return dist + 1
                if neighbor not in visited and neighbor not in obstacles:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return float('inf')  # No path found

    def __call__(self, node):
        """Compute heuristic value for the given state."""
        state = node.state
        if self.goals <= state:  # Goal reached
            return 0

        robot_pos = self._get_robot_position(state)
        box_positions = self._get_box_positions(state)
        obstacles = set(box_positions.values())  # Boxes block movement

        total_cost = 0

        for box, goal_loc in self.box_goals.items():
            current_loc = box_positions.get(box)
            if current_loc == goal_loc:
                continue  # Box already at goal

            # Cost to push box to goal (Manhattan distance)
            push_cost = self._manhattan_distance(current_loc, goal_loc)

            # Cost for robot to reach pushing position
            # The robot needs to be adjacent to the box on the path toward the goal
            # We approximate this with the shortest path to any adjacent cell
            min_robot_cost = float('inf')
            for direction in ['up', 'down', 'left', 'right']:
                # Find the position the robot needs to be in to push toward goal
                # This is simplified - in reality would need to consider push direction
                for neighbor in self.adjacency.get(current_loc, []):
                    path_cost = self._shortest_path_length(robot_pos, neighbor, obstacles)
                    if path_cost < min_robot_cost:
                        min_robot_cost = path_cost

            if min_robot_cost == float('inf'):
                min_robot_cost = 10  # Large penalty if no path found

            total_cost += push_cost + min_robot_cost

        return total_cost
