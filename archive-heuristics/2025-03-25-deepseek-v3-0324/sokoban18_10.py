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
    This heuristic estimates the number of actions needed to solve a Sokoban puzzle by:
    1. Calculating the Manhattan distance from each box to its nearest goal position
    2. Calculating the Manhattan distance from the robot to each box
    3. Adding these distances with appropriate weights to account for pushing costs

    # Assumptions:
    - Each box has exactly one goal position (though multiple boxes can share the same goal)
    - The robot can only push one box at a time
    - Moving without pushing a box is generally cheaper than pushing
    - Pushing a box in the correct direction is better than pushing it away

    # Heuristic Initialization
    - Extract goal positions for all boxes from the task goals
    - Build an adjacency graph from the static 'adjacent' facts for pathfinding
    - Identify all clear locations (where boxes or robot can move)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box that's not already at its goal:
        a) Calculate the Manhattan distance from the box to its goal
        b) Multiply by a weight (2) to account for needing to push it
    2. For each box that's not at its goal:
        a) Calculate the Manhattan distance from the robot to the box
        b) Use the minimum of these distances
    3. Sum the box-to-goal distances and add the robot-to-box distance
    4. If all boxes are at their goals, just return the distance from robot to nearest goal
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

        # Build adjacency graph
        self.adjacency = {}
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                if loc1 not in self.adjacency:
                    self.adjacency[loc1] = set()
                self.adjacency[loc1].add(loc2)
                if loc2 not in self.adjacency:
                    self.adjacency[loc2] = set()
                self.adjacency[loc2].add(loc1)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Get current positions of boxes and robot
        box_positions = {}
        robot_pos = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_pos = get_parts(fact)[1]
            elif match(fact, "at", "*", "*"):
                _, box, loc = get_parts(fact)
                box_positions[box] = loc

        # If all boxes are at goals, return distance to nearest goal
        all_boxes_at_goals = all(
            box_positions.get(box) == goal_loc
            for box, goal_loc in self.box_goals.items()
        )
        if all_boxes_at_goals:
            return self._min_distance(robot_pos, set(self.box_goals.values()))

        total_cost = 0

        # Calculate box-to-goal distances
        box_to_goal = 0
        for box, goal_loc in self.box_goals.items():
            if box in box_positions and box_positions[box] != goal_loc:
                box_to_goal += 2 * self._manhattan_distance(box_positions[box], goal_loc)

        # Calculate robot-to-box distance (minimum)
        robot_to_box = min(
            self._manhattan_distance(robot_pos, box_loc)
            for box_loc in box_positions.values()
            if box_loc not in self.box_goals.values() or box_positions[box] != self.box_goals[box]
        ) if box_positions else 0

        total_cost = box_to_goal + robot_to_box

        return total_cost

    def _manhattan_distance(self, loc1, loc2):
        """Calculate Manhattan distance between two locations."""
        if loc1 == loc2:
            return 0
        try:
            _, x1, y1 = loc1.split('_')
            _, x2, y2 = loc2.split('_')
            return abs(int(x1) - int(x2)) + abs(int(y1) - int(y2))
        except:
            # Fallback for non-standard location names
            return 1

    def _min_distance(self, start, goals):
        """Calculate minimum Manhattan distance from start to any goal."""
        if not goals:
            return 0
        return min(self._manhattan_distance(start, goal) for goal in goals)
