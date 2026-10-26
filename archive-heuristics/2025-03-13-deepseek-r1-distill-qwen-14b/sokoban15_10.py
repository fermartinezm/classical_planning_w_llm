from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes to their target locations.

    # Assumptions:
    - The robot can move to adjacent locations and push boxes.
    - Each box needs to be moved to a specific target location.
    - The robot may need to move to the location of a box before pushing it.

    # Heuristic Initialization
    - Extract the goal locations for each box and static facts (adjacent locations) from the task.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Identify the current location of the robot and each box.
    2. For each box that needs to be moved:
        a. Calculate the Manhattan distance from the robot to the box.
        b. Calculate the Manhattan distance from the box to its target location.
        c. Sum these distances for the total cost.
    3. If the robot is not adjacent to a box, add the distance to move to the box's location.
    4. Sum the costs for all boxes to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, box, loc = get_parts(goal)
            if predicate == "at":
                self.goal_locations[box] = loc

        # Precompute adjacent locations for efficient lookup
        self.adjacent = {}
        for fact in self.static:
            if fact.startswith("(adjacent "):
                parts = get_parts(fact)
                loc1, loc2, dir = parts
                if loc1 not in self.adjacent:
                    self.adjacent[loc1] = []
                self.adjacent[loc1].append(loc2)
                if loc2 not in self.adjacent:
                    self.adjacent[loc2] = []
                self.adjacent[loc2].append(loc1)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract current locations
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at-robot":
                current_locations["robot"] = args[0]
            elif predicate == "at":
                current_locations[args[1]] = args[0]  # box at location

        total_cost = 0

        # For each box that needs to be moved
        for box in self.goal_locations:
            goal_loc = self.goal_locations[box]
            if box not in current_locations:
                continue  # box is already at goal

            current_loc = current_locations[box]
            if current_loc == goal_loc:
                continue  # box is already in place

            # Calculate distance from robot to current location of the box
            robot_loc = current_locations.get("robot", None)
            if robot_loc is None:
                continue  # robot location not found

            # Manhattan distance between robot and box
            robot_x, robot_y = self.parse_loc(robot_loc)
            box_x, box_y = self.parse_loc(current_loc)
            dist_robot_to_box = abs(robot_x - box_x) + abs(robot_y - box_y)

            # Manhattan distance between box and goal
            goal_x, goal_y = self.parse_loc(goal_loc)
            dist_box_to_goal = abs(box_x - goal_x) + abs(box_y - goal_y)

            # Total distance for this box
            total_cost += dist_robot_to_box + dist_box_to_goal

            # If the robot needs to move to the box's location first
            if dist_robot_to_box > 0:
                total_cost += 1  # move to box's location

        return total_cost

    def parse_loc(self, loc):
        """Parse a location string (e.g., 'loc_3_4') into (x, y) coordinates."""
        parts = loc.split('_')
        return int(parts[1]), int(parts[2])

    def get_parts(self, fact):
        """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
        return fact[1:-1].split()

    def match(self, fact, *args):
        """
        Check if a PDDL fact matches a given pattern.

        - `fact`: The complete fact as a string, e.g., "(at-robot loc_3_4)".
        - `args`: The expected pattern (wildcards `*` allowed).
        - Returns `True` if the fact matches the pattern, else `False`.
        """
        parts = self.get_parts(fact)
        return all(fnmatch(part, arg) for part, arg in zip(parts, args))
