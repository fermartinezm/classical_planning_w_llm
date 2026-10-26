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
    This heuristic estimates the number of actions needed to move all boxes to their target locations.

    # Assumptions:
    - The robot can move to adjacent locations and push boxes.
    - Each box must be moved to a specific target location.
    - The robot must be adjacent to a box to push it.

    # Heuristic Initialization
    - Extract the target locations for each box from the goal state.
    - Extract static facts, particularly the adjacency relationships between locations.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. For each box, determine its current location and target location.
    2. If the box is already at the target location, no actions are needed.
    3. If the box is not at the target location, calculate the Manhattan distance between the box's current location and target location.
    4. Add the number of moves required for the robot to reach the box's current location from its current position.
    5. For each box that needs to be moved, add the calculated distance plus the moves required for the robot to position itself correctly to push the box.
    6. Sum these values for all boxes to get the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract adjacency information for quick lookup
        self.adjacent = {}
        for fact in self.static:
            if match(fact, "adjacent", "*", "*", "*"):
                loc1, loc2, dir = get_parts(fact)
                if loc1 not in self.adjacent:
                    self.adjacent[loc1] = {}
                self.adjacent[loc1][loc2] = dir
                if loc2 not in self.adjacent:
                    self.adjacent[loc2] = {}
                self.adjacent[loc2][loc1] = dir

        # Extract goal locations for each box
        self.box_goals = {}
        for goal in self.goals:
            predicate, box, loc = get_parts(goal)
            if predicate == "at":
                self.box_goals[box] = loc

    def __call__(self, node):
        """Estimate the minimum number of actions to reach the goal state."""
        state = node.state

        # Extract current locations of boxes and the robot
        current_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                if parts[1] == "box":
                    current_locations[parts[2]] = parts[1]  # Store as 'box loc'
                elif parts[1] == "at-robot":
                    current_locations["robot"] = parts[2]
            elif parts[0] == "at-robot":
                current_locations["robot"] = parts[1]

        # If all boxes are already at their goals, return 0
        if all(fact in state for fact in self.goals):
            return 0

        total_cost = 0

        # For each box, calculate the cost to move it to its target location
        for box, target in self.box_goals.items():
            # Check if the box is already at the target
            if f"at box{box} {target}" in state:
                continue

            # Find current location of the box
            current_box_loc = None
            for fact in state:
                if fact.startswith(f"(at box{box} "):
                    current_box_loc = fact.split()[2]
                    break

            if current_box_loc is None:
                continue  # Box is not present, shouldn't happen in Sokoban

            # Calculate Manhattan distance between current and target locations
            # Extracting coordinates from location strings (e.g., "loc_1_2" -> (1,2))
            def parse_loc(loc_str):
                x, y = loc_str.split('_')[1:]
                return (int(x), int(y))

            current_x, current_y = parse_loc(current_box_loc)
            target_x, target_y = parse_loc(target)

            distance = abs(current_x - target_x) + abs(current_y - target_y)

            # Add the distance to the total cost
            total_cost += distance

            # Add the cost for the robot to reach the box's current location
            # Find robot's current location
            robot_loc = current_locations.get("robot", None)
            if robot_loc:
                robot_x, robot_y = parse_loc(robot_loc)
                moves_to_box = abs(robot_x - current_x) + abs(robot_y - current_y)
                total_cost += moves_to_box

            # Add the cost to push the box (1 move to adjacent cell and 1 push action)
            total_cost += 2  # One move to position and one push action

        return total_cost
