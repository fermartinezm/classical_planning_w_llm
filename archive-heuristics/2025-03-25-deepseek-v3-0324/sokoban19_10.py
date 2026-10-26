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
    1. The Manhattan distance from each box to its goal position.
    2. The Manhattan distance from the robot to each box.
    3. A penalty for boxes that are not adjacent to clear spaces in the direction of their goal.

    # Assumptions:
    - Each box has exactly one goal position.
    - The grid is rectangular and coordinates follow the pattern loc_X_Y.
    - Pushing a box always requires moving the robot to an adjacent position first.

    # Heuristic Initialization
    - Extract goal positions for boxes from the task goals.
    - Parse static adjacency information into a graph structure.
    - Store all clear locations that can be moved to.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not at its goal:
        a) Calculate Manhattan distance from current position to goal.
        b) Calculate Manhattan distance from robot to box.
        c) Check if box is blocked (no clear space in direction toward goal).
    2. Sum these components with appropriate weights:
        - Box-to-goal distance is most important (direct progress).
        - Robot-to-box distance accounts for movement needed before pushing.
        - Blocked boxes get an additional penalty to encourage freeing them.
    3. The total heuristic is the sum of these weighted components.
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
                    self.adjacency[loc1] = []
                self.adjacency[loc1].append(loc2)

    def __call__(self, node):
        """Compute heuristic estimate for the given state."""
        state = node.state

        # Get current positions
        robot_pos = None
        box_positions = {}
        clear_locations = set()

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at-robot", "*"):
                robot_pos = parts[1]
            elif match(fact, "at", "*", "*"):
                box = parts[1]
                loc = parts[2]
                box_positions[box] = loc
            elif match(fact, "clear", "*"):
                clear_locations.add(parts[1])

        # If all boxes are at goals, heuristic is 0
        if all(box_positions.get(box) == goal_loc 
               for box, goal_loc in self.box_goals.items()):
            return 0

        total_cost = 0

        for box, goal_loc in self.box_goals.items():
            current_loc = box_positions.get(box)
            if current_loc == goal_loc:
                continue

            # Parse coordinates from location names
            try:
                _, curr_x, curr_y = current_loc.split('_')
                curr_x, curr_y = int(curr_x), int(curr_y)
                _, goal_x, goal_y = goal_loc.split('_')
                goal_x, goal_y = int(goal_x), int(goal_y)
            except (ValueError, AttributeError):
                # Fallback if location names don't follow expected pattern
                box_dist = 1
                robot_dist = 1
                is_blocked = 0
            else:
                # Calculate Manhattan distance from box to goal
                box_dist = abs(curr_x - goal_x) + abs(curr_y - goal_y)

                # Calculate Manhattan distance from robot to box
                if robot_pos:
                    _, robot_x, robot_y = robot_pos.split('_')
                    robot_x, robot_y = int(robot_x), int(robot_y)
                    robot_dist = abs(robot_x - curr_x) + abs(robot_y - curr_y)
                else:
                    robot_dist = 0

                # Check if box is blocked in direction toward goal
                is_blocked = 0
                if curr_x < goal_x:
                    push_dir = 'right'
                elif curr_x > goal_x:
                    push_dir = 'left'
                elif curr_y < goal_y:
                    push_dir = 'down'
                else:
                    push_dir = 'up'

                # Find location the box would be pushed from
                push_from = None
                for adj in self.adjacency.get(current_loc, []):
                    adj_parts = adj.split('_')
                    adj_x, adj_y = int(adj_parts[1]), int(adj_parts[2])
                    if (push_dir == 'right' and adj_x < curr_x) or \
                       (push_dir == 'left' and adj_x > curr_x) or \
                       (push_dir == 'down' and adj_y < curr_y) or \
                       (push_dir == 'up' and adj_y > curr_y):
                        push_from = adj
                        break

                if push_from and push_from not in clear_locations:
                    is_blocked = 1

            # Weighted sum of components
            total_cost += 3 * box_dist + robot_dist + 5 * is_blocked

        return total_cost
