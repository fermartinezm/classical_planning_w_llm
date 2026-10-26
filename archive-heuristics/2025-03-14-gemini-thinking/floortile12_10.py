from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all tiles according to the goal specification.
    It prioritizes painting actions and considers color changes and robot movements as necessary steps.
    The heuristic is not admissible but aims to be informative and efficiently computable for greedy best-first search.

    # Assumptions
    - The primary goal is to achieve all `painted` predicates in the goal state.
    - Movement actions are necessary to position the robot to paint tiles.
    - Color change actions are necessary if the robot does not have the required color.
    - We assume that for each tile to be painted, there is a path to reach an adjacent tile and a way to obtain the required color.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates, specifically focusing on `painted` goals.
    - It also extracts static information about tile adjacencies (up, down, left, right) to estimate movement costs.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal predicate `(painted tile color)` that is not satisfied in the current state:
    1. Check if any robot is currently at a tile adjacent to the target `tile` (up, down, left, or right) and `robot-has` the required `color`.
       - If yes, the estimated cost for this goal is 1 (for the paint action).
    2. If no robot is in a suitable position with the correct color, check if any robot `robot-has` the required `color`, regardless of its position.
       - If yes, the estimated cost is 2 (1 for moving to an adjacent tile + 1 for painting).
    3. If no robot has the required `color`, consider the color change action.
       - The estimated cost is 3 (1 for `change_color` + 1 for moving to an adjacent tile + 1 for painting).
    4. Sum up the estimated costs for all unsatisfied `painted` goal predicates.
    5. The final heuristic value is the total estimated cost. If all goal predicates are satisfied, the heuristic value is 0.
    """

    def __init__(self, task):
        """
        Initialize the floortile heuristic.

        - Extracts goal predicates related to 'painted' tiles.
        - Processes static facts to understand tile adjacencies (up, down, left, right), although not directly used in this simplified heuristic.
        """
        self.goals = task.goals
        self.painted_goals = [goal for goal in self.goals if match(goal, "painted", "*", "*")]
        self.static_facts = task.static


    def __call__(self, node):
        """
        Compute the heuristic value for a given state.

        The heuristic value is an estimate of the number of actions needed to reach the goal state from the current state.
        It focuses on satisfying the 'painted' goal predicates.
        """
        state = node.state
        heuristic_value = 0

        # Extract robot positions and held colors for efficient lookup
        robot_positions = {}
        robot_colors = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                parts = get_parts(fact)
                robot_positions[parts[1]] = parts[2] # robot -> tile
            if match(fact, "robot-has", "*", "*"):
                parts = get_parts(fact)
                robot_colors[parts[1]] = parts[2] # robot -> color

        for goal in self.painted_goals:
            if goal not in state:
                goal_parts = get_parts(goal)
                goal_tile = goal_parts[1]
                goal_color = goal_parts[2]
                min_cost_for_goal = float('inf')

                found_paint_action = False
                for robot, robot_tile in robot_positions.items():
                    robot_current_color = robot_colors.get(robot)
                    if robot_current_color == goal_color:
                        # Check adjacency (simplified adjacency check - assuming static facts provide adjacency)
                        for direction_predicate in ["up", "down", "left", "right"]:
                            if f'({direction_predicate} {goal_tile} {robot_tile})' in self.static_facts or f'({direction_predicate} {robot_tile} {goal_tile})' in self.static_facts:
                                min_cost_for_goal = min(min_cost_for_goal, 1) # Paint action
                                found_paint_action = True
                                break
                        if found_paint_action:
                            break # Found a robot that can paint from adjacent tile

                if not found_paint_action:
                    found_move_and_paint = False
                    for robot, robot_tile in robot_positions.items():
                        robot_current_color = robot_colors.get(robot)
                        if robot_current_color == goal_color:
                            min_cost_for_goal = min(min_cost_for_goal, 2) # Move + Paint
                            found_move_and_paint = True
                            break
                    if not found_move_and_paint:
                        min_cost_for_goal = min(min_cost_for_goal, 3) # Change color + Move + Paint

                heuristic_value += min_cost_for_goal if min_cost_for_goal != float('inf') else 1 # If no better estimate, assume at least 1 action needed

        return heuristic_value
