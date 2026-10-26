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
    This heuristic estimates the minimum number of actions required to achieve the goal state in the floortile domain.
    It focuses on the number of tiles that are not yet painted with the desired color and estimates the cost of painting them,
    considering the robot's current color and position.

    # Assumptions
    - The heuristic assumes that for each unpainted tile in the goal, at least one paint action is needed.
    - It also considers the necessity of color change actions if the robot does not have the required color.
    - It simplifies movement cost by assuming that if the robot is not at an adjacent tile, one move action is sufficient to get closer.
    - It does not explicitly plan the sequence of moves or color changes, but rather estimates the minimum actions needed.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates and static facts from the task.
    - It extracts the goal painted tiles and their colors.
    - It also pre-processes the static facts to efficiently check for adjacency relations (up, down, left, right) between tiles.

    # Step-By-Step Thinking for Computing Heuristic
    For each goal condition `(painted tile_goal color_goal)`:
    1. Check if the goal condition `(painted tile_goal color_goal)` is already satisfied in the current state.
       If yes, no cost is added for this goal.
    2. If the goal condition is not satisfied:
       - Increment the heuristic cost by 1, assuming at least one 'paint' action is needed.
       - Check if the robot `robot1` currently holds the required `color_goal`.
         If not, increment the heuristic cost by 1, assuming a 'change_color' action is needed.
       - Determine the current location `tile_robot` of the robot `robot1`.
       - Check if `tile_goal` is adjacent to `tile_robot` (up, down, left, or right) using the pre-processed static adjacency facts.
         If not adjacent, increment the heuristic cost by 1, assuming at least one 'move' action is needed to get closer to `tile_goal`.
    3. The total heuristic value is the sum of the costs calculated for each unsatisfied goal condition.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.adjacency = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right']:
                relation, tile1, tile2 = parts
                if tile1 not in self.adjacency:
                    self.adjacency[tile1] = {}
                self.adjacency[tile1][relation] = tile2

        self.goal_painted_tiles = {}
        for goal in self.goals:
            if match(goal, 'painted', '*', '*'):
                parts = get_parts(goal)
                tile_name = parts[1]
                color_name = parts[2]
                self.goal_painted_tiles[tile_name] = color_name

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        """
        state = node.state
        heuristic_value = 0

        # Get robot's current location and color
        robot_location = None
        robot_color = None
        for fact in state:
            if match(fact, 'robot-at', 'robot1', '*'):
                robot_location = get_parts(fact)[2]
            if match(fact, 'robot-has', 'robot1', '*'):
                robot_color = get_parts(fact)[2]

        for tile_goal, color_goal in self.goal_painted_tiles.items():
            goal_fact = f'(painted {tile_goal} {color_goal})'
            if goal_fact not in state:
                heuristic_value += 1  # For paint action

                if robot_color != color_goal:
                    heuristic_value += 1  # For change_color action

                if robot_location:
                    is_adjacent = False
                    for direction in ['up', 'down', 'left', 'right']:
                        if robot_location in self.adjacency and direction in self.adjacency[robot_location] and self.adjacency[robot_location][direction] == tile_goal:
                            is_adjacent = True
                            break
                        for rev_direction in ['up', 'down', 'left', 'right']:
                            rev_relation = None
                            if rev_direction == 'up': rev_relation = 'down'
                            if rev_direction == 'down': rev_relation = 'up'
                            if rev_direction == 'left': rev_relation = 'right'
                            if rev_direction == 'right': rev_relation = 'left'
                            if tile_goal in self.adjacency and rev_direction in self.adjacency[tile_goal] and self.adjacency[tile_goal][rev_direction] == robot_location:
                                is_adjacent = True
                                break
                        if is_adjacent:
                            break

                    if not is_adjacent:
                        heuristic_value += 1  # For move action (simplified)

        return heuristic_value
