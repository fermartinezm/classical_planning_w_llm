from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_coordinates(tile):
    parts = tile.split('_')
    return (int(parts[1]), int(parts[2]))

class floortile13Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles with their respective colors. It considers the movement of robots to adjacent tiles, color changes, and painting actions. For each unpainted goal tile, the heuristic calculates the minimal cost for any robot to reach a position from which the tile can be painted, considering possible color changes and moving other robots if the tile is occupied.

    # Assumptions
    - Robots can only paint tiles that are directly up or down from their current position.
    - Moving a robot off an occupied tile requires one action.
    - The minimal path for movement is computed using Manhattan distance.
    - Color changes are considered when the robot does not have the required color.

    # Heuristic Initialization
    - Extracts the goal conditions for each tile and its required color.
    - Builds adjacency lists for up and down neighbors from static facts.
    - Identifies all robots from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each goal tile, check if it's already painted correctly. If not, proceed.
    2. If the tile is occupied by a robot, add a cost to move that robot away.
    3. For each robot, compute the minimal cost to reach any adjacent tile (up or down) from which the goal tile can be painted:
       a. Calculate Manhattan distance from the robot's current position to the adjacent tile.
       b. Add a color change cost if the robot's current color doesn't match the required color.
       c. Include the paint action cost.
    4. Sum the minimal costs for all unpainted goal tiles.
    """

    def __init__(self, task):
        self.goal_painted = {}
        for goal in task.goals:
            if match(goal, "painted", "*", "*"):
                parts = get_parts(goal)
                tile = parts[1]
                color = parts[2]
                self.goal_painted[tile] = color

        self.up_neighbors = {}
        self.down_neighbors = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'up':
                y = parts[1]
                x = parts[2]
                if y not in self.up_neighbors:
                    self.up_neighbors[y] = []
                self.up_neighbors[y].append(x)
            elif parts[0] == 'down':
                y = parts[1]
                x = parts[2]
                if y not in self.down_neighbors:
                    self.down_neighbors[y] = []
                self.down_neighbors[y].append(x)

        self.robots = set()
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                self.robots.add(parts[1])
        self.robots = list(self.robots)

    def __call__(self, node):
        state = node.state
        total_cost = 0

        for tile, required_color in self.goal_painted.items():
            current_color = None
            for fact in state:
                if match(fact, "painted", tile, "*"):
                    current_color = get_parts(fact)[2]
                    break
            if current_color == required_color:
                continue

            is_clear = any(fact == f'(clear {tile})' for fact in state)
            if not is_clear:
                total_cost += 1

            min_robot_cost = float('inf')
            for robot in self.robots:
                robot_pos = None
                robot_color = None
                for fact in state:
                    parts = get_parts(fact)
                    if parts[0] == 'robot-at' and parts[1] == robot:
                        robot_pos = parts[2]
                    elif parts[0] == 'robot-has' and parts[1] == robot:
                        robot_color = parts[2]
                if not robot_pos or not robot_color:
                    continue

                possible_xs = []
                if tile in self.up_neighbors:
                    possible_xs.extend(self.up_neighbors[tile])
                if tile in self.down_neighbors:
                    possible_xs.extend(self.down_neighbors[tile])
                if not possible_xs:
                    continue

                ri, rj = parse_coordinates(robot_pos)
                min_distance = float('inf')
                for x in possible_xs:
                    xi, xj = parse_coordinates(x)
                    distance = abs(ri - xi) + abs(rj - xj)
                    if distance < min_distance:
                        min_distance = distance

                color_cost = 0 if robot_color == required_color else 1
                total_steps = min_distance + color_cost + 1
                if total_steps < min_robot_cost:
                    min_robot_cost = total_steps

            if min_robot_cost != float('inf'):
                total_cost += min_robot_cost

        return total_cost
