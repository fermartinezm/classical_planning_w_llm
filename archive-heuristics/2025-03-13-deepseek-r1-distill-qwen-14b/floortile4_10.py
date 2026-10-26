from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import sys

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed for the robot to paint all required tiles. It considers the robot's current position, the colors it holds, and the distances to the target tiles.

    # Assumptions:
    - The robot can move up, down, left, right on a grid.
    - Each tile can be painted only once.
    - The robot can change colors, which takes two actions (drop and pick up).

    # Heuristic Initialization
    - Extracts goal conditions (which tiles need to be painted and their colors).
    - Builds a grid map from static facts to determine tile connectivity.
    - Determines the robot's initial position and color.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify all tiles that need to be painted and their required colors.
    2. For each unpainted goal tile, calculate the Manhattan distance from the robot's current position.
    3. Sum these distances to estimate the movement cost.
    4. For each color change needed, add two actions (drop and pick up the new color).
    5. If multiple tiles require the same color, group them to minimize color changes.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations and required colors
        self.goal_info = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                self.goal_info[tile] = color

        # Build grid map from static facts
        self.grid = {}
        for fact in self.static:
            if match(fact, "up", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.grid[(x)] = {'up': y}
            if match(fact, "down", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.grid[x]['down'] = y
            if match(fact, "left", "*", "*"):
                x, y = get_parts(fact)[0], get_parts(fact)[1]
                self.grid[y]['right'] = x
            if match(fact, "right", "*", "*"):
                x, y = get_parts(fact)[0], get_parts(fact)[1]
                self.grid[y]['left'] = x

        # Determine robot's initial position and color
        self.robot_pos = None
        self.robot_color = None
        for fact in task.init:
            if match(fact, "robot-at", "*", "*"):
                self.robot_pos = get_parts(fact)[1]
            if match(fact, "robot-has", "*", "*"):
                self.robot_color = get_parts(fact)[1]

    def __call__(self, node):
        """Estimate the minimum number of actions to reach the goal state."""
        state = node.state

        # Extract current painted tiles
        painted = set()
        for fact in state:
            if match(fact, "painted", "*", "*"):
                painted.add((get_parts(fact)[1], get_parts(fact)[2]))

        # Extract current color of the robot
        current_color = self.robot_color
        for fact in state:
            if match(fact, "robot-has", "*", "*"):
                current_color = get_parts(fact)[1]

        # Extract current position of the robot
        current_pos = self.robot_pos
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                current_pos = get_parts(fact)[1]

        # If all goals are already achieved, return 0
        if all((tile, color) in painted for tile, color in self.goal_info.items()):
            return 0

        total_cost = 0

        # Group tiles by their required color
        color_groups = {}
        for tile, color in self.goal_info.items():
            if (tile, color) not in painted:
                if color not in color_groups:
                    color_groups[color] = []
                color_groups[color].append(tile)

        # For each color group, calculate the cost
        for color, tiles in color_groups.items():
            if not tiles:
                continue

            # If the robot already has the required color, no need to change
            if current_color == color:
                change_cost = 0
            else:
                change_cost = 2  # drop current color and pick up new color

            # Calculate the distance for each tile in the group
            for tile in tiles:
                # Find the shortest path from current_pos to tile
                path = self.breadth_first_search(current_pos, tile)
                if path is None:
                    # If no path exists, which shouldn't happen in solvable problems
                    return float('inf')
                distance = len(path) - 1  # number of moves
                total_cost += distance

            # Add the change cost only once per color group
            if change_cost > 0:
                total_cost += change_cost

            # After handling a color group, update current_color and current_pos
            current_color = color
            current_pos = tiles[-1]  # assume moving to last tile in group

        return total_cost

    def breadth_first_search(self, start, goal):
        """
        Perform BFS to find the shortest path from start to goal in the grid.
        Returns a list of positions representing the path.
        """
        visited = set()
        queue = [(start, [start])]

        while queue:
            current, path = queue.pop(0)
            if current == goal:
                return path
            if current in visited:
                continue
            visited.add(current)

            # Explore all possible moves
            for move in ['up', 'down', 'left', 'right']:
                if move in self.grid.get(current, {}):
                    neighbor = self.grid[current][move]
                    if neighbor not in visited:
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path))

        # If no path found (shouldn't happen in solvable problems)
        return None
