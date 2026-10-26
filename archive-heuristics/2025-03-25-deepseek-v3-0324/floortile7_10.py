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


class FloortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all tiles according to the goal pattern.
    It considers:
    - The number of unpainted tiles that need to be painted.
    - The distance robots must travel to reach unpainted tiles.
    - The color changes required for robots to match the goal colors.

    # Assumptions
    - Robots can move freely between adjacent tiles (up, down, left, right).
    - Each robot can carry only one color at a time.
    - Changing color requires an action.
    - Painting a tile requires the robot to be adjacent to it and have the correct color.

    # Heuristic Initialization
    - Extract the goal painting conditions (which tiles need which colors).
    - Build a graph representation of tile adjacencies from static facts.
    - Identify available colors from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (according to goals):
        - If already painted correctly, no cost.
        - Otherwise:
            a) Find the nearest robot that can paint it (considering current color and distance).
            b) Add cost for:
                - Moving the robot to the tile (Manhattan distance).
                - Changing color if needed (1 action per change).
                - Painting the tile (1 action).
    2. Sum the costs for all tiles to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract goal painting conditions: {(tile, color)}
        self.goal_paintings = set()
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                _, tile, color = get_parts(goal)
                self.goal_paintings.add((tile, color))

        # Build adjacency graph: {tile: {direction: neighbor}}
        self.adjacency = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ("up", "down", "left", "right"):
                direction, tile1, tile2 = parts
                if tile1 not in self.adjacency:
                    self.adjacency[tile1] = {}
                self.adjacency[tile1][direction] = tile2

        # Extract available colors
        self.available_colors = {
            parts[1] for fact in static_facts 
            if match(fact, "available-color", "*")
        }

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Get current robot positions and colors: {robot: (tile, color)}
        robots = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                _, robot, tile = get_parts(fact)
                robots[robot] = [tile, None]
            elif match(fact, "robot-has", "*", "*"):
                _, robot, color = get_parts(fact)
                if robot in robots:
                    robots[robot][1] = color

        # Get current paintings: {tile: color}
        current_paintings = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                _, tile, color = get_parts(fact)
                current_paintings[tile] = color

        total_cost = 0

        for tile, goal_color in self.goal_paintings:
            # Skip if already painted correctly
            if tile in current_paintings and current_paintings[tile] == goal_color:
                continue

            min_cost = float('inf')

            # Find the best robot to paint this tile
            for robot, (robot_tile, robot_color) in robots.items():
                # Calculate movement cost (Manhattan distance)
                movement_cost = self._manhattan_distance(robot_tile, tile)
                
                # Calculate color change cost if needed
                color_cost = 0 if robot_color == goal_color else 1
                
                # Total cost for this robot
                total_robot_cost = movement_cost + color_cost + 1  # +1 for painting
                
                if total_robot_cost < min_cost:
                    min_cost = total_robot_cost

            total_cost += min_cost

        return total_cost

    def _manhattan_distance(self, tile1, tile2):
        """Calculate Manhattan distance between two tiles based on their coordinates."""
        # Extract coordinates from tile names (format tile_X_Y)
        try:
            x1, y1 = map(int, tile1.split('_')[1:])
            x2, y2 = map(int, tile2.split('_')[1:])
            return abs(x1 - x2) + abs(y1 - y2)
        except:
            # Fallback for non-standard tile names
            return 1
