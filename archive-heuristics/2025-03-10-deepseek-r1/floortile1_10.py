from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class floortile1Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles with their required colors. It considers the minimal steps for robots to move adjacent to each unpainted tile, the necessary color changes, and the paint actions.

    # Assumptions
    - Each tile can be painted by the closest robot.
    - Robots can change color once per tile if needed, even though in reality a single color change can affect multiple tiles.
    - Movement cost is based on Manhattan distance to an adjacent tile of the target.

    # Heuristic Initialization
    - Extract the goal conditions to determine which tiles need to be painted which colors.
    - Parse static adjacency facts to build a coordinate system for each tile.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Extract Goal Information**: Identify which tiles need to be painted and their required colors.
    2. **Current State Check**: For each goal tile, check if it's already correctly painted.
    3. **Robot Positions and Colors**: Track each robot's current position and paint color.
    4. **Cost Calculation per Tile**:
        a. For each unpainted tile, calculate the minimal cost considering all robots.
        b. Compute Manhattan distance from each robot to the tile, adjust to steps needed to reach an adjacent tile.
        c. Add color change cost if the robot's current color doesn't match the required color.
        d. Add 1 for the paint action.
    5. **Sum Costs**: Total heuristic is the sum of minimal costs for all unpainted tiles.
    """

    def __init__(self, task):
        """Initialize the heuristic with goal and tile information."""
        self.goal_painted = {}
        self.tile_coords = {}
        self.static = task.static

        # Extract goal painted conditions
        for goal in task.goals:
            parts = self._get_parts(goal)
            if parts[0] == 'painted':
                tile = parts[1]
                color = parts[2]
                self.goal_painted[tile] = color

        # Extract tile coordinates from static adjacency facts
        for fact in self.static:
            parts = self._get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right']:
                tile1, tile2 = parts[1], parts[2]
                for tile in [tile1, tile2]:
                    if tile not in self.tile_coords:
                        coords = tile.split('_')[1:]
                        x, y = int(coords[0]), int(coords[1])
                        self.tile_coords[tile] = (x, y)

    def _get_parts(self, fact):
        """Split a PDDL fact into its components."""
        return fact[1:-1].split()

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_painted = {}
        robots = {}

        # Extract current painted tiles
        for fact in state:
            parts = self._get_parts(fact)
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                current_painted[tile] = color

        # Extract robot positions and colors
        for fact in state:
            parts = self._get_parts(fact)
            if parts[0] == 'robot-at':
                robot_name, tile = parts[1], parts[2]
                if tile in self.tile_coords:
                    robots[robot_name] = {
                        'pos': self.tile_coords[tile],
                        'color': None
                    }
        for fact in state:
            parts = self._get_parts(fact)
            if parts[0] == 'robot-has':
                robot_name, color = parts[1], parts[2]
                if robot_name in robots:
                    robots[robot_name]['color'] = color

        total_cost = 0

        for tile, req_color in self.goal_painted.items():
            if current_painted.get(tile) != req_color:
                tx, ty = self.tile_coords[tile]
                min_cost = float('inf')

                for robot in robots.values():
                    rx, ry = robot['pos']
                    distance = abs(rx - tx) + abs(ry - ty)
                    steps = distance - 1 if distance > 0 else 1
                    color_cost = 0 if robot['color'] == req_color else 1
                    cost = steps + color_cost + 1  # +1 for paint

                    if cost < min_cost:
                        min_cost = cost

                if min_cost != float('inf'):
                    total_cost += min_cost

        return total_cost
