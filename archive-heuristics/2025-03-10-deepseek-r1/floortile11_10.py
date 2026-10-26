from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a pattern with wildcards."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortile11Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles by considering:
    - Movement cost to reach adjacent tiles.
    - Color change cost if the robot's current color is incorrect.
    - Painting action cost.

    # Assumptions
    - Robots can move freely between adjacent tiles (ignoring obstacles).
    - Each paint action requires the robot to be on an adjacent tile with the correct color.
    - Changing color is a single action regardless of current and target color.

    # Heuristic Initialization
    - Extract goal painted conditions and static adjacency information.
    - Precompute adjacency maps for each tile from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each goal tile, check if it's already painted correctly.
    2. For each unpainted tile, calculate the minimal cost for each robot:
        a. Movement cost to reach an adjacent tile.
        b. Color change cost if the robot's color is incorrect.
        c. Add one for the paint action.
    3. Sum the minimal costs across all robots and tiles.
    """

    def __init__(self, task):
        """Initialize heuristic with goal and static information."""
        self.goal_painted = {}
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted' and len(parts) == 3:
                self.goal_painted[parts[1]] = parts[2]

        # Build adjacency map: tile -> list of adjacent tiles (from up, down, left, right)
        self.adjacent_x_map = defaultdict(list)
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right'] and len(parts) == 3:
                y, x = parts[1], parts[2]
                self.adjacent_x_map[y].append(x)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_painted = {}
        robots = defaultdict(lambda: {'pos': None, 'color': None})

        # Extract current painted tiles and robot states
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'painted' and len(parts) == 3:
                current_painted[parts[1]] = parts[2]
            elif parts[0] == 'robot-at' and len(parts) == 3:
                robots[parts[1]]['pos'] = parts[2]
            elif parts[0] == 'robot-has' and len(parts) == 3:
                robots[parts[1]]['color'] = parts[2]

        total_cost = 0

        for tile, req_color in self.goal_painted.items():
            if current_painted.get(tile, None) == req_color:
                continue  # Already painted correctly

            min_tile_cost = float('inf')
            for robot in robots.values():
                r_pos = robot['pos']
                r_color = robot['color']
                if not r_pos:
                    continue  # Robot position unknown

                # Get all adjacent tiles to the target tile
                adjacent_x = self.adjacent_x_map.get(tile, [])
                if not adjacent_x:
                    continue  # No adjacent tiles (unlikely in solvable instances)

                for x in adjacent_x:
                    # Calculate Manhattan distance between robot's position and x
                    try:
                        r_coords = list(map(int, r_pos.split('_')[1:3]))
                        x_coords = list(map(int, x.split('_')[1:3]))
                        distance = abs(r_coords[0] - x_coords[0]) + abs(r_coords[1] - x_coords[1])
                    except (IndexError, ValueError):
                        distance = 0  # Fallback if tile name parsing fails

                    color_cost = 0 if r_color == req_color else 1
                    cost = distance + color_cost + 1  # movement + color change + paint
                    if cost < min_tile_cost:
                        min_tile_cost = cost

            if min_tile_cost != float('inf'):
                total_cost += min_tile_cost

        return total_cost
