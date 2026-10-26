from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(up tile1 tile2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class FloortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles
    with their goal colors. It considers:
    - The distance robots need to move to reach unpainted tiles
    - The color changes required
    - The painting actions needed

    # Assumptions
    - Robots can move freely between adjacent tiles when clear
    - Each painting action requires the robot to be adjacent to the tile
    - Color changes are only needed when the robot doesn't have the required color
    - Multiple robots can work in parallel (though the heuristic doesn't explicitly coordinate them)

    # Heuristic Initialization
    - Extract goal painting conditions
    - Build adjacency maps for efficient movement calculations
    - Identify available colors

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile that needs to be painted (goal conditions):
        a. If already painted correctly, no cost
        b. Else:
            i. Find the nearest robot that can paint it (considering current color)
            ii. Calculate movement cost (Manhattan distance)
            iii. Add color change cost if needed
            iv. Add painting action cost
    2. Sum all costs while trying to optimize robot assignments
    3. Add a small penalty for each remaining color change to encourage batching
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building adjacency maps."""
        self.goals = task.goals
        self.static = task.static

        # Extract adjacency relationships
        self.adjacency = {}  # tile -> {direction: neighbor}
        self.all_tiles = set()
        
        for fact in self.static:
            if match(fact, "up", "*", "*"):
                _, y, x = get_parts(fact)
                self.adjacency.setdefault(x, {})["up"] = y
                self.all_tiles.update([x, y])
            elif match(fact, "down", "*", "*"):
                _, x, y = get_parts(fact)
                self.adjacency.setdefault(x, {})["down"] = y
                self.all_tiles.update([x, y])
            elif match(fact, "left", "*", "*"):
                _, x, y = get_parts(fact)
                self.adjacency.setdefault(x, {})["left"] = y
                self.all_tiles.update([x, y])
            elif match(fact, "right", "*", "*"):
                _, y, x = get_parts(fact)
                self.adjacency.setdefault(x, {})["right"] = y
                self.all_tiles.update([x, y])

        # Extract goal painting conditions
        self.goal_paintings = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                _, tile, color = get_parts(goal)
                self.goal_paintings[tile] = color

        # Extract available colors
        self.available_colors = {
            get_parts(fact)[1] for fact in self.static if match(fact, "available-color", "*")
        }

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract current robot positions and colors
        robots = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                _, robot, tile = get_parts(fact)
                robots[robot] = {"pos": tile}
            elif match(fact, "robot-has", "*", "*"):
                _, robot, color = get_parts(fact)
                robots[robot]["color"] = color

        # Extract currently painted tiles
        current_paintings = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                _, tile, color = get_parts(fact)
                current_paintings[tile] = color

        total_cost = 0
        color_changes_needed = 0

        # For each tile that needs painting
        for tile, goal_color in self.goal_paintings.items():
            # Skip if already painted correctly
            if tile in current_paintings and current_paintings[tile] == goal_color:
                continue

            # Find the best robot for this tile
            min_cost = float('inf')
            best_robot = None

            for robot, data in robots.items():
                robot_pos = data["pos"]
                robot_color = data["color"]

                # Calculate movement cost (Manhattan distance)
                x1, y1 = map(int, robot_pos.split('_')[1:])
                x2, y2 = map(int, tile.split('_')[1:])
                movement_cost = abs(x1 - x2) + abs(y1 - y2)

                # Calculate color change cost if needed
                color_cost = 0 if robot_color == goal_color else 1

                total_robot_cost = movement_cost + color_cost + 1  # +1 for paint action

                if total_robot_cost < min_cost:
                    min_cost = total_robot_cost
                    best_robot = robot

            if best_robot is not None:
                total_cost += min_cost
                if robots[best_robot]["color"] != goal_color:
                    color_changes_needed += 1

        # Add a small penalty for each color change to encourage batching
        total_cost += color_changes_needed * 0.1

        return total_cost
