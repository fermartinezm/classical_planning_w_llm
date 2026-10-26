from fnmatch import fnmatch

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

class floortileHeuristic:
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles by considering the minimal steps each robot needs to reach each tile, including movement and color change.

    # Assumptions:
    - The robot can move up, down, left, right.
    - Each tile that needs painting is reachable by at least one robot.
    - The minimal steps for a tile is the minimal steps required by any robot to reach it and paint it.

    # Heuristic Initialization
    - Extract the goal tiles and their required colors from the task's goals.
    - Parse the static facts to identify available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each tile, determine if it's already painted correctly. If not, add to the list of required tiles.
    2. For each required tile, calculate the minimal steps needed by any robot to paint it:
       a. For each robot, compute the Manhattan distance from its current position to the tile.
       b. If the robot's current color matches the tile's required color, steps = distance + 1 (paint action).
       c. If not, steps = distance + 2 (change color + paint action).
       d. Keep the minimal steps for the tile.
    3. Sum the minimal steps for all required tiles.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract available colors from static facts
        self.available_colors = set()
        for fact in self.static:
            if match(fact, "available-color", "*"):
                color = get_parts(fact)[1]
                self.available_colors.add(color)

        # Extract goal tiles and their required colors
        self.goal_tiles = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                tile = get_parts(goal)[1]
                color = get_parts(goal)[2]
                self.goal_tiles[tile] = color

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract current positions and colors of robots
        current_robot_positions = {}
        current_robot_colors = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot, tile = get_parts(fact)
                current_robot_positions[robot] = tile
            if match(fact, "robot-has", "*", "*"):
                robot, color = get_parts(fact)
                current_robot_colors[robot] = color

        # Identify required tiles that are not yet painted correctly
        required_tiles = []
        for tile, color in self.goal_tiles.items():
            # Check if the tile is already painted correctly
            is_painted_correctly = any(
                match(f, f"(painted {tile} {color})") for f in state
            )
            if not is_painted_correctly:
                required_tiles.append((tile, color))

        total_cost = 0

        for tile, color in required_tiles:
            # Parse the tile name into coordinates
            tile_parts = tile.split('_')
            tile_x = int(tile_parts[1])
            tile_y = int(tile_parts[2])

            min_steps = float('inf')

            for robot in current_robot_positions:
                # Get the robot's current position
                robot_tile = current_robot_positions[robot]
                robot_parts = robot_tile.split('_')
                robot_x = int(robot_parts[1])
                robot_y = int(robot_parts[2])

                # Calculate Manhattan distance
                distance = abs(robot_x - tile_x) + abs(robot_y - tile_y)

                # Determine if color change is needed
                if current_robot_colors[robot] == color:
                    steps = distance + 1  # paint action
                else:
                    steps = distance + 2  # change color + paint action

                if steps < min_steps:
                    min_steps = steps

            total_cost += min_steps

        return total_cost
