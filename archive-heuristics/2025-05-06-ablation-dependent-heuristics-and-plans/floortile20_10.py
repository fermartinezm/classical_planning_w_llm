# Assume Heuristic base class is defined elsewhere and imported correctly
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided, for standalone testing
# In the actual planner environment, the real Heuristic class would be used.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Dummy class for local testing if the base class is not available
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic base class not implemented")


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def parse_tile(tile_name):
    """Parses a tile name like 'tile_R_C' into (row, col) integers."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return row, col
        except ValueError:
            # Handle cases where R or C are not integers
            raise ValueError(f"Could not parse tile name {tile_name} as tile_R_C (R or C not integer)")
    else:
        # Handle cases where the format is unexpected
        raise ValueError(f"Unexpected tile name format: {tile_name}")

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles based on their names."""
    r1, c1 = parse_tile(tile1_name)
    r2, c2 = parse_tile(tile2_name)
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the total number of actions required to paint all
    goal tiles with their correct colors. It sums the estimated minimum cost
    for each unpainted goal tile independently, considering the closest robot's
    movement and color change needs.

    # Assumptions
    - Tiles are arranged in a grid, and their names follow the 'tile_R_C' format,
      allowing Manhattan distance calculation.
    - Robots can move between adjacent clear tiles (up, down, left, right).
    - Robots can change color if the new color is available.
    - Once a tile is painted with a color, it remains painted with that color.
    - If a tile is painted with a color different from its goal color, the state is unsolvable.
    - All goal colors are available.

    # Heuristic Initialization
    - Extracts the goal tiles and their required colors from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal predicates of the form `(painted T C)`.
    2. Extract the current location and color of each robot from the state.
    3. Initialize the total heuristic value to 0.
    4. For each goal tile T and its required color C:
        a. Check if the fact `(painted T C)` is present in the current state. If yes, this goal is satisfied for this tile; continue to the next goal tile.
        b. If `(painted T C)` is not in the state, check if tile T is painted with *any* other color C'. This is done by iterating through state facts looking for `(painted T TILE_NAME C')` where C' is not C. If such a fact exists, the tile is painted incorrectly, and the state is considered unsolvable in this domain; return infinity.
        c. If the tile is not painted correctly (and not painted incorrectly), it needs to be painted. Calculate the minimum cost to paint this tile by considering each robot:
            i. For each robot R with location R_loc and color R_color:
                - Calculate the color change cost: 1 if R_color is not equal to C, otherwise 0. (Assumes required color C is available).
                - Calculate the movement cost: Find the Manhattan distance between R_loc and T. The robot needs to reach a tile adjacent to T. The minimum steps to get adjacent is `max(0, distance - 1)`.
                - The paint action cost is 1.
                - The total cost for robot R to paint tile T is `color_change_cost + movement_cost + paint_action_cost`.
            ii. Find the minimum of these costs over all robots.
        d. Add this minimum cost for tile T to the total heuristic value.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals

        # Store goal tiles and their required colors
        self.goal_tiles = {}
        for goal in self.goals:
            # Goal facts are like '(painted tile_1_1 white)'
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_tiles[tile] = color

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract robot locations and colors from the current state
        robot_locations = {}
        robot_colors = {}
        robots = set() # Keep track of robot names
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
                robots.add(robot)
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
                robots.add(robot)

        total_heuristic = 0

        # Iterate through each goal tile
        for goal_tile, required_color in self.goal_tiles.items():
            # Check if the goal for this tile is already satisfied
            if f"(painted {goal_tile} {required_color})" in state:
                continue # This tile is already painted correctly

            # Check if the tile is painted with the wrong color
            is_painted_wrong = False
            for fact in state:
                 parts = get_parts(fact)
                 # Check if the fact is a painted fact for the current goal_tile
                 if parts[0] == "painted" and len(parts) == 3 and parts[1] == goal_tile and parts[2] != required_color:
                     is_painted_wrong = True
                     break

            if is_painted_wrong:
                 # Tile is painted with the wrong color, likely unsolvable
                 return float('inf')

            # The tile is not painted correctly (and not painted wrong), so it needs painting.
            # Estimate the minimum cost to paint this single tile.
            min_cost_for_tile = float('inf')

            # Consider each robot
            if not robots: # If there are unpainted tiles but no robots, it's unsolvable
                 return float('inf')

            for robot in robots:
                r_loc = robot_locations.get(robot)
                r_color = robot_colors.get(robot)

                if r_loc is None or r_color is None:
                    # Robot state is incomplete, this robot cannot paint
                    continue

                # Cost to get the correct color
                color_cost = 0
                if r_color != required_color:
                    color_cost = 1 # Cost to change color

                # Cost to move adjacent to the tile
                try:
                    dist = manhattan_distance(r_loc, goal_tile)
                    move_cost = max(0, dist - 1) # Need to be adjacent, not on the tile
                except ValueError:
                    # If tile name parsing fails, this robot cannot reach the tile
                    move_cost = float('inf')

                # Cost to paint the tile
                paint_cost = 1

                # Total cost for this robot to paint this tile
                cost = color_cost + move_cost + paint_cost

                # Update minimum cost for this tile
                min_cost_for_tile = min(min_cost_for_tile, cost)

            # If after checking all robots, min_cost_for_tile is still infinity,
            # it means no robot can reach or paint this tile (e.g., parsing failed for all,
            # or no robots were found in state). This state is likely unsolvable.
            if min_cost_for_tile == float('inf'):
                 return float('inf')

            # Add the minimum estimated cost for this tile to the total heuristic
            total_heuristic += min_cost_for_tile

        return total_heuristic
