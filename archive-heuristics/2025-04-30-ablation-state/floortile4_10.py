import math
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# --- Utility functions ---

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at package1 city1-1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_coords(tile_name):
    """Parses tile name 'tile_r_c' into (row, col) integer coordinates."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            return None # Not a tile_r_c format or invalid numbers
    return None # Not a tile_r_c format

def manhattan_distance(coords1, coords2):
    """Calculates Manhattan distance between two coordinate pairs (r1, c1) and (r2, c2)."""
    if coords1 is None or coords2 is None:
        return math.inf
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

# --- Heuristic Class ---

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their required colors. It sums the minimum estimated cost for each
    unsatisfied goal tile, considering the color and location of each robot.

    # Assumptions
    - Tiles are arranged in a grid, and tile names follow the pattern 'tile_row_col'.
    - Robots always have a color (predicate `robot-has`). They don't become `free-color`.
    - If a goal tile is painted with the wrong color in the current state, the problem
      is considered unsolvable from this state (as there's no unpaint/repaint action),
      and the heuristic returns infinity.
    - There are no obstacles blocking movement between adjacent clear tiles.
    - The cost of moving between adjacent tiles is 1.
    - The cost of changing color is 1.
    - The cost of painting a tile is 1.

    # Heuristic Initialization
    - Extract the goal conditions, specifically the required color for each goal tile.
    - Parse static facts (`up`, `down`, `left`, `right`) to identify all tiles and their coordinates.
    - Build a mapping from tile name to (row, col) integer coordinates. This mapping is crucial
      for calculating movement distances.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state. If yes, the heuristic value is 0.
    2. Identify all goal tiles that are not currently painted with their required color.
       This is done by iterating through the `self.goal_requirements` dictionary
       and checking if the corresponding `(painted tile color)` fact exists in the state.
    3. For each such unsatisfied goal tile `tile_X` requiring `color_Y`:
        a. Check if `tile_X` is currently painted with *any* color. If it is painted
           with a color different from `color_Y`, the state is considered unsolvable
           from this point, and the heuristic returns infinity.
        b. If `tile_X` is not painted with the wrong color (meaning it's either clear
           or painted correctly, which is handled by step 2), find the minimum cost
           for *any* robot `R` to paint `tile_X` with `color_Y`.
        c. Iterate through each robot `R` present in the current state, determining its location (`tile_R_loc`) and current color (`color_R`).
        d. For each robot `R`, calculate the estimated cost to paint `tile_X`:
            i. Calculate the color change cost: 1 if `color_R` is not `color_Y`, 0 otherwise.
            ii. Calculate the movement cost for `R` to get into a position adjacent to `tile_X` such that `tile_X` is clear and paintable.
                - Get coordinates: `robot_coords = get_coords(tile_R_loc)` and `tile_coords = get_coords(tile_X)`.
                - Calculate Manhattan distance: `dist = manhattan_distance(robot_coords, tile_coords)`.
                - If `dist == 0` (robot is currently *at* `tile_X`): The robot must first move off `tile_X` (1 action) to make it clear. After this move, the robot is at an adjacent tile and `tile_X` is clear, allowing the paint action. The move cost is 1.
                - If `dist > 0`: The minimum number of moves to reach *any* tile adjacent to `tile_X` is `dist - 1`. The move cost is `dist - 1`.
            iii. Calculate the total estimated cost for robot `R` to paint `tile_X`: (color change cost) + (movement cost) + 1 (cost of the paint action itself).
        e. The minimum cost for `tile_X` is the minimum of the total estimated costs calculated for all robots. If no robots are available or their locations are invalid, this minimum might remain infinity, indicating the tile cannot be painted by any known robot.
    4. The total heuristic value is the sum of the minimum estimated costs calculated for all unsatisfied goal tiles.
    5. If at any point a goal tile is found to be wrongly painted or its location is unknown, the heuristic returns infinity. Otherwise, it returns the calculated total finite cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and grid structure."""
        # Store goal requirements: {tile_name: required_color}
        self.goal_requirements = {}
        for goal in task.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_requirements[tile] = color

        # Map tile names to coordinates
        self.tile_coords = {}

        # Collect all tile names from static facts involving up/down/left/right
        all_tile_names = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] in ['up', 'down', 'left', 'right']:
                if len(parts) > 1: all_tile_names.add(parts[1])
                if len(parts) > 2: all_tile_names.add(parts[2])

        # Parse coordinates for all collected tile names
        for tile_name in all_tile_names:
            coords = get_coords(tile_name)
            if coords is not None:
                self.tile_coords[tile_name] = coords
            # else: Tile name doesn't follow expected pattern, ignore or handle error

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Identify current robot locations and colors
        robot_info = {} # robot_name -> {'location': tile_name, 'color': color_name}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'robot-at' and len(parts) == 3:
                robot, location = parts[1], parts[2]
                robot_info.setdefault(robot, {})['location'] = location
            elif parts and parts[0] == 'robot-has' and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_info.setdefault(robot, {})['color'] = color

        total_heuristic = 0

        # Iterate through goal tiles that need painting
        for goal_tile, required_color in self.goal_requirements.items():
            # Check if the goal for this tile is already satisfied
            if f"(painted {goal_tile} {required_color})" in state:
                continue

            # Check if the tile is painted with the wrong color
            is_wrongly_painted = False
            for fact in state:
                 if match(fact, "painted", goal_tile, "*"):
                     current_painted_color = get_parts(fact)[2] # Extract the color
                     if current_painted_color != required_color:
                         is_wrongly_painted = True
                     break

            if is_wrongly_painted:
                 # This tile is painted, but with the wrong color.
                 # In this domain, there's no way to unpaint or repaint.
                 # This state is likely a dead end or unreachable in a solvable problem.
                 # Return a large value to prune this branch.
                 return math.inf # Problem likely unsolvable from here

            # If we reach here, the tile is either clear or painted correctly (handled by continue).
            # So, this goal tile is clear and needs painting.

            tile_coords = self.tile_coords.get(goal_tile)
            if tile_coords is None:
                 # Goal tile not found in the parsed grid structure.
                 # This indicates an issue with the instance or parsing.
                 # Return a large value.
                 return math.inf # Cannot estimate cost if tile location is unknown

            min_cost_for_tile = math.inf

            # Calculate cost for each robot to paint this tile
            for robot_name, info in robot_info.items():
                robot_location = info.get('location')
                robot_color = info.get('color')

                if robot_location is None or robot_color is None:
                    # Robot info incomplete - problem with state representation?
                    continue # Skip this robot

                robot_coords = self.tile_coords.get(robot_location)
                if robot_coords is None:
                    # Robot location not found in grid structure.
                    continue # Skip this robot

                # 1. Color change cost
                color_cost = 1 if robot_color != required_color else 0

                # 2. Movement cost to be adjacent to goal_tile
                dist = manhattan_distance(robot_coords, tile_coords)

                # Cost to move from robot_coords to *any* tile adjacent to tile_coords
                if dist == 0:
                    # Robot is at the goal tile. Needs 1 move to get off and be adjacent.
                    move_cost = 1
                else: # dist > 0
                    # Robot is not at the goal tile. Needs dist - 1 moves to reach an adjacent tile.
                    move_cost = dist - 1

                # 3. Paint action cost
                paint_cost = 1

                # Total cost for this robot to paint this tile
                cost_this_robot = color_cost + move_cost + paint_cost

                min_cost_for_tile = min(min_cost_for_tile, cost_this_robot)

            # Add the minimum cost for this tile to the total heuristic
            if min_cost_for_tile == math.inf:
                 # No robot can paint this tile (e.g., no robots, or robot locations invalid)
                 # This implies unsolvability or a very high cost path.
                 # Return a large value.
                 return math.inf # Cannot paint this goal tile

            total_heuristic += min_cost_for_tile

        return total_heuristic
