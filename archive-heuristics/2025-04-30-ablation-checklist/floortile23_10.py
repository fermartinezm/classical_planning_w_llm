from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Import math for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_coords(tile_name):
    """
    Parses a tile name like 'tile_row_col' into a (row, col) tuple of integers.
    Assumes tile names follow this specific format.
    """
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            # Handle unexpected tile name format
            print(f"Warning: Unexpected tile name format: {tile_name}")
            return None
    except (ValueError, IndexError):
        # Handle errors during parsing
        print(f"Error parsing tile coordinates for: {tile_name}")
        return None

def manhattan_distance(coords1, coords2):
    """Calculates the Manhattan distance between two (row, col) coordinate tuples."""
    if coords1 is None or coords2 is None:
        return math.inf # Cannot calculate distance if coordinates are invalid
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are not yet painted correctly. It considers the movement cost for the robot
    to reach a tile adjacent to the target tile, the cost of painting, and the
    estimated cost of changing colors.

    # Assumptions:
    - Tiles are arranged in a grid and named 'tile_row_col'.
    - The grid structure (up, down, left, right) corresponds to adjacent tiles
      in the grid coordinates.
    - Only one robot exists (based on examples).
    - Tiles that are painted with the wrong color cannot be repainted (implied by
      the domain requiring `(clear ?y)` for painting, and painting removes `(clear ?y)`
      without adding a way to make it clear again). Thus, we only consider goal
      tiles that are currently `clear`.

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need to be painted
      and with which colors.
    - Identifies all valid tile coordinates from the initial state to validate
      adjacent tile calculations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal facts of the form `(painted tile color)`.
    2. Filter these goals to find the "unsatisfied" painting goals: those
       `(painted tile color)` facts that are not true in the current state.
       Crucially, based on domain rules, a tile must be `clear` to be painted.
       If a goal tile is *not* clear and *not* painted with the goal color,
       it's an unsolvable state according to the domain, but we assume solvable
       problems. So, we focus on goal tiles that are currently `clear`.
    3. If there are no unsatisfied painting goals (i.e., all goal tiles are
       painted correctly), the heuristic value is 0.
    4. Find the robot's current location and color.
    5. For each unsatisfied painting goal `(painted tile color)`:
       - Calculate the coordinates of the target `tile`.
       - Determine the coordinates of all tiles adjacent to the target tile
         (up, down, left, right).
       - Filter these adjacent coordinates to include only those that correspond
         to valid tiles in the problem instance.
       - Calculate the minimum Manhattan distance from the robot's current
         location to any of these valid adjacent tile coordinates. This is the
         estimated movement cost to get the robot into a position to paint the tile.
       - Add this minimum distance + 1 (for the paint action itself) to a running
         total for movement and painting costs.
       - Collect the required `color` for this tile.
    6. After processing all unsatisfied painting goals, determine the set of
       distinct colors required.
    7. Estimate the color change cost: This is the number of distinct colors
       needed minus 1 (since the robot starts with some color), plus an
       additional 1 if the robot's current color is *not* one of the needed colors
        (representing the first change to get a useful color). If no colors are needed (shouldn't happen if there are unpainted goals), cost is 0.
    8. The total heuristic value is the sum of the total movement/painting cost
       and the estimated color change cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # Parse goal facts to get target paintings {tile_name: color_name}
        self.goal_paintings = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'painted' and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color

        # Identify all valid tile coordinates from the initial state and static facts
        # We assume all objects listed in the initial state are valid objects.
        # Tile objects are named 'tile_row_col'.
        self.valid_coords = set()
        # A robust way is to find all facts involving tiles and extract tile names
        all_facts = set(task.initial_state) | set(task.static)
        for fact in all_facts:
             parts = get_parts(fact)
             for part in parts:
                 if part.startswith('tile_'):
                     coords = get_coords(part)
                     if coords is not None:
                         self.valid_coords.add(coords)

        # Identify the robot name (assuming only one robot based on examples)
        self.robot_name = None
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == 'robot-at' and len(parts) == 3:
                 self.robot_name = parts[1]
                 break # Found the robot

        if self.robot_name is None:
             print("Warning: Could not find robot in initial state.")


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unsatisfied painting goals that are currently clear
        # (We assume incorrectly painted tiles that are goal tiles make the problem unsolvable)
        unsatisfied_goals = {}
        for tile, goal_color in self.goal_paintings.items():
            # Check if the tile is *not* painted with the goal color
            is_painted_correctly = f'(painted {tile} {goal_color})' in state
            # Check if the tile is clear (can be painted)
            is_clear = f'(clear {tile})' in state

            # A tile needs painting if it's a goal tile and not painted correctly.
            # We only consider those that are currently clear, as others cannot be painted.
            if not is_painted_correctly and is_clear:
                 unsatisfied_goals[tile] = goal_color
            # Note: If a tile is a goal tile but is painted with the WRONG color,
            # the domain doesn't allow repainting it because it's not clear.
            # We assume such states don't occur in solvable problems or reachable states
            # unless the tile wasn't a goal tile initially. So we ignore incorrectly
            # painted goal tiles that are not clear.

        # 2. If all goal tiles are painted correctly (or are unpaintable and not clear), return 0
        if not unsatisfied_goals:
            return 0

        # 3. Find robot's current location and color
        robot_tile = None
        robot_color = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'robot-at' and len(parts) == 3 and parts[1] == self.robot_name:
                robot_tile = parts[2]
            elif parts and parts[0] == 'robot-has' and len(parts) == 3 and parts[1] == self.robot_name:
                 robot_color = parts[2]

        if robot_tile is None:
             # Robot location not found - problem state is likely invalid or goal unreachable
             return math.inf # Or a large number

        robot_coords = get_coords(robot_tile)
        if robot_coords is None:
             # Robot tile name is malformed
             return math.inf # Or a large number


        # 4. Calculate movement and paint cost for each unsatisfied goal tile
        total_move_paint_cost = 0
        needed_colors = set()

        for tile, color in unsatisfied_goals.items():
            tile_coords = get_coords(tile)
            if tile_coords is None:
                 # Goal tile name is malformed
                 return math.inf # Or a large number

            # Find valid adjacent tile coordinates
            r, c = tile_coords
            potential_adj_coords = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            valid_adj_coords = [ac for ac in potential_adj_coords if ac in self.valid_coords]

            if not valid_adj_coords:
                 # A goal tile has no valid adjacent tiles - likely an invalid problem setup
                 return math.inf # Or a large number

            # Calculate minimum distance from robot to any valid adjacent tile
            min_dist_to_adj = math.inf
            for adj_coords in valid_adj_coords:
                 dist = manhattan_distance(robot_coords, adj_coords)
                 if dist < min_dist_to_adj:
                     min_dist_to_adj = dist

            # Cost for this tile: move to adjacent + paint action
            total_move_paint_cost += min_dist_to_adj + 1
            needed_colors.add(color)

        # 5. Calculate color change cost
        num_distinct_needed_colors = len(needed_colors)
        color_cost = 0

        if num_distinct_needed_colors > 0:
            # Minimum switches needed is num_distinct_needed_colors - 1
            color_cost = max(0, num_distinct_needed_colors - 1)
            # If the robot's current color is not one of the needed colors,
            # it needs at least one initial change action to get a useful color.
            if robot_color not in needed_colors:
                 color_cost += 1

        # 6. Total heuristic is sum of movement/paint costs and color change cost
        total_h = total_move_paint_cost + color_cost

        return total_h

