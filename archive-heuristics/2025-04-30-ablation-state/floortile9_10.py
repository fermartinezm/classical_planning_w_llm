from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Import math for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args has wildcards at the end
    if len(parts) < len(args) or not all(fnmatch(part, arg) for part, arg in zip(parts, args)):
         return False
    # Handle cases where fact has more parts than args (e.g., matching "(at obj loc extra)" with "at obj loc")
    # For this domain, predicates have fixed arity, so len(parts) == len(args) is expected for a full match.
    # Let's refine the check to require exact number of parts unless the last arg is a wildcard pattern like '*'
    if len(parts) != len(args) and not (len(args) > 0 and args[-1] == '*'):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


def get_tile_coords(tile_name):
    """Extract (row, col) from tile name like 'tile_r_c'."""
    parts = tile_name.split('_')
    # Assumes format tile_row_col
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            pass # Not a standard tile name format
    return None # Or raise an error, depending on expected input

def manhattan_distance(coords1, coords2):
    """Calculate Manhattan distance between two (row, col) tuples."""
    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their correct colors. It sums the minimum estimated cost for any single
    robot to paint each unpainted goal tile, calculated independently for each tile.

    # Assumptions
    - Tiles are arranged in a grid, and tile names like 'tile_r_c' encode their
      (row, col) coordinates.
    - Movement actions (up, down, left, right) correspond to moving between
      adjacent tiles in the grid and cost 1.
    - Paint actions (paint_up, paint_down) cost 1 and require the robot to be
      adjacent to the tile, hold the correct color, and the tile to be clear.
    - Change_color action costs 1 and requires the robot to hold some color
      and the target color to be available.
    - If a goal tile is painted with the wrong color, the problem is considered
      unsolvable in this domain as there's no action to clear a painted tile.
    - Unpainted goal tiles are assumed to be currently 'clear' if they are not
      painted with the wrong color.
    - The heuristic sums individual costs per unpainted tile, which might
      overestimate by double-counting shared movement or color change costs,
      but aims to provide a good estimate for greedy search.

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need to be painted
      and with which color. Stores this in `self.goal_paintings`.
    - Pre-calculates coordinates for all tiles based on their names, storing
      them in `self.tile_coords`. This is done by iterating through all objects
      that are tiles (found in initial state and static facts) and parsing their names.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of tiles that need to be painted to satisfy the goal.
       These are tiles `T` for which `(painted T C)` is a goal fact, but
       `(painted T C)` is not true in the current state.
    2. While identifying tiles needing painting, check if any goal tile `T` is
       currently painted with a color `C'` that is *not* its required goal color.
       If such a tile is found, the state is considered unsolvable, and the
       heuristic returns `float('inf')`.
    3. For the remaining tiles `T` that need painting (i.e., they are not painted
       correctly and are not painted wrongly, implying they are clear), store
       them along with their required goal color `C_T`. Let this set be `tiles_to_paint`.
    4. If `tiles_to_paint` is empty, the goal is reached, and the heuristic is 0.
    5. Get the current location and held color for each robot from the state.
    6. Initialize the total heuristic value to 0.
    7. For each tile `T` in `tiles_to_paint` needing color `C_T`:
        a. Determine the coordinates of tile `T`.
        b. Calculate the minimum cost for *any* robot `R` to paint this tile `T`,
           considering the cost for robot `R` to individually complete this task
           (ignoring other tiles and robots for this specific calculation).
        c. The estimated cost for robot `R` to paint tile `T` with color `C_T` is:
           - Cost to get color `C_T`: 1 if robot `R` currently holds a different color (`R_color != C_T`), otherwise 0.
           - Cost to move adjacent to `T`: Calculate the Manhattan distance between
             robot `R`'s current location (`R_loc`) and tile `T`'s location (`T_coords`).
             The minimum number of moves to reach *any* tile adjacent to `T` is
             `max(0, dist(R_loc, T_coords) - 1)`.
           - Cost to paint `T`: 1 (for the `paint_up` or `paint_down` action).
           - Total cost for robot `R` = `(1 if R_color != C_T else 0) + max(0, dist(R_loc, T_coords) - 1) + 1`.
        d. Find the minimum of this cost over all available robots `R`. Let this be `min_cost_for_tile`.
        e. If no robot can reach/paint the tile (e.g., due to missing coordinate info), `min_cost_for_tile` remains `float('inf')`, indicating an unsolvable state. Return `float('inf')`.
        f. Add `min_cost_for_tile` to the `total_heuristic`.
    8. Return the `total_heuristic`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and tile coordinates.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state_facts = task.initial_state # Include initial state facts to find all objects

        # Store goal locations and colors for each tile that needs painting.
        # Map: tile_name -> goal_color
        self.goal_paintings = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_paintings[tile] = color

        # Map tile name to (row, col) coordinates.
        self.tile_coords = {}
        # Find all tile objects by looking at all objects mentioned in initial state and static facts
        all_objects = set()
        for fact in (initial_state_facts | static_facts):
             parts = get_parts(fact)
             # Add all arguments as potential objects
             for part in parts[1:]:
                 all_objects.add(part)

        # Filter for objects that look like tiles and parse coordinates
        for obj_name in all_objects:
             coords = get_tile_coords(obj_name)
             if coords is not None:
                 self.tile_coords[obj_name] = coords

        # If tile names don't follow tile_r_c format, we would need to build the grid
        # structure from up/down/left/right facts using BFS/DFS, assigning coordinates.
        # Assuming tile_r_c format is reliable based on examples.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Identify tiles that need painting (goal not achieved) and check for unsolvable states.
        # Map: tile_name -> required_color
        tiles_to_paint = {}
        for tile, goal_color in self.goal_paintings.items():
            is_painted_correctly = f"(painted {tile} {goal_color})" in state
            if is_painted_correctly:
                continue # Already satisfied

            # Tile needs painting. Check its current status to detect unsolvable states.
            is_painted_wrongly = False
            for fact in state:
                 # Check if the fact is a painted predicate for this tile, but with a different color
                 if match(fact, "painted", tile, "*"):
                     current_color = get_parts(fact)[2]
                     if current_color != goal_color:
                         is_painted_wrongly = True
                         break # Found wrong color

            if is_painted_wrongly:
                # Goal tile is painted with the wrong color. Domain has no unpaint action. Unsolvable.
                return float('inf')
            # If not painted correctly and not painted wrongly, it must be clear (assuming valid states).
            # So, it needs painting.
            tiles_to_paint[tile] = goal_color


        # If all goal tiles are painted correctly, the heuristic is 0.
        if not tiles_to_paint:
            return 0

        # Get robot locations and colors
        robot_info = {} # Map: robot_name -> {'location': tile_name, 'color': color_name}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot, location = parts[1], parts[2]
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['location'] = location
            elif parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['color'] = color

        # If there are tiles to paint but no robots, it's unsolvable
        if not robot_info:
             return float('inf')

        total_heuristic = 0

        # Calculate the minimum cost for any robot to paint each unpainted tile individually
        for tile, goal_color in tiles_to_paint.items():
            tile_coords = self.tile_coords.get(tile)
            if tile_coords is None:
                 # Tile name didn't match expected format or wasn't found in initial/static facts.
                 # This indicates an issue with the problem definition or parsing. Treat as unsolvable.
                 return float('inf')

            min_cost_for_tile = float('inf')

            for robot_name, info in robot_info.items():
                robot_location = info.get('location')
                robot_color = info.get('color')

                if robot_location is None or robot_color is None:
                    # Robot info is incomplete, indicates an issue with the state. Treat as unsolvable.
                    return float('inf')

                robot_coords = self.tile_coords.get(robot_location)
                if robot_coords is None:
                    # Robot is at an unknown location. Treat as unsolvable.
                    return float('inf')

                # Cost to get the correct color
                color_cost = 1 if robot_color != goal_color else 0

                # Cost to move adjacent to the tile
                # Manhattan distance between robot location and tile location
                dist = manhattan_distance(robot_coords, tile_coords)
                # Moves needed to reach a tile adjacent to the target tile
                # If dist is 0 (robot is on the tile), needs 1 move to get adjacent.
                # If dist is 1 (robot is adjacent), needs 0 moves.
                # If dist > 1, needs dist - 1 moves.
                movement_cost = max(0, dist - 1)

                # Cost to paint the tile
                paint_cost = 1

                # Total cost for this robot to paint this tile
                cost_this_robot = color_cost + movement_cost + paint_cost

                min_cost_for_tile = min(min_cost_for_tile, cost_this_robot)

            # If min_cost_for_tile is still infinity after checking all robots,
            # it means no robot could potentially paint this tile (e.g., no robots exist,
            # or robot locations/colors were incomplete). This should be caught earlier,
            # but this check is a safeguard.
            if min_cost_for_tile == float('inf'):
                 return float('inf') # Unsolvable state

            total_heuristic += min_cost_for_tile

        return total_heuristic
