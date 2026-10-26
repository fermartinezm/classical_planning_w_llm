import re
from fnmatch import fnmatch
# Assuming heuristic_base.py is in a directory named heuristics relative to where this file is run
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Use zip up to the length of the shorter sequence, fnmatch handles wildcards
    # This assumes the pattern length matches the fact parts length for relevant predicates
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the total number of actions required to paint all goal tiles with their desired colors. It sums the estimated cost for each unpainted goal tile independently. The cost for a single unpainted goal tile is estimated as 1 (for the paint action) plus the minimum cost to get any robot into the correct position with the correct color to paint that tile.

    # Assumptions
    - The grid structure is defined by 'up', 'down', 'left', 'right' static facts, and tile names follow the 'tile_row_col' format allowing coordinate mapping.
    - Movement cost between adjacent tiles is 1.
    - Changing color costs 1.
    - Painting a tile costs 1.
    - The heuristic uses Manhattan distance on the grid as a lower bound for movement cost, ignoring potential path blockages from other objects or robots and ignoring the 'clear' precondition for intermediate movement tiles.
    - If a goal tile is already painted with the wrong color, the state is considered a dead end, and a very large heuristic value is returned.
    - If a goal tile cannot be painted from any adjacent tile based on static facts, the state is considered a dead end, and a very large heuristic value is returned.

    # Heuristic Initialization
    - Parses tile objects to map tile names (e.g., 'tile_1_2') to grid coordinates (e.g., (1, 2)).
    - Parses goal conditions to store the required color for each goal tile.
    - Parses static adjacency facts ('up', 'down', 'left', 'right') to determine, for each tile, the set of adjacent tiles where a robot must be located to paint it using the corresponding paint action.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Extract the current location and color of each robot from the state.
    2. Extract the current painted status and color of each tile from the state.
    3. Check if any goal tile is currently painted with a color different from its required goal color. If so, return a very large number (indicating a likely dead end, as tiles cannot be unpainted or repainted).
    4. Initialize the total heuristic cost to 0.
    5. Iterate through each goal tile and its required color as identified during initialization.
    6. For the current goal tile `T` and required color `C`:
       a. Check if `T` is already painted with `C` in the current state. If yes, this goal is satisfied for this tile; continue to the next goal tile.
       b. If `T` is not painted with `C` (it must be clear if it's not painted, assuming valid states):
          i. Add 1 to the cost for the paint action itself.
          ii. Determine the set of tiles `X` where a robot must be positioned to paint `T` (based on the static adjacency facts and paint action definitions). This set was pre-calculated during initialization and stored in `self.paint_positions`.
          iii. If this set of required positions is empty (meaning the tile cannot be painted from anywhere according to the domain rules), return a very large number (indicating an unsolvable state).
          iv. Calculate the minimum "preparation cost" across all robots and all required painting positions for tile `T`. The preparation cost for a robot `R` to paint tile `T` from a required position `X` is:
              - The Manhattan distance between `R`'s current location and `X`.
              - Plus 1 if `R` does not currently have color `C` (for the `change_color` action).
          v. Find the minimum `robot_prep_cost` calculated in step 6.b.iv over all robots and all required positions `X`.
          vi. If there are no robots, or if no robot can reach any required position (e.g., disconnected grid, though unlikely), return a very large number (indicating an unsolvable state).
          vii. Add this minimum preparation cost to the total heuristic cost.
    7. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting grid structure, goal tiles,
        and required robot painting positions from static facts and objects.
        """
        self.goals = task.goals
        static_facts = task.static
        objects = task.objects

        self.tile_coords = {}
        # Regex to parse tile names like 'tile_row_col'
        tile_name_pattern = re.compile(r'tile_(\d+)_(\d+)')

        # Map tile names to coordinates (row, col)
        for obj_type, obj_list in objects.items():
            if obj_type == 'tile':
                for tile_name in obj_list:
                    match = tile_name_pattern.match(tile_name)
                    if match:
                        # PDDL tile_i_j seems to map to (row=i, col=j)
                        row, col = int(match.group(1)), int(match.group(2))
                        self.tile_coords[tile_name] = (row, col)

        # Store goal locations and colors for tiles that need painting
        self.goal_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_tiles[tile] = color

        # Store mapping from a tile (that needs to be painted) to the set of
        # tiles where a robot must be located to paint it using the
        # corresponding paint action (up, down, left, or right).
        # paint_positions[painted_tile] = {robot_pos_tile1, robot_pos_tile2, ...}
        self.paint_positions = {}
        for fact in static_facts:
            parts = get_parts(fact)
            # Check for adjacency facts that define paintable relationships
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                painted_tile, robot_pos_tile = parts[1], parts[2]
                # Example: (up tile_1_1 tile_0_1) means tile_1_1 is UP from tile_0_1.
                # A robot at tile_0_1 can paint tile_1_1 using paint_up.
                # So, tile_0_1 is a required position to paint tile_1_1.
                self.paint_positions.setdefault(painted_tile, set()).add(robot_pos_tile)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract current robot locations and colors
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                _, robot, tile = get_parts(fact)
                robot_locations[robot] = tile
            elif match(fact, "robot-has", "*", "*"):
                _, robot, color = get_parts(fact)
                robot_colors[robot] = color

        # Extract current painted tiles and their colors
        current_painted_tiles = {}
        for fact in state:
            if match(fact, "painted", "*", "*"):
                _, tile, color = get_parts(fact)
                current_painted_tiles[tile] = color

        # Check for wrongly painted goal tiles (dead end)
        for goal_tile, goal_color in self.goal_tiles.items():
            if goal_tile in current_painted_tiles and current_painted_tiles[goal_tile] != goal_color:
                # This goal tile is painted with the wrong color. Assuming this is a dead end.
                return 1000000 # Return a large value

        total_cost = 0

        # Calculate cost for each unpainted goal tile
        for goal_tile, goal_color in self.goal_tiles.items():
            # If the tile is already painted with the correct color, cost is 0 for this tile.
            if goal_tile in current_painted_tiles and current_painted_tiles[goal_tile] == goal_color:
                continue

            # This tile needs to be painted. Cost includes paint action + robot prep.
            cost_for_tile = 1 # Cost of the paint action itself

            # Find the set of tiles where a robot must be to paint this goal_tile
            required_pos_tiles = self.paint_positions.get(goal_tile, set())

            # If there are no positions from which this tile can be painted, it's unsolvable.
            if not required_pos_tiles:
                 # This tile is a goal but cannot be painted according to static facts.
                 return 1000000 # Return a large value

            min_robot_prep_cost = float('inf')

            # Find the minimum cost for any robot to reach any required position with the correct color
            for paint_pos_tile in required_pos_tiles:
                if paint_pos_tile not in self.tile_coords:
                     # Should not happen in valid problems, but defensive check
                     continue

                (X_r, X_c) = self.tile_coords[paint_pos_tile]

                for robot_name, robot_tile in robot_locations.items():
                    if robot_tile not in self.tile_coords:
                         # Should not happen in valid problems, but defensive check
                         continue

                    (Rr, Rc) = self.tile_coords[robot_tile]

                    # Manhattan distance from robot's current tile to the required painting position tile
                    dist = abs(Rr - X_r) + abs(Rc - X_c)

                    robot_prep_cost = dist

                    # Add cost for changing color if needed
                    robot_color = robot_colors.get(robot_name)
                    if robot_color != goal_color:
                        robot_prep_cost += 1 # Cost of change_color action

                    min_robot_prep_cost = min(min_robot_prep_cost, robot_prep_cost)

            # If min_robot_prep_cost is still infinity, it means there are no robots,
            # or robots exist but cannot reach any required position (e.g., disconnected grid).
            # This indicates an unsolvable state.
            if min_robot_prep_cost == float('inf'):
                 return 1000000 # Unreachable painting position or no robots

            total_cost += cost_for_tile + min_robot_prep_cost

        return total_cost

