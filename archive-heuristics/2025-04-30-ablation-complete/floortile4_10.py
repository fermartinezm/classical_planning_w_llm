from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Import math for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts defensively
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
         # Return empty list for malformed facts
         return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their specified colors. It sums, for each unpainted goal tile, the cost
    of painting it, which includes the paint action itself plus the minimum cost
    for any robot to get into position (adjacent to the tile) with the correct color.
    Movement cost is estimated using Manhattan distance on the grid coordinates
    derived from tile names.

    # Assumptions
    - Tile names follow the format 'tile_R_C' where R and C are integers representing
      row and column.
    - The grid structure is implicitly defined by 'up', 'down', 'left', 'right' predicates,
      forming a grid where 'up' typically decreases the row index, 'down' increases,
      'left' decreases the column index, and 'right' increases. The heuristic relies
      on parsing the tile names directly for coordinates.
    - All goal tiles are initially 'clear' or already painted with the correct color.
      If a goal tile is found to be painted with the wrong color in any state, the
      problem is considered unsolvable from that state (heuristic returns infinity).
    - Robots always possess a color.
    - Movement cost between adjacent tiles is 1. Color change cost is 1. Paint cost is 1.
    - Manhattan distance is used as an admissible estimate for movement cost on the grid,
      ignoring potential obstacles ('painted' tiles).

    # Heuristic Initialization
    - Parses all tile objects from the initial state and static facts to create
      a mapping between tile names and their (row, column) coordinates. This map
      is crucial for calculating Manhattan distances.
    - Extracts the required color for each goal tile from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location and color of each robot by parsing the state facts. Store this information, e.g., in a dictionary mapping robot names to their properties.
    2. Identify the set of tiles that are currently clear by parsing the state facts.
    3. Initialize the total heuristic cost to 0.
    4. Iterate through each goal condition of the form `(painted tile_name required_color)` that was extracted during initialization.
    5. For the current goal tile `tile_name` and its `required_color`:
        a. Check if the goal predicate `(painted tile_name required_color)` is already present in the current state. If it is, this goal is satisfied for this tile, and no further cost is added for this tile. Continue to the next goal tile.
        b. If the goal is not satisfied for this tile, check if `(clear tile_name)` is present in the current state.
        c. If `(clear tile_name)` is *not* present in the state (and the goal is not met), it implies the tile is painted with a color different from the `required_color`. Since there are no actions to unpaint or repaint a tile, this state is effectively a dead end for achieving this specific goal. The problem is considered unsolvable from this state. Return `float('inf')`.
        d. If `(clear tile_name)` is present in the state, this tile needs to be painted. The cost to paint this tile involves:
            i. The paint action itself, which costs 1.
            ii. Getting a robot into a state where it can perform the paint action. This requires a robot to be at a tile adjacent to `tile_name` and to be holding the `required_color`. Calculate the minimum cost for *any* robot to reach this ready state:
                - For each robot identified in step 1:
                    - Get the robot's current location (`R_loc`) and color (`R_color`).
                    - Calculate the minimum Manhattan distance from `R_loc` to *any* tile that is adjacent to `tile_name`. This minimum distance represents the minimum number of move actions required.
                    - Calculate the color change cost: 1 if the robot's current color `R_color` is not the `required_color`, and 0 otherwise.
                    - The total cost for this specific robot to become ready to paint `tile_name` is the calculated minimum movement cost plus the color change cost.
                - Find the minimum of these "robot ready costs" over all available robots. This is `min_robot_ready_cost`.
            iii. The total cost estimated for achieving the goal for this specific tile is `1 (paint action cost) + min_robot_ready_cost`.
            iv. Add this calculated cost for the current tile to the `total_cost`.
    6. After iterating through all goal tiles, the accumulated `total_cost` is the heuristic estimate for the current state. Return `total_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting tile coordinates and goal colors.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Map tile names to (row, col) coordinates and vice versa
        self.tile_name_to_coords = {}
        self.coords_to_tile_name = {}

        # Extract all potential tile objects from initial state and static facts
        # We look for objects used in predicates that typically involve tiles
        tile_related_predicates = {"up", "down", "left", "right", "clear", "painted", "robot-at"}
        all_potential_tiles = set()
        for fact in task.initial_state | task.static_facts:
             parts = get_parts(fact)
             if parts and parts[0] in tile_related_predicates:
                 # Add all arguments of tile-related predicates as potential tiles
                 all_potential_tiles.update(parts[1:])

        # Filter objects that look like tiles and parse coordinates
        for obj in all_potential_tiles:
            if obj.startswith('tile_'):
                try:
                    # Tile names are like tile_R_C
                    parts = obj.split('_')
                    if len(parts) == 3:
                        row = int(parts[1])
                        col = int(parts[2])
                        self.tile_name_to_coords[obj] = (row, col)
                        self.coords_to_tile_name[(row, col)] = obj
                except ValueError:
                    # Ignore objects that start with 'tile_' but aren't in the expected format
                    pass # Or log a warning

        # Store goal locations and required colors for each tile
        self.goal_colors = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                # Ensure the tile object exists in our parsed tiles
                if args and args[0] in self.tile_name_to_coords:
                    tile, color = args
                    self.goal_colors[tile] = color
                # else: Log a warning about a goal tile not found in facts?

    def get_manhattan_distance(self, tile1_name, tile2_name):
        """Calculate Manhattan distance between two tiles."""
        coords1 = self.tile_name_to_coords.get(tile1_name)
        coords2 = self.tile_name_to_coords.get(tile2_name)
        if coords1 is None or coords2 is None:
            # This indicates an issue if tile names are expected to be valid
            # Returning infinity implies unreachable
            return float('inf')
        r1, c1 = coords1
        r2, c2 = coords2
        return abs(r1 - r2) + abs(c1 - c2)

    def get_adjacent_tiles(self, tile_name):
        """Get names of tiles adjacent to the given tile based on coordinate parsing."""
        coords = self.tile_name_to_coords.get(tile_name)
        if coords is None:
            return [] # Should not happen for valid tile names

        r, c = coords
        # Potential adjacent coordinates based on grid structure
        potential_adj_coords = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        adjacent_tiles = []
        for adj_r, adj_c in potential_adj_coords:
            # Check if a tile exists at these coordinates in the problem instance
            adj_tile_name = self.coords_to_tile_name.get((adj_r, adj_c))
            if adj_tile_name:
                adjacent_tiles.append(adj_tile_name)
        return adjacent_tiles


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse robot locations and colors from the current state
        robot_info = {} # {robot_name: {'location': tile_name, 'color': color_name}}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                _, robot, location = get_parts(fact)
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['location'] = location
            elif match(fact, "robot-has", "*", "*"):
                _, robot, color = get_parts(fact)
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['color'] = color

        # Identify clear tiles for quick lookup
        clear_tiles = {get_parts(fact)[1] for fact in state if match(fact, "clear", "*")}

        total_cost = 0

        # Iterate through each goal tile and its required color
        for tile, required_color in self.goal_colors.items():
            # Check if the goal for this tile is already satisfied
            if f"(painted {tile} {required_color})" in state:
                continue

            # If goal not satisfied, check if it's painted with the wrong color
            # We iterate through all painted facts for this tile
            is_wrongly_painted = False
            for fact in state:
                if match(fact, "painted", tile, "*"):
                    painted_color = get_parts(fact)[2]
                    if painted_color != required_color:
                        is_wrongly_painted = True
                        break # Found wrong color, no need to check other painted facts for this tile

            if is_wrongly_painted:
                # Problem is unsolvable if a goal tile is painted with the wrong color
                return float('inf')

            # If the tile is clear and needs painting
            if tile in clear_tiles:
                # Cost includes the paint action (1) + cost to get a robot ready
                paint_action_cost = 1

                min_robot_ready_cost = float('inf')

                # Calculate the minimum cost for any robot to get ready for this tile
                adjacent_tiles = self.get_adjacent_tiles(tile)

                # If a tile has no adjacent tiles (unlikely in typical grid problems,
                # but defensive check), it cannot be painted by moving to an adjacent tile.
                # If the robot is already AT the tile, it still can't paint itself.
                # So, if no adjacent tiles exist, this goal is likely unreachable.
                if not adjacent_tiles:
                     return float('inf') # Cannot paint a tile with no adjacent tiles

                # Calculate the minimum cost for each robot to get ready
                for robot, info in robot_info.items():
                    robot_location = info.get('location')
                    robot_color = info.get('color')

                    # Ensure robot info is complete (should be if parsed correctly)
                    if robot_location is None or robot_color is None:
                         continue # Skip this robot if its state is incomplete

                    # Calculate minimum movement cost to an adjacent tile
                    min_move_cost = float('inf')
                    for adj_tile in adjacent_tiles:
                        move_cost = self.get_manhattan_distance(robot_location, adj_tile)
                        min_move_cost = min(min_move_cost, move_cost)

                    # Calculate color change cost
                    color_change_cost = 0 if robot_color == required_color else 1

                    # Total cost for this robot to be ready for this specific tile
                    robot_ready_cost = min_move_cost + color_change_cost

                    # Update the minimum ready cost across all robots
                    min_robot_ready_cost = min(min_robot_ready_cost, robot_ready_cost)

                # Add the cost for this tile if a robot can reach it and get the color
                if min_robot_ready_cost != float('inf'):
                    total_cost += paint_action_cost + min_robot_ready_cost
                else:
                    # If no robot can reach an adjacent tile or get the color,
                    # this specific goal tile cannot be painted.
                    return float('inf') # Problem is unsolvable from this state

        # Return the accumulated cost for all unpainted goal tiles
        return total_cost

