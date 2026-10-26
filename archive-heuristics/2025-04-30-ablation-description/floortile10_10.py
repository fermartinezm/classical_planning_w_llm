import re
from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a given pattern."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_grid_info(static_facts):
    """
    Parses static facts to build grid structure and adjacency list.
    Assumes tile names are in the format 'tile_R_C'.
    """
    tile_coords = {}
    adj_list = {}
    tile_pattern = re.compile(r'tile_(\d+)_(\d+)')

    # First pass: Identify all tiles and their coordinates
    for fact in static_facts:
        parts = get_parts(fact)
        for part in parts:
            match_coord = tile_pattern.match(part)
            if match_coord and part not in tile_coords:
                try:
                    row, col = int(match_coord.group(1)), int(match_coord.group(2))
                    tile_coords[part] = (row, col)
                    adj_list[part] = set() # Initialize adjacency set
                except ValueError:
                    # Should not happen with the regex, but good practice
                    pass


    # Second pass: Build adjacency list from directional facts
    for fact in static_facts:
        parts = get_parts(fact)
        if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
            tile1, tile2 = parts[1], parts[2]
            # Ensure both are valid tiles found in the first pass
            if tile1 in adj_list and tile2 in adj_list:
                 adj_list[tile1].add(tile2)
                 adj_list[tile2].add(tile1) # Adjacency is symmetric

    return tile_coords, adj_list

def manhattan_distance(tile1, tile2, tile_coords):
    """Calculates Manhattan distance between two tiles."""
    if tile1 not in tile_coords or tile2 not in tile_coords:
        # This indicates an issue with tile names not found during parsing
        # or attempting to calculate distance for non-tile objects.
        # For heuristic purposes, return infinity or a large number.
        # In a well-formed problem, this path shouldn't be taken for tiles.
        return float('inf')
    r1, c1 = tile_coords[tile1]
    r2, c2 = tile_coords[tile2]
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing up
    the estimated cost for each individual goal tile that is not yet painted
    correctly. The estimated cost for a single tile includes the paint action
    itself, plus the minimum cost to get any robot to the correct adjacent
    location with the required color.

    # Assumptions
    - Goal tiles are initially either clear or painted with the correct color.
      (The domain structure prevents repainting or moving onto painted tiles).
    - Tile names follow the format 'tile_R_C' allowing grid coordinate parsing.
    - Manhattan distance is a reasonable estimate for movement cost on the grid.
    - Changing color costs 1 action.
    - Getting a robot to an adjacent tile is estimated by the minimum Manhattan
      distance from the robot's current location to any tile adjacent to the target tile.
    - The heuristic returns infinity if a goal tile is painted with the wrong color
      or if a goal tile has no adjacent tiles (making it impossible to paint).
    - Assumes robots always have a color predicate (`robot-has` or implicitly `free-color`
      if the domain used it, though `change_color` requires `robot-has`). The heuristic
      simplifies color cost to 1 if the robot doesn't have the required color.

    # Heuristic Initialization
    - Parses static facts to build a grid representation: mapping tile names
      to (row, column) coordinates and building an adjacency list.
    - Extracts the goal conditions, specifically the required color for each
      goal tile.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Identify the set of goal tiles that are not currently painted with the
       correct color according to the goal state.
    3. For each such unpainted goal tile (let's call it `goal_tile`) that needs
       to be painted with `required_color`:
        a. Check if `goal_tile` is currently painted with a *different* color.
           If so, the state is likely unsolvable in this domain, return a very
           large number (infinity).
        b. If `goal_tile` is not painted with the required color (and is assumed
           to be clear based on domain structure):
            i. Add 1 to the total heuristic cost (for the paint action itself).
            ii. Calculate the minimum cost to get *any* robot ready to paint
                this specific `goal_tile` with `required_color`.
                - Find the current location and color of each robot in the state.
                - Find all tiles adjacent to `goal_tile` using the precomputed
                  adjacency information.
                - If `goal_tile` has no adjacent tiles, return infinity (cannot paint).
                - For each robot:
                    - Calculate the color cost: 1 if the robot does not currently
                      have `required_color`, 0 otherwise. (Assumes robot can change color).
                    - Calculate the minimum movement cost: Find the minimum
                      Manhattan distance from the robot's current location to
                      any tile adjacent to `goal_tile`.
                    - The cost for this robot is `color_cost + minimum_movement_cost`.
                - The minimum robot cost for `goal_tile` is the minimum of these
                  costs over all robots.
                - If no robot can reach an adjacent tile (e.g., no robots exist or grid disconnected), return infinity.
            iii. Add this minimum robot cost to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Parse grid information from static facts
        self.tile_coords, self.adj_list = parse_grid_info(static_facts)

        # Store goal colors for each goal tile
        self.goal_colors = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'painted' and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_colors[tile] = color

    def __call__(self, node):
        """Estimate the minimum cost to reach the goal state from the current state."""
        state = node.state
        total_cost = 0

        # Find current robot locations and colors
        robot_info = {} # {robot_name: {'location': tile_name, 'color': color_name}}
        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == 'robot-at' and len(parts) == 3:
                    robot, location = parts[1], parts[2]
                    if robot not in robot_info:
                        robot_info[robot] = {}
                    robot_info[robot]['location'] = location
                elif parts[0] == 'robot-has' and len(parts) == 3:
                    robot, color = parts[1], parts[2]
                    if robot not in robot_info:
                        robot_info[robot] = {}
                    robot_info[robot]['color'] = color
                # Note: Does not handle 'free-color' explicitly, assumes robot-has is present or needed

        # Identify unpainted goal tiles
        unpainted_goal_tiles = {} # {tile_name: required_color}
        for goal_tile, required_color in self.goal_colors.items():
            goal_fact = f'(painted {goal_tile} {required_color})'
            if goal_fact not in state:
                # Check if it's painted with the wrong color
                painted_wrong_color = False
                for fact in state:
                    parts = get_parts(fact)
                    if parts and parts[0] == 'painted' and len(parts) == 3 and parts[1] == goal_tile:
                         # It's painted, but not with the required color
                         painted_wrong_color = True
                         break
                if painted_wrong_color:
                    # Cannot repaint in this domain structure
                    return float('inf') # State is likely unsolvable

                # If not painted correctly and not painted wrong, it needs painting
                unpainted_goal_tiles[goal_tile] = required_color

        # Calculate cost for each unpainted goal tile
        for goal_tile, required_color in unpainted_goal_tiles.items():
            # Cost for the paint action itself
            tile_cost = 1

            # Cost to get a robot ready for this tile
            min_robot_cost_for_tile = float('inf')

            adjacent_tiles = self.adj_list.get(goal_tile, set())

            if not adjacent_tiles:
                 # This goal tile has no adjacent tiles, cannot be painted
                 return float('inf') # State is likely unsolvable

            # If there are no robots, no tile can be painted
            if not robot_info:
                 return float('inf')

            for robot, info in robot_info.items():
                robot_location = info.get('location')
                robot_color = info.get('color') # Can be None if robot has free-color or no color fact

                if robot_location is None:
                    # Robot location unknown, cannot calculate cost
                    continue # Skip this robot

                # Cost to get the correct color
                # If robot_color is None (e.g., free-color), it needs to acquire the color (cost 1)
                # If robot_color is present but wrong, it needs to change color (cost 1)
                # If robot_color is present and correct, cost is 0
                color_cost = 1 if robot_color != required_color else 0

                # Cost to move to an adjacent tile
                min_move_cost = float('inf')
                for adj_tile in adjacent_tiles:
                    move_cost = manhattan_distance(robot_location, adj_tile, self.tile_coords)
                    min_move_cost = min(min_move_cost, move_cost)

                if min_move_cost != float('inf'):
                     # Total cost for this robot to paint this tile
                     robot_total_cost = color_cost + min_move_cost
                     min_robot_cost_for_tile = min(min_robot_cost_for_tile, robot_total_cost)

            if min_robot_cost_for_tile == float('inf'):
                 # No robot can reach an adjacent tile (e.g. grid is disconnected or no robots)
                 return float('inf') # State is likely unsolvable

            tile_cost += min_robot_cost_for_tile
            total_cost += tile_cost

        return total_cost
