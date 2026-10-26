from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    with the correct colors. It considers the cost of painting, robot movement,
    color changes, and penalties for incorrectly painted tiles or occupied goal tiles.

    # Assumptions
    - Tiles are arranged in a grid, and tile names follow the format 'tile_row_col'.
    - Movement is restricted to adjacent tiles on the grid.
    - Robots can only move onto 'clear' tiles.
    - Painting a tile makes it 'not clear'.
    - If a goal tile is painted with the wrong color, it cannot be repainted (dead end for that tile).
    - All colors needed in the goal are available.

    # Heuristic Initialization
    - Extracts the goal conditions (which tiles need which colors).
    - Builds an adjacency map of the tile grid from the static 'up', 'down', 'left', 'right' facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize heuristic value `h = 0`.
    2. Identify all goal predicates `(painted T C)`.
    3. Categorize goal tiles:
       - `wrongly_painted_tiles`: Tiles `T` where `(painted T C_goal)` is a goal, but `(painted T C')` is true in the state for `C' != C_goal`. Add a large penalty to `h` for each such tile.
       - `unpainted_goals`: Tiles `T` where `(painted T C_goal)` is a goal, but no `(painted T C')` is true in the state.
    4. Further categorize `unpainted_goals`:
       - `unpainted_occupied_goals`: Tiles `T` in `unpainted_goals` that are not `clear` (and not painted), implying a robot is on them. Add 1 to `h` for each such tile (cost for the robot to move off).
       - `unpainted_clear_goals`: Tiles `T` in `unpainted_goals` that are `clear`. These are ready to be painted once a robot with the correct color arrives.
    5. Identify colors needed for `unpainted_clear_goals`.
    6. Identify colors currently held by robots.
    7. Calculate `colors_to_acquire`: colors needed for `unpainted_clear_goals` that no robot currently holds. Add the number of such colors to `h` (cost for robots to change color).
    8. For each tile `(T, C_goal)` in `unpainted_clear_goals`:
       - Add 1 to `h` (cost for the paint action).
       - Calculate the minimum Manhattan distance from any robot's current location to any tile adjacent to `T`. Add this minimum distance to `h` (cost for robot movement).
    9. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Store goal colors for each tile
        self.goal_colors = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted":
                if len(parts) == 3:
                    _, tile, color = parts
                    self.goal_colors[tile] = color

        # Build adjacency map for distance calculation
        self.adj_map = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] in ['up', 'down', 'left', 'right']:
                if len(parts) == 3: # Ensure fact has correct number of parts
                    tile1, tile2 = parts[1], parts[2]
                    self.adj_map.setdefault(tile1, set()).add(tile2)
                    self.adj_map.setdefault(tile2, set()).add(tile1)

    def get_adjacent_tiles(self, tile):
        """Get the set of tiles adjacent to the given tile."""
        return self.adj_map.get(tile, set())

    def get_coords(self, tile_name):
        """Parse tile name 'tile_row_col' to get (row, col) coordinates."""
        try:
            # Assuming tile names are consistently 'tile_row_col'
            parts = tile_name.split('_')
            if len(parts) == 3 and parts[0] == 'tile':
                 return int(parts[1]), int(parts[2])
            # print(f"Warning: Unexpected tile name format: {tile_name}")
            return None # Indicate failure
        except (ValueError, IndexError):
            # print(f"Warning: Could not parse coordinates from tile name: {tile_name}")
            return None # Indicate failure


    def manhattan_distance(self, tile1, tile2):
        """Calculate Manhattan distance between two tiles based on coordinates."""
        coords1 = self.get_coords(tile1)
        coords2 = self.get_coords(tile2)
        if coords1 is None or coords2 is None:
             # If coordinates cannot be parsed, return a large distance or handle differently
             # Returning a large number effectively makes paths involving these tiles very expensive.
             return 1000000 # Large distance

        r1, c1 = coords1
        r2, c2 = coords2
        return abs(r1 - r2) + abs(c1 - c2)

    def min_dist_to_adjacent(self, robot_location, target_tile):
        """Find the minimum Manhattan distance from robot_location to any tile adjacent to target_tile."""
        min_d = float('inf')
        adjacent_tiles = self.get_adjacent_tiles(target_tile)

        if not adjacent_tiles:
             # If target_tile has no adjacent tiles in the map, it cannot be painted.
             # This state is likely a dead end or problem definition issue.
             return 1000000 # Large penalty for unpaintable tile

        for adj_tile in adjacent_tiles:
            min_d = min(min_d, self.manhattan_distance(robot_location, adj_tile))

        # The distance calculated is the number of moves to reach the adjacent tile.
        # If robot is already adjacent, distance is 0.
        return min_d if min_d != float('inf') else 1000000 # Return large penalty if no adjacent tile reachable (e.g., disconnected graph)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        h = 0
        unpainted_goals = set()
        wrongly_painted_tiles = set()
        painted_tiles_in_state = {} # {tile: color} for tiles painted in the current state

        # Identify painted tiles in the current state
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "painted":
                if len(parts) == 3:
                    _, tile, color = parts
                    painted_tiles_in_state[tile] = color

        # Categorize goal tiles
        for goal_tile, goal_color in self.goal_colors.items():
            if goal_tile in painted_tiles_in_state:
                current_color = painted_tiles_in_state[goal_tile]
                if current_color != goal_color:
                    wrongly_painted_tiles.add(goal_tile)
                # If current_color == goal_color, the goal is met for this tile, cost is 0.
            else:
                # Tile is not painted at all (or painted with an unknown predicate, unlikely)
                unpainted_goals.add((goal_tile, goal_color))

        # Add penalty for wrongly painted tiles
        h += len(wrongly_painted_tiles) * 1000 # Large penalty

        # Categorize unpainted goals based on 'clear' status
        unpainted_clear_goals = set()
        unpainted_occupied_goals = set() # Tiles that are not clear and not painted (implies occupied by robot)

        clear_tiles_in_state = {get_parts(fact)[1] for fact in state if match(fact, "clear", "*")}

        for tile, color in unpainted_goals:
            if tile in clear_tiles_in_state:
                unpainted_clear_goals.add((tile, color))
            else:
                # If not clear and not painted (checked above), assume occupied by a robot
                unpainted_occupied_goals.add((tile, color))

        # Add cost for robots to move off occupied goal tiles
        h += len(unpainted_occupied_goals) # Each occupied goal tile needs at least one move action by the occupying robot

        # Identify robot states
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "robot-at" and len(parts) == 3:
                    _, robot, location = parts
                    robot_locations[robot] = location
                elif parts[0] == "robot-has" and len(parts) == 3:
                    _, robot, color = parts
                    robot_colors[robot] = color

        # Identify colors needed for unpainted clear goals that no robot currently holds
        needed_colors_for_clear_goals = {color for tile, color in unpainted_clear_goals}
        held_colors = set(robot_colors.values())
        colors_to_acquire = needed_colors_for_clear_goals - held_colors
        h += len(colors_to_acquire) # Cost for changing color for each needed color not held by *any* robot.

        # Add cost for painting and movement for unpainted clear goals
        for tile, goal_color in unpainted_clear_goals:
            h += 1 # Cost for the paint action

            # Find minimum movement cost from any robot to get adjacent to the tile
            min_move_cost = float('inf')
            if not robot_locations: # Should not happen in valid problems, but safety check
                 min_move_cost = 1000000 # Cannot paint if no robot exists

            for robot, location in robot_locations.items():
                 dist = self.min_dist_to_adjacent(location, tile)
                 min_move_cost = min(min_move_cost, dist)

            if min_move_cost == float('inf'):
                 # This can happen if a tile has no adjacent tiles in the map, or robot_location is invalid.
                 # Indicates an unpaintable tile due to structural issue.
                 h += 1000000 # Large penalty
            else:
                 h += min_move_cost

        return h
