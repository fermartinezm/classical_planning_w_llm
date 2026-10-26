from heuristics.heuristic_base import Heuristic
import math

# Helper functions

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def parse_tile_name(tile_name):
    """Parses 'tile_R_C' into (R, C) tuple of integers."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            pass # Not a valid tile name format
    raise ValueError(f"Invalid tile name format: {tile_name}")

def manhattan_distance(coord1, coord2):
    """Calculates Manhattan distance between two (row, col) tuples."""
    r1, c1 = coord1
    r2, c2 = coord2
    return abs(r1 - r2) + abs(c1 - c2)

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the required number of actions to paint all goal tiles
    correctly. It sums up the estimated cost for each unpainted goal tile independently,
    considering the paint action itself, the cost to clear the tile if occupied,
    the movement cost for the closest robot to reach an adjacent tile, and the
    color change cost if the closest robot does not have the required color.

    # Assumptions
    - The tile names follow the format 'tile_R_C' where R and C are integers representing
      row and column.
    - Robots always hold exactly one color.
    - The grid structure allows movement between adjacent tiles (Manhattan distance applies).
    - Robot-robot conflicts and complex tile clearing scenarios are simplified (e.g.,
      clearing a tile costs a fixed amount, ignoring where the robot moves).
    - The heuristic is non-admissible and designed for greedy best-first search.

    # Heuristic Initialization
    - Extracts the goal conditions.
    - Identifies all tile objects from the task definition and parses their row/column
      coordinates to build a mapping from tile name to (row, col) tuple.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Unpainted Goal Tiles:
       - Create a set of (tile, color) pairs for all `(painted tile color)` facts
         present in the goal state but not in the current state.

    2. Initialize Heuristic Value:
       - Set the total heuristic cost `h` to 0.

    3. Get Current Robot Information:
       - Determine the current location and held color for each robot from the state facts.

    4. Get Tile Clear Status:
       - Determine for each tile whether it is currently `clear`.

    5. Iterate Through Unpainted Goal Tiles:
       - For each `(tile_T, color_C)` pair in the set of unpainted goal tiles:

         a. Add Base Paint Cost:
            - Add 1 to `h` for the `paint` action required for this tile.

         b. Add Clearing Cost:
            - If `tile_T` is not `clear` in the current state, it means a robot
              is occupying it. Add 1 to `h` as an estimated cost to move the
              occupying robot off the tile.

         c. Calculate Minimum Movement Cost to Adjacent Tile:
            - Find the minimum Manhattan distance from any robot's current location
              to the target tile `tile_T`.
            - The number of moves required for a robot at distance `dist` to reach
              an adjacent tile is `max(0, dist - 1)`.
            - Identify the robot that minimizes this movement cost and its current color.
            - Add this minimum number of moves to `h`.

         d. Add Color Change Cost:
            - If the color held by the closest robot (identified in the previous step)
              is not `color_C`, add 1 to `h` as an estimated cost for a `change_color`
              action. This assumes the closest robot will be the one to paint the tile.

    6. Return Total Heuristic Value:
       - The final value of `h` is the estimated total cost. If there are no
         unpainted goal tiles, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and tile coordinates.
        """
        self.goals = task.goals  # Goal conditions.

        # Build mapping from tile name to (row, col) coordinates.
        self.tile_coords = {}
        # task.facts contains all ground facts mentioned in the domain/problem,
        # including type predicates like (tile tile_0_1).
        for fact in task.facts:
            parts = get_parts(fact)
            if len(parts) == 2 and parts[0] == 'tile':
                tile_name = parts[1]
                try:
                    row, col = parse_tile_name(tile_name)
                    self.tile_coords[tile_name] = (row, col)
                except ValueError:
                    # Ignore objects that are not in the expected tile_R_C format
                    pass

        self.all_tiles = list(self.tile_coords.keys())

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.
        """
        state = node.state  # Current world state (frozenset of fact strings).

        # 1. Identify Unpainted Goal Tiles
        unpainted_goal_tiles = set()
        for goal_fact in self.goals:
            goal_parts = get_parts(goal_fact)
            if goal_parts and goal_parts[0] == 'painted' and len(goal_parts) == 3:
                tile_T, color_C = goal_parts[1], goal_parts[2]
                # Check if the tile is NOT painted with the correct color in the current state
                # We assume solvable problems don't have tiles painted with the wrong color
                # if they are goal tiles. So we only check if the exact goal fact is missing.
                if goal_fact not in state:
                     unpainted_goal_tiles.add((tile_T, color_C))

        # 2. Initialize Heuristic Value
        if not unpainted_goal_tiles:
            return 0 # Goal reached

        h = 0

        # 3. Get Current Robot Information
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            if parts[0] == 'robot-at' and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts[0] == 'robot-has' and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        robots_info = {
            r: {'loc': robot_locations[r], 'color': robot_colors[r]}
            for r in robot_locations if r in robot_colors # Ensure we have both location and color for the robot
        }

        # 4. Get Tile Clear Status
        tile_clear_status = {
            tile: (f'(clear {tile})' in state)
            for tile in self.all_tiles
        }

        # 5. Iterate Through Unpainted Goal Tiles
        for tile_T, color_C in unpainted_goal_tiles:
            # a. Add Base Paint Cost
            h += 1

            # b. Add Clearing Cost
            if not tile_clear_status.get(tile_T, True): # Default to True if tile_T not in status (shouldn't happen if all_tiles is correct)
                h += 1 # Cost to clear the tile

            # c. Calculate Minimum Movement Cost to Adjacent Tile
            min_moves_to_adj = float('inf')
            closest_robot_color = None

            target_coords = self.tile_coords.get(tile_T)
            if target_coords is None:
                 # This goal tile doesn't exist in our tile map, likely an invalid problem
                 # Treat as unreachable or assign high cost.
                 return float('inf') # Or a large number

            for robot_R, info in robots_info.items():
                robot_loc = info['loc']
                robot_coords = self.tile_coords.get(robot_loc)

                if robot_coords is None:
                    # Robot is at a location not in our tile map, invalid state?
                    continue # Skip this robot

                dist = manhattan_distance(robot_coords, target_coords)
                # Moves required to get from robot_loc to a tile adjacent to tile_T.
                # If dist is 0 (robot is on tile_T), it needs to move off (cost covered by !clear)
                # and then move back to an adjacent tile (cost 1). The heuristic simplifies
                # this movement cost to 0 here, relying on the !clear cost.
                # If dist is 1 (robot is adjacent), moves_to_adj = 0.
                # If dist > 1, moves_to_adj = dist - 1.
                moves_to_adj = max(0, dist - 1)

                if moves_to_adj < min_moves_to_adj:
                    min_moves_to_adj = moves_to_adj
                    closest_robot_color = info['color']

            # If there are no robots, min_moves_to_adj remains infinity.
            if min_moves_to_adj == float('inf'):
                 return float('inf') # Cannot paint if no robots exist

            h += min_moves_to_adj # Cost to move closest robot

            # d. Add Color Change Cost
            if closest_robot_color != color_C:
                h += 1 # Cost for color change

        # 6. Return Total Heuristic Value
        return h
