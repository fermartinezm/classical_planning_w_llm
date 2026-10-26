from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re
import math

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct colors. It considers the number of tiles that need painting,
    the colors that need to be acquired by robots, and the movement cost for
    robots to reach tiles adjacent to the ones needing paint. It is designed
    for greedy best-first search and is not admissible.

    # Assumptions
    - Tiles are arranged in a grid structure, and tile names follow the pattern 'tile_R_C'.
    - The grid is connected according to the 'up', 'down', 'left', 'right' predicates.
    - If a tile is painted with the wrong color according to the goal, the state is considered unsolvable from that point.
    - Each paint action paints one tile.
    - Each color change action allows a robot to acquire a new color.
    - Movement cost is estimated using Manhattan distance on the grid.
    - The goal only consists of `painted` predicates.

    # Heuristic Initialization
    - Extracts goal conditions to identify which tiles need to be painted and with which colors.
    - Parses tile names from static facts and goals to build a mapping between tile names and grid coordinates (row, column).
    - Stores lists of all robots and colors available in the domain by parsing object names from static facts and goals.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all goal tiles that are not currently painted with the correct color.
    2. For each such tile:
       - If it is painted with a *wrong* color (different from the goal color), return a very large value (indicating an effectively unsolvable state in this domain).
       - If it is `clear`, add it to a list of tiles needing paint (`clear_goal_tiles`).
    3. If the list of tiles needing paint is empty, the goal (regarding painted tiles) is reached, so return 0.
    4. Initialize the heuristic value `h`.
    5. Add the number of tiles needing paint to `h`. This accounts for the minimum number of paint actions required.
    6. Calculate color cost:
       - Identify the set of distinct colors required for the tiles in `clear_goal_tiles`.
       - For each required color, check if any robot currently possesses that color (`robot-has`).
       - Add 1 to `h` for each required color that no robot currently possesses. This estimates the cost of acquiring the necessary colors via `change_color` actions.
    7. Calculate movement cost:
       - For each tile in `clear_goal_tiles`:
         - Find its grid coordinates (R, C).
         - Determine the coordinates of its potential adjacent tiles (R-1, C), (R+1, C), (R, C-1), (R, C+1).
         - Filter these to include only coordinates that correspond to actual tiles in the grid.
         - For each robot, calculate the Manhattan distance from its current location to *each* of the valid adjacent tile coordinates of the target tile.
         - Find the minimum of these distances over all robots and all valid adjacent tiles. This is the estimated minimum movement cost to get *any* robot into a position to paint this specific tile.
         - Add this minimum distance to the total heuristic value `h`.
    8. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Extract goal tile-color mapping
        self.goal_painted_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_painted_tiles[tile] = color

        # Extract all objects and categorize them by parsing names
        all_objects_set = set()
        for fact in self.static_facts:
             all_objects_set.update(get_parts(fact)[1:])
        for goal in self.goals:
             all_objects_set.update(get_parts(goal)[1:])
        # Note: Objects from initial state are not explicitly given in the problem description
        # format for __init__, but the Task object constructor implies knowledge of all objects.
        # Parsing names from static facts and goals is a practical approach here.

        self.all_tiles = {obj for obj in all_objects_set if obj.startswith('tile_')}
        self.all_robots = {obj for obj in all_objects_set if obj.startswith('robot')}
        # Get available colors from static facts
        self.all_colors = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "available-color", "*")}


        # Build tile coordinate mapping (assuming tile_R_C format)
        self.tile_coords = {}
        self.coords_tile = {}
        tile_pattern = re.compile(r'tile_(\d+)_(\d+)')
        for tile in self.all_tiles:
            match_obj = tile_pattern.match(tile)
            if match_obj:
                r, c = int(match_obj.group(1)), int(match_obj.group(2))
                self.tile_coords[tile] = (r, c)
                self.coords_tile[(r, c)] = tile
            # else: Tile name doesn't match pattern, cannot determine coordinates.
            # This heuristic relies on the pattern. If a tile doesn't match, it won't be
            # included in coordinate calculations, potentially affecting heuristic accuracy.
            # Assuming all relevant tiles follow the pattern based on examples.


    def get_tile_neighbors_coords(self, tile_name):
        """Get coordinates of potential neighbors for a given tile and filter for valid tiles."""
        if tile_name not in self.tile_coords:
            return [] # Should not happen for valid tiles in the grid

        r, c = self.tile_coords[tile_name]
        potential_neighbors_coords = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]

        # Filter for coordinates that map to actual tiles in the problem
        valid_neighbors_coords = [
            (nr, nc) for nr, nc in potential_neighbors_coords
            if (nr, nc) in self.coords_tile
        ]
        return valid_neighbors_coords

    def manhattan_distance(self, coord1, coord2):
        """Calculate Manhattan distance between two coordinates (r, c)."""
        return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached (assuming goal is only painted predicates)
        # A more robust check would be self.goals <= state, but the heuristic
        # calculation below specifically targets painted goals. If clear_goal_tiles
        # is empty, it implies all goal painted predicates are met.
        # Let's rely on the check below for efficiency in the common case.

        # Identify current robot locations and colors
        robot_locations = {}
        robot_colors = {}
        for robot in self.all_robots:
            for fact in state:
                if match(fact, "robot-at", robot, "*"):
                    robot_locations[robot] = get_parts(fact)[2]
                if match(fact, "robot-has", robot, "*"):
                    robot_colors[robot] = get_parts(fact)[2]

        # Identify currently painted tiles and clear tiles
        painted_tiles_state = {}
        clear_tiles_state = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "painted":
                painted_tiles_state[parts[1]] = parts[2]
            elif parts[0] == "clear":
                clear_tiles_state.add(parts[1])

        # Identify tiles that need painting according to the goal
        clear_goal_tiles = [] # List of (tile, goal_color)
        needed_colors = set()

        for tile, goal_color in self.goal_painted_tiles.items():
            if tile in painted_tiles_state:
                current_color = painted_tiles_state[tile]
                if current_color != goal_color:
                    # Tile is painted with the wrong color - likely unsolvable
                    return 1000000 # Large heuristic value
            elif tile in clear_tiles_state:
                # Tile is clear but needs painting
                clear_goal_tiles.append((tile, goal_color))
                needed_colors.add(goal_color)
            # else: tile is not clear and not painted with goal color.
            # This could mean it's painted with a different color (handled above)
            # or it's in an unexpected state (e.g., neither clear nor painted).
            # Assuming valid states are either clear or painted.

        # If no tiles need painting, all painted goals are met.
        # Assuming the overall goal is only painted predicates, return 0.
        if not clear_goal_tiles:
             return 0

        h = 0

        # 1. Cost for paint actions: Each clear goal tile needs one paint action.
        h += len(clear_goal_tiles)

        # 2. Cost for color changes: Estimate cost to get needed colors onto robots.
        colors_to_get = set()
        for color in needed_colors:
            has_color = False
            for robot in self.all_robots:
                if robot in robot_colors and robot_colors[robot] == color:
                    has_color = True
                    break
            if not has_color:
                colors_to_get.add(color)
        # Add 1 for each color that is needed for at least one clear tile,
        # but is not currently held by any robot.
        h += len(colors_to_get)

        # 3. Cost for movement: Estimate cost for robots to reach painting positions.
        # Sum the minimum distance for each clear goal tile from any robot to any adjacent tile.
        for tile, goal_color in clear_goal_tiles:
            # Ensure tile coordinates are known (should be if it's in self.all_tiles and matches pattern)
            if tile not in self.tile_coords:
                 # Cannot calculate movement for this tile, return large value
                 return 1000000

            tile_coord = self.tile_coords[tile]
            neighbor_coords = self.get_tile_neighbors_coords(tile)

            # If a tile has no valid neighbors (e.g., 1x1 grid), something is wrong
            if not neighbor_coords:
                 return 1000000

            min_dist_to_adj_tile = math.inf

            # Find the minimum distance from any robot to any adjacent tile of the target tile
            for robot in self.all_robots:
                if robot not in robot_locations:
                    # Robot location unknown, cannot calculate distance.
                    # This shouldn't happen in a valid state representation.
                    # Treat as unreachable for this robot, or return large value.
                    # Let's skip this robot for now, min_dist will remain high if no robots are found.
                    continue

                robot_loc_name = robot_locations[robot]
                if robot_loc_name not in self.tile_coords:
                     # Robot at a non-tile location? Should not happen.
                     continue # Skip this robot

                robot_coord = self.tile_coords[robot_loc_name]

                for adj_coord in neighbor_coords:
                    dist = self.manhattan_distance(robot_coord, adj_coord)
                    min_dist_to_adj_tile = min(min_dist_to_adj_tile, dist)

            # If min_dist_to_adj_tile is still infinity, it means no robots were found or
            # no valid paths could be calculated (e.g., no valid robot locations).
            if min_dist_to_adj_tile == math.inf:
                 # This indicates a problem state where no robot can reach the tile.
                 return 1000000

            h += min_dist_to_adj_tile

        return h
