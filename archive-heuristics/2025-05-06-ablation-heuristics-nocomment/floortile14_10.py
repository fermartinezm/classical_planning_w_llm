from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math

def get_parts(fact):
    """Helper function to parse a PDDL fact string into a list of parts."""
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """Helper function to check if a fact matches a pattern."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing the
    estimated cost for each unpainted goal tile. For each unpainted goal tile,
    it calculates the minimum cost among all robots to paint that tile.
    The cost for a robot to paint a tile is estimated as the sum of:
    1. Cost to change color if the robot doesn't have the required color (1 if needed, 0 otherwise).
    2. Manhattan distance from the robot's current location to any tile adjacent to the target tile.
    3. Cost of the paint action (1).
    This heuristic is non-admissible as it independently calculates the cost for
    each unpainted tile, potentially double-counting robot movement or color
    changes if a single robot paints multiple tiles. However, it aims to
    prioritize states where unpainted tiles are closer to robots with the
    correct color, guiding a greedy best-first search.

    Assumptions:
    - The tiles form a rectangular grid, and tile names follow the format 'tile_row_col'.
    - Manhattan distance is a reasonable approximation of movement cost, ignoring obstacles (painted tiles).
    - Robots always start with a specific color (not 'free-color').
    - Tiles that need to be painted according to the goal are either currently clear or already painted with the correct color. Tiles painted with the wrong color are not considered fixable within the domain actions and are ignored by the heuristic (they don't contribute to the positive heuristic value).

    Heuristic Initialization:
    - Parses static facts and initial state facts to identify all tile objects.
    - Parses tile names ('tile_row_col') to build a mapping from tile names to (row, col) integer coordinates.
    - Parses static facts ('up', 'down', 'left', 'right') to build an adjacency map for tiles.
    - Stores the goal facts, specifically identifying which tiles need to be painted and with which color.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize the total heuristic value to 0.
    2. Identify the set of 'unpainted goal tiles'. These are tiles 't' such that '(painted t c)' is a goal fact, but '(painted t c)' is NOT present in the current state, and '(clear t)' IS present in the current state.
    3. If there are no unpainted goal tiles (that are clear and need painting), the state satisfies this part of the goal, so the heuristic contribution from painting is 0. Return 0.
    4. Determine the current location and color for each robot from the current state.
    5. For each unpainted goal tile (tile_name, required_color) identified in step 2:
        a. Find the set of tiles adjacent to tile_name using the precomputed adjacency map.
        b. Get the (row, col) coordinates for tile_name and all its adjacent tiles using the precomputed coordinate map. Handle cases where adjacent tiles might not be in the coordinate map (e.g., boundary tiles).
        c. Initialize the minimum cost to paint this specific tile (min_cost_for_tile) to infinity.
        d. For each robot:
            i. Get the robot's current location (robot_loc) and color (robot_color). Skip if robot info is incomplete.
            ii. Calculate the color change cost: 1 if robot_color is not required_color, otherwise 0.
            iii. Calculate the minimum Manhattan distance from the robot's current coordinates to the coordinates of any adjacent tile found in step 5b. Initialize min_move_cost for this robot to infinity. For each adjacent tile's coordinates (ar, ac), calculate Manhattan distance = abs(robot_row - ar) + abs(robot_col - ac) and update min_move_cost. Handle cases where the robot's location is not in the coordinate map.
            iv. If a reachable adjacent tile was found (min_move_cost is not infinity), calculate the total estimated cost for this robot to paint this tile: color_change_cost + min_move_cost + 1 (for the paint action itself).
            v. Update min_cost_for_tile with the minimum of its current value and the cost calculated in step 5d.iv.
        e. If min_cost_for_tile is still infinity after checking all robots, it means this required tile cannot be painted by any robot in this state. This state is likely unsolvable. Return infinity.
        f. Add min_cost_for_tile to the total heuristic value.
    6. Return the total heuristic value.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # Map tile name to (row, col) coordinates
        self.tile_coords = {}
        # Map tile name to set of adjacent tile names
        self.adj_map = {}

        # Collect all objects to find tiles
        # Objects are listed in the instance file, but we can also find them
        # by looking at predicate arguments in initial state and static facts.
        all_objects = set()
        for fact in task.initial_state | static_facts:
             all_objects.update(get_parts(fact)[1:])

        # Identify tiles and parse coordinates
        # Assuming tile names are like 'tile_R_C'
        for obj in all_objects:
            if obj.startswith('tile_'):
                try:
                    parts = obj.split('_')
                    # PDDL objects are case-insensitive, but planner internal
                    # representation might preserve case. Let's assume lowercase.
                    row = int(parts[1])
                    col = int(parts[2])
                    self.tile_coords[obj] = (row, col)
                    self.adj_map[obj] = set() # Initialize adjacency set
                except (ValueError, IndexError):
                    # Not a tile_R_C format, ignore
                    pass

        # Build adjacency map from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right']:
                # (predicate tile_y tile_x) where tile_y is in the direction from tile_x
                # e.g., (up tile_1_1 tile_0_1) means tile_1_1 is up from tile_0_1
                # Adjacency is symmetric for movement/painting
                tile1 = parts[1]
                tile2 = parts[2]
                if tile1 in self.adj_map:
                    self.adj_map[tile1].add(tile2)
                if tile2 in self.adj_map:
                    self.adj_map[tile2].add(tile1)

        # Store goal tiles and their required colors
        self.goal_paintings = {} # Map tile name to required color
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color

    def __call__(self, node):
        state = node.state

        # 1. Identify unpainted goal tiles that are clear
        unpainted_goal_tiles = [] # List of (tile_name, required_color)
        current_painted = {} # Map tile to color if painted
        current_clear = set() # Set of clear tiles
        robot_locations = {} # Map robot name to location tile name
        robot_colors = {} # Map robot name to color name
        robots = set() # Collect robot names

        # Extract relevant state information in one pass
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'painted':
                current_painted[parts[1]] = parts[2]
            elif parts[0] == 'clear':
                current_clear.add(parts[1])
            elif parts[0] == 'robot-at':
                robot, loc = parts[1], parts[2]
                robot_locations[robot] = loc
                robots.add(robot)
            elif parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
                robots.add(robot)
            # Assuming robots always have a color, not free-color

        # 2. Check which goal paintings are not met by a correctly painted tile
        # and are on a clear tile (meaning they can be painted).
        for tile, required_color in self.goal_paintings.items():
            # Check if the tile is already painted with the correct color
            is_correctly_painted = (tile in current_painted and current_painted[tile] == required_color)

            # If not correctly painted AND it's clear (meaning it can be painted)
            # Note: If it's not clear and not correctly painted, it's an obstacle
            # painted with the wrong color, which we assume is not fixable and
            # doesn't contribute to the heuristic for solvable problems.
            if not is_correctly_painted and tile in current_clear:
                 unpainted_goal_tiles.append((tile, required_color))

        # 3. If no unpainted goal tiles (that are clear and need painting), heuristic is 0
        if not unpainted_goal_tiles:
            return 0

        # 5. Compute total heuristic
        total_heuristic = 0

        for tile_name, required_color in unpainted_goal_tiles:
            # 5a. Find adjacent tiles
            adjacent_tiles = self.adj_map.get(tile_name, set())
            if not adjacent_tiles:
                 # This goal tile has no adjacent tiles in the map - likely an issue with problem definition
                 # or a tile not in the grid. Treat as unreachable.
                 return math.inf # Unsolvable state

            # 5b. Get coordinates of adjacent tiles
            adj_coords = {self.tile_coords[adj_t] for adj_t in adjacent_tiles if adj_t in self.tile_coords}
            if not adj_coords:
                 # Adjacent tiles exist by name, but not in our coordinate map? Problematic.
                 return math.inf # Unsolvable state

            # 5c. Initialize min cost for this tile
            min_cost_for_tile = math.inf

            # 5d. For each robot
            for robot in robots:
                robot_loc = robot_locations.get(robot)
                robot_color = robot_colors.get(robot)

                # Robot must exist and have a location and color in the current state
                if robot_loc is None or robot_color is None:
                    continue # Skip robots not fully defined in state (shouldn't happen in valid states)

                # i. Calculate color change cost
                color_change_cost = 1 if robot_color != required_color else 0

                # ii. Calculate min movement cost to adjacent tile
                min_move_cost = math.inf
                if robot_loc in self.tile_coords:
                    robot_row, robot_col = self.tile_coords[robot_loc]
                    for ar, ac in adj_coords:
                        move_cost = abs(robot_row - ar) + abs(robot_col - ac)
                        min_move_cost = min(min_move_cost, move_cost)

                # If min_move_cost is still inf, robot cannot reach any adjacent tile (e.g., robot_loc not in tile_coords or no valid adj_coords)
                if min_move_cost == math.inf:
                    continue # This robot cannot paint this tile from its current location

                # iv. Total estimated cost for this robot on this tile
                total_cost_for_robot = color_change_cost + min_move_cost + 1 # +1 for paint action

                # v. Update min cost for this tile
                min_cost_for_tile = min(min_cost_for_tile, total_cost_for_robot)

            # If no robot can paint this tile (min_cost_for_tile is still inf)
            if min_cost_for_tile == math.inf:
                 # This state is likely unsolvable because a required goal tile
                 # cannot be painted by any robot.
                 return math.inf

            # f. Add min cost for this tile to total
            total_heuristic += min_cost_for_tile

        # 6. Return total heuristic
        return total_heuristic
