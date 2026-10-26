from fnmatch import fnmatch
# Assuming Heuristic base class is available in a 'heuristics' directory
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def get_coords(tile_name):
    """
    Parses a tile name like 'tile_row_col' into (row, col) integers.
    Returns None if the name format is unexpected.
    """
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            return None
    return None

def manhattan_distance(tile1, tile2, tile_name_to_coords):
    """
    Calculates the Manhattan distance between two tiles using their coordinates.
    Returns a large number if coordinates are not found for either tile.
    """
    coords1 = tile_name_to_coords.get(tile1)
    coords2 = tile_name_to_coords.get(tile2)

    # If coordinates are missing for a tile, it's likely not a grid tile
    # or there was a parsing issue. Treat as unreachable for distance purposes.
    if coords1 is None or coords2 is None:
        return 1000000 # Large number

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their required colors. It sums the estimated cost for each unpainted goal tile.
    The cost for a single tile includes:
    1. A base cost for painting (1 action).
    2. A cost if the tile is currently occupied by a robot (1 action to move off).
    3. The minimum cost among all robots to reach an adjacent tile and paint it,
       including changing color if necessary. Movement cost is estimated using
       Manhattan distance to the target tile minus 1 (for reaching an adjacent tile).

    # Assumptions
    - The grid structure is defined by 'up', 'down', 'left', 'right' predicates,
      and tiles are named 'tile_row_col'.
    - Robots can change color if the target color is available.
    - Movement cost is approximated by Manhattan distance, ignoring dynamic obstacles
      (other painted tiles, other robots' positions) except for the tile being painted
      itself if a robot is on it.
    - States where a goal tile is painted with the wrong color are considered unsolvable
      (assigned a very high heuristic value).

    # Heuristic Initialization
    - Extracts goal conditions to identify which tiles need to be painted and with which color.
    - Parses tile object names from the task definition to build a mapping from tile
      name to grid coordinates (row, col). This is used for Manhattan distance calculation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal tiles and their required colors from the precomputed goal set.
    2. For each goal tile that is not yet painted correctly:
        a. Check if the tile is painted with an incorrect color. If yes, the state is likely unsolvable; return a very large heuristic value.
        b. Initialize the cost for this specific goal tile. It includes a base cost of 1 for the paint action.
        c. Check if any robot is currently located *on* this goal tile. If yes, add an additional cost of 1 (for the robot to move off the tile).
        d. Calculate the minimum cost for *any* robot to paint this tile:
            - Iterate through all robots found in the state.
            - For each robot, determine its current location and the color it holds.
            - Calculate the estimated cost for this robot to paint the tile:
                - Start with 0.
                - If the robot's current color is different from the required color for the tile, add 1 (for the `change_color` action).
                - Estimate movement cost: Calculate the Manhattan distance from the robot's location to the goal tile's location using the precomputed coordinates. The robot needs to reach an *adjacent* tile to paint. The minimum moves to reach an adjacent tile is `max(0, distance - 1)`. Add this movement cost.
            - Keep track of the minimum cost found among all robots.
        e. Add the minimum robot cost (calculated in step 2d) to the tile's cost (calculated in steps 2b and 2c).
    3. Sum the costs calculated for all unpainted goal tiles. This sum is the total heuristic value for the state.
    4. If any goal tile was painted with the wrong color, the total heuristic is the large value returned in step 2a.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and tile coordinates.
        """
        super().__init__(task) # Call the base class constructor

        # Store goal locations and colors for each tile.
        self.goal_tiles = {}
        # task.goals is a frozenset of goal fact strings
        for goal_fact_str in self.goals:
            parts = get_parts(goal_fact_str)
            if len(parts) == 3 and parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_tiles[tile] = color

        # Map tile names to their (row, col) coordinates.
        self.tile_name_to_coords = {}
        # task.objects is a list of object name strings (e.g., ['tile_0_1', 'robot1', ...])
        for obj_name in self.objects:
             coords = get_coords(obj_name)
             if coords is not None:
                 self.tile_name_to_coords[obj_name] = coords

        # Define a large cost for unsolvable states
        self.UNSOLVABLE_COST = 1000000


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state as a frozenset of strings

        # Check if it's a goal state (optional, but good practice)
        # The loop below will compute 0 if all goals are met, but an explicit check is fast.
        if self.goals.issubset(state):
             return 0

        # Extract robot locations and colors from the current state
        robot_locations = {}
        robot_colors = {}
        robots = set() # Collect robot names
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3:
                if parts[0] == 'robot-at':
                    robot, tile = parts[1], parts[2]
                    robot_locations[robot] = tile
                    robots.add(robot)
                elif parts[0] == 'robot-has':
                    robot, color = parts[1], parts[2]
                    robot_colors[robot] = color
                    robots.add(robot)

        # Extract painted tiles from the current state
        painted_tiles = {} # {tile: color}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == 'painted':
                 painted_tiles[parts[1]] = parts[2]

        # Identify tiles currently occupied by a robot
        occupied_tiles = set(robot_locations.values())

        total_heuristic = 0

        # Iterate through all goal tiles
        for goal_tile, required_color in self.goal_tiles.items():
            # Check if the goal tile is already painted correctly
            if painted_tiles.get(goal_tile) == required_color:
                continue # This goal is satisfied

            # Check if the goal tile is painted with the wrong color
            if goal_tile in painted_tiles and painted_tiles[goal_tile] != required_color:
                return self.UNSOLVABLE_COST # State is unsolvable

            # The tile needs to be painted with required_color

            # Base cost for painting this tile
            tile_cost = 1 # Cost of the paint action

            # Check if a robot is currently on the tile that needs painting
            if goal_tile in occupied_tiles:
                 # Add cost for the robot to move off the tile
                 tile_cost += 1

            # Calculate the minimum cost for any robot to paint this tile
            min_robot_painting_cost = self.UNSOLVABLE_COST # Initialize with a large value

            # If there are no robots, this tile cannot be painted
            if not robots:
                 return self.UNSOLVABLE_COST

            for robot in robots:
                robot_loc = robot_locations.get(robot)
                robot_color = robot_colors.get(robot)

                # Should always find robot_loc and robot_color if robot is in the 'robots' set
                # derived from state facts, but defensive check doesn't hurt.
                if robot_loc is None or robot_color is None:
                    continue

                current_robot_cost = 0

                # Cost for changing color if needed
                if robot_color != required_color:
                    current_robot_cost += 1 # Cost of change_color action

                # Cost for movement
                # Robot needs to reach a tile adjacent to goal_tile.
                # Manhattan distance from robot_loc to goal_tile is an estimate.
                # The minimum moves to reach an adjacent tile is max(0, dist - 1).
                dist = manhattan_distance(robot_loc, goal_tile, self.tile_name_to_coords)

                if dist >= self.UNSOLVABLE_COST: # Handle case where coordinates were not found
                     current_robot_cost = self.UNSOLVABLE_COST
                else:
                     movement_cost = max(0, dist - 1)
                     current_robot_cost += movement_cost

                min_robot_painting_cost = min(min_robot_painting_cost, current_robot_cost)

            # If min_robot_painting_cost is still the large initial value,
            # it means no robot could potentially paint this tile (e.g., no robots exist,
            # or coordinates were missing for all robots/the tile).
            if min_robot_painting_cost >= self.UNSOLVABLE_COST:
                 return self.UNSOLVABLE_COST

            tile_cost += min_robot_painting_cost
            total_heuristic += tile_cost

        return total_heuristic
