from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are currently clear. For each such tile, it calculates the minimum cost
    for any robot to reach an adjacent tile with the correct color and paint it,
    and sums these minimum costs.

    # Assumptions
    - Tiles needing to be painted according to the goal are initially clear.
    - Tiles painted with the wrong color cannot be repainted (as paint actions require the tile to be clear).
    - The tile names follow a 'tile_row_col' format allowing Manhattan distance calculation.
    - All colors required for goal paintings are available.
    - Robots always hold a color if the 'robot-has' predicate exists for them.

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need which colors.
    - Identifies all tile, robot, and color objects present in the problem instance
      by parsing initial state, goals, and static facts.
    - Stores the mapping from goal tiles to their required colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal tiles that are currently in a 'clear' state. These are the tiles that still need to be painted.
    2. Identify the current location and held color for each robot from the current state.
    3. Initialize the total heuristic value to 0.
    4. For each tile that needs to be painted (is clear in the current state and is a goal tile):
        a. Determine the color required for this tile according to the goal.
        b. Find the set of tiles adjacent to this goal tile based on the grid structure implied by tile names.
        c. Calculate the minimum cost for *any* robot to paint this specific tile:
            i. For each robot:
                - Calculate the cost to acquire the needed color (1 if the robot has a different color than required, 0 otherwise).
                - Calculate the minimum Manhattan distance from the robot's current location to any of the adjacent tiles of the goal tile.
                - The cost for this robot to paint this tile is the color acquisition cost + the minimum move distance + 1 (for the paint action itself).
            ii. The minimum cost for the tile is the minimum of these costs over all robots.
        d. Add this minimum cost found for the tile (among all robots) to the total heuristic value.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and object lists.
        """
        self.goals = task.goals

        # Extract all objects by type from initial state, goals, and static facts
        all_objects = set()
        # Iterate through all facts provided in the task definition
        # This includes initial state, goals, and static facts
        all_facts_strings = set(task.initial_state) | set(task.goals) | set(task.static)

        for fact_string in all_facts_strings:
            parts = self._get_parts(fact_string)
            if parts and len(parts) > 1: # Ensure fact has predicate and at least one argument
                 all_objects.update(parts[1:]) # Add all arguments as potential objects

        # Filter objects based on naming conventions
        self.all_tiles = {obj for obj in all_objects if obj.startswith('tile_')}
        self.all_robots = {obj for obj in all_objects if obj.startswith('robot')}
        # Assume anything else starting with a letter is a color (simple heuristic)
        self.all_colors = {obj for obj in all_objects if obj[0].isalpha() and not obj.startswith('tile_') and not obj.startswith('robot')}


        # Store goal paintings: {tile: color}
        self.goal_paintings = {}
        for goal_fact in self.goals:
            if self._match(goal_fact, "painted", "*", "*"):
                _, tile, needed_color = self._get_parts(goal_fact)
                self.goal_paintings[tile] = needed_color

    def _get_parts(self, fact):
        """Extract the components of a PDDL fact string."""
        # Handle potential empty strings or malformed facts gracefully
        if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
             return []
        return fact[1:-1].split()

    def _match(self, fact, *args):
        """Check if a PDDL fact string matches a given pattern."""
        parts = self._get_parts(fact)
        if len(parts) != len(args):
            return False
        return all(fnmatch(part, arg) for part, arg in zip(parts, args))

    def _get_tile_coords(self, tile_name):
        """Parses tile name 'tile_row_col' into (row, col) tuple."""
        try:
            parts = tile_name.split('_')
            if len(parts) == 3 and parts[0] == 'tile':
                row = int(parts[1])
                col = int(parts[2])
                return (row, col)
        except (ValueError, IndexError):
            pass # Handle cases where tile name format is unexpected
        return None # Indicate parsing failure

    def _manhattan_distance(self, tile1_name, tile2_name):
        """Calculates Manhattan distance between two tiles."""
        coords1 = self._get_tile_coords(tile1_name)
        coords2 = self._get_tile_coords(tile2_name)
        if coords1 is None or coords2 is None:
            # Cannot calculate distance for invalid tile names
            return math.inf
        return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

    def _get_adjacent_tiles(self, tile_name):
        """Finds valid adjacent tiles based on grid naming and existing tiles."""
        coords = self._get_tile_coords(tile_name)
        if coords is None:
            return [] # Cannot find adjacent tiles for invalid name

        r, c = coords
        potential_neighbors_coords = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        adjacent_tiles = []
        for nr, nc in potential_neighbors_coords:
            neighbor_name = f"tile_{nr}_{nc}"
            # Check if the potential neighbor tile actually exists in the problem
            if neighbor_name in self.all_tiles:
                adjacent_tiles.append(neighbor_name)
        return adjacent_tiles

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_heuristic = 0

        # Identify unpainted goal tiles that are currently clear
        unpainted_goal_tiles = {} # {tile: needed_color}
        for tile, needed_color in self.goal_paintings.items():
             goal_fact = f"(painted {tile} {needed_color})"
             # Check if the goal painting is NOT satisfied AND the tile is clear
             if goal_fact not in state and f"(clear {tile})" in state:
                 unpainted_goal_tiles[tile] = needed_color

        # If no tiles need painting from a clear state, the painting goals are met
        if not unpainted_goal_tiles:
            return 0

        # Identify robot locations and colors
        robot_locations = {} # {robot: tile}
        robot_colors = {} # {robot: color}
        for fact in state:
            if self._match(fact, "robot-at", "*", "*"):
                _, robot, loc = self._get_parts(fact)
                robot_locations[robot] = loc
            elif self._match(fact, "robot-has", "*", "*"):
                _, robot, color = self._get_parts(fact)
                robot_colors[robot] = color

        # Calculate heuristic contribution for each unpainted goal tile
        for tile, needed_color in unpainted_goal_tiles.items():
            min_cost_for_tile = math.inf

            adjacent_tiles = self._get_adjacent_tiles(tile)
            if not adjacent_tiles:
                 # This tile cannot be painted as it has no adjacent tiles in the grid
                 # This indicates a malformed problem instance or disconnected grid.
                 # Returning infinity for this tile's contribution.
                 total_heuristic += math.inf
                 continue # Move to the next unpainted tile

            # Consider each robot as a potential candidate to paint this tile
            for robot in self.all_robots:
                if robot not in robot_locations:
                    # Robot is not at any tile? Malformed state. Cannot use this robot.
                    continue

                robot_loc = robot_locations[robot]
                robot_color = robot_colors.get(robot) # Use .get() in case robot has no color initially

                # Cost to get the needed color
                # Assuming robot always has *some* color if robot-has predicate exists
                # If robot_color is None, assume it needs 1 action to get the first color if needed_color is not None
                color_cost = 0
                if needed_color is not None: # Should always be true for goal paintings
                    if robot_color != needed_color:
                         color_cost = 1 # change_color action

                # Minimum moves for this robot to reach any tile adjacent to the target tile
                min_moves_to_adjacent = math.inf
                for adj_tile in adjacent_tiles:
                    dist = self._manhattan_distance(robot_loc, adj_tile)
                    if dist is not math.inf: # Only consider reachable adjacent tiles
                        min_moves_to_adjacent = min(min_moves_to_adjacent, dist)

                # If no adjacent tile is reachable by this robot, this robot cannot paint this tile
                if min_moves_to_adjacent is math.inf:
                    continue

                # Total estimated cost for this robot to paint this specific tile:
                # color_cost (change color) + min_moves_to_adjacent (move) + 1 (paint action)
                cost_for_robot = color_cost + min_moves_to_adjacent + 1

                min_cost_for_tile = min(min_cost_for_tile, cost_for_robot)

            # Add the minimum cost found among all robots for this tile's painting task
            total_heuristic += min_cost_for_tile

        return total_heuristic
