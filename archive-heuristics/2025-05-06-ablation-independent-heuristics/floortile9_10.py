import re
from heuristics.heuristic_base import Heuristic
# The Task class is assumed to be available in the environment where this heuristic is used.
# from task import Task

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing
    the estimated costs for each unsatisfied goal tile. For each tile
    that needs to be painted with a specific color, it calculates the
    minimum cost for any robot to reach an adjacent tile, acquire the
    correct color, and perform the paint action. The total heuristic
    value is the sum of these minimum costs over all unsatisfied goal tiles.
    It returns infinity if a tile is painted with the wrong color or if
    a required color is not available.

    Assumptions:
    - Tile names follow the format 'tile_R_C' where R and C are integers.
    - The grid is connected (any tile can be reached from any other tile).
    - All colors required by the goal are available colors.
    - Robots always hold a color (based on the provided actions and init examples).
    - The cost of each action (move, change_color, paint) is 1.
    - The heuristic ignores the 'clear' precondition for painting, assuming
      the search handles keeping goal tiles clear or moving robots off them.

    Heuristic Initialization:
    - Parses static facts to build the grid adjacency graph and store tile coordinates.
    - Parses static facts to identify available colors.
    - Parses goal facts to identify which tiles need to be painted with which colors.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse the current state to determine the location and color of each robot,
       and the current painted status of each tile.
    2. Identify unsatisfied goal tiles: tiles that are required to be painted
       with a specific color according to the goal, but are not currently painted
       with that color.
    3. Check for dead ends: If any tile is painted with a color different from
       the required goal color, the state is unreachable, and the heuristic
       returns infinity.
    4. If there are no unsatisfied goal tiles, the state is a goal state,
       and the heuristic returns 0.
    5. Initialize the total heuristic value `h` to 0.
    6. For each unsatisfied goal tile `T` that needs color `C`:
        a. Find all tiles `Y` that are adjacent to `T` based on the precomputed
           grid graph.
        b. Initialize the minimum cost to paint tile `T` (`min_cost_for_tile`)
           to infinity.
        c. For each robot `R`:
            i. Get the robot's current location `R_loc` and color `R_color`.
            ii. Calculate the minimum Manhattan distance from `R_loc` to any
                adjacent tile `Y` of `T`. This is the estimated number of move
                actions required.
            iii. Calculate the color change cost: 0 if `R_color` is already `C`,
                 1 otherwise (assuming `C` is available, checked during init).
            iv. The estimated cost for robot `R` to paint tile `T` is the minimum
                distance (moves) + color change cost + 1 (for the paint action itself).
            v. Update `min_cost_for_tile` with the minimum cost found among all robots.
        d. Add `min_cost_for_tile` to the total heuristic value `h`.
    7. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.available_colors = set()
        self.goal_painting = {}  # {tile: color}
        self.adj = {}  # {tile: [adj_tile, ...]}
        self.tile_coords = {}  # {tile: (r, c)}
        self.tiles = set()

        # Preprocess static facts
        for fact_str in task.static:
            parts = self._parse_fact(fact_str)
            if not parts:
                continue # Skip malformed facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'available-color':
                self.available_colors.add(args[0])
            elif predicate in ['up', 'down', 'left', 'right']:
                t1, t2 = args
                self._add_adjacency(t1, t2)
                self.tiles.add(t1)
                self.tiles.add(t2)

        # Parse coordinates for all identified tiles
        for tile_name in self.tiles:
            self.tile_coords[tile_name] = self._parse_tile_name(tile_name)

        # Preprocess goal facts
        for fact_str in self.goals:
            parts = self._parse_fact(fact_str)
            if not parts:
                continue # Skip malformed facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'painted':
                tile, color = args
                self.goal_painting[tile] = color
                # Note: We don't check color availability here in __init__
                # because the heuristic calculation in __call__ handles
                # the case where a required color might not be available
                # in the current state (though it should be in static).
                # The check for required_color not in self.available_colors
                # is done in __call__ and returns inf if needed.


    def _parse_fact(self, fact_str):
        """Parses a PDDL fact string into a list of strings."""
        # Remove outer parentheses and split by space
        # Handles cases like '(predicate arg1 arg2)'
        fact_str = fact_str.strip()
        if not fact_str.startswith('(') or not fact_str.endswith(')'):
            return None # Not a valid fact string format

        content = fact_str[1:-1].strip()
        if not content:
            return None # Empty fact

        # Simple split by space might be enough for this domain
        return content.split()

    def _add_adjacency(self, t1, t2):
        """Adds adjacency relationship between two tiles."""
        # Assuming t1 is adjacent to t2. The specific direction
        # doesn't matter for simple adjacency list.
        if t1 not in self.adj:
            self.adj[t1] = []
        if t2 not in self.adj:
            self.adj[t2] = []
        if t2 not in self.adj[t1]:
            self.adj[t1].append(t2)
        if t1 not in self.adj[t2]:
            self.adj[t2].append(t1)


    def _parse_tile_name(self, tile_name):
        """Parses 'tile_R_C' string into (R, C) tuple."""
        match = re.match(r'tile_(\d+)_(\d+)', tile_name)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        # Handle potential errors - maybe return None or raise error
        # For this problem, assume valid tile names
        return None # Should not happen with valid PDDL

    def _manhattan_distance(self, tile1_name, tile2_name):
        """Calculates Manhattan distance between two tiles."""
        coord1 = self.tile_coords.get(tile1_name)
        coord2 = self.tile_coords.get(tile2_name)
        if coord1 is None or coord2 is None:
            # Should not happen if tiles were correctly parsed
            return float('inf')
        return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

    def __call__(self, node):
        state = node.state

        robot_loc = {}  # {robot: tile}
        robot_color = {}  # {robot: color}
        current_painting = {} # {tile: color}
        # clear_tiles = set() # Not strictly needed for this heuristic logic

        # Parse current state facts
        for fact_str in state:
            parts = self._parse_fact(fact_str)
            if not parts:
                continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'robot-at':
                robot, tile = args
                robot_loc[robot] = tile
            elif predicate == 'robot-has':
                robot, color = args
                robot_color[robot] = color
            elif predicate == 'painted':
                tile, color = args
                current_painting[tile] = color
            # elif predicate == 'clear':
            #     clear_tiles.add(args[0])

        unsatisfied_goals = [] # [(tile, required_color)]

        # Identify unsatisfied goals and check for dead ends
        for tile, required_color in self.goal_painting.items():
            if tile in current_painting:
                if current_painting[tile] != required_color:
                    # Tile is painted with the wrong color - dead end
                    return float('inf')
            else:
                # Tile needs to be painted
                unsatisfied_goals.append((tile, required_color))

        # If all goals are satisfied
        if not unsatisfied_goals:
            return 0

        h = 0

        # Calculate cost for each unsatisfied goal tile
        for tile, required_color in unsatisfied_goals:
            # Check if the required color is available at all
            if required_color not in self.available_colors:
                 # This goal is unreachable
                 return float('inf')

            min_cost_for_tile = float('inf')

            # Find adjacent tiles
            adjacent_tiles = self.adj.get(tile, [])
            if not adjacent_tiles:
                 # Cannot paint if no adjacent tile exists (shouldn't happen in grid)
                 return float('inf')

            # Find the best robot to paint this tile
            for robot in robot_loc:
                R_loc = robot_loc[robot]
                R_color = robot_color.get(robot) # Get robot's current color

                # Calculate min distance from robot to any adjacent tile
                min_dist_R_to_Adj_T = float('inf')
                for adj_tile in adjacent_tiles:
                    dist = self._manhattan_distance(R_loc, adj_tile)
                    min_dist_R_to_Adj_T = min(min_dist_R_to_Adj_T, dist)

                # Calculate color change cost and check if robot can paint
                color_cost = 0
                can_paint = True

                if R_color is None:
                    # Robot has no color fact, cannot paint
                    can_paint = False
                elif R_color != required_color:
                    # Robot has wrong color, needs to change
                    color_cost = 1

                # Total cost for this robot to paint this tile
                # Moves + Color Change + Paint Action
                cost_R_to_paint_T = float('inf') # Default to inf if cannot paint or reach
                if can_paint and min_dist_R_to_Adj_T != float('inf'):
                     cost_R_to_paint_T = min_dist_R_to_Adj_T + color_cost + 1

                # Update minimum cost for this tile
                min_cost_for_tile = min(min_cost_for_tile, cost_R_to_paint_T)


            # If min_cost_for_tile is still inf, it means no robot could reach
            # an adjacent tile or no robot can paint this color.
            if min_cost_for_tile == float('inf'):
                 # This implies the goal is unreachable from this state
                 return float('inf')

            h += min_cost_for_tile

        return h
