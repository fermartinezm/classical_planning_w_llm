import re

def parse_tile_name(tile_name):
    """Parses 'tile_r_c' into (r, c) tuple."""
    match = re.match(r'tile_(\d+)_(\d+)', tile_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None # Should not happen based on problem description

def manhattan_distance(coords1, coords2):
    """Calculates Manhattan distance between two (r, c) tuples."""
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

class floortileHeuristic:
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing up
    the estimated costs for each unsatisfied goal condition (painting a tile
    with a specific color). The estimated cost for painting a single tile
    includes:
    1. The cost of the paint action (always 1).
    2. The cost for a robot to acquire the required color (estimated as 1
       if no robot currently holds the color, 0 otherwise, summed over
       all required colors not held).
    3. The cost for a robot to move to a tile from which it can paint the
       target tile (estimated as the minimum Manhattan distance from any
       robot's current location to any valid painting position for the
       target tile, summed over all tiles that need painting).

    Assumptions:
    - Tile names follow the format 'tile_r_c' where r and c are integers
      representing row and column.
    - The grid is connected as defined by the up, down, left, right predicates.
    - Solvable problems do not require unpainting or repainting tiles that
      are already painted with the wrong color (i.e., any tile required
      to be painted in the goal is either clear or painted with the correct
      color in any reachable state). The heuristic assumes unsatisfied
      painted goals correspond to tiles that are currently clear.
    - Robots always hold a color (no 'free-color' state is considered).
    - All tiles mentioned in initial state, goal, or static facts follow the 'tile_r_c' format.

    Heuristic Initialization:
    The constructor pre-processes the static facts and other task information
    to build:
    - A mapping from tile names to their (row, column) coordinates by parsing
      the 'tile_r_c' format for all tiles present in the task.
    - A mapping from (row, column) coordinates back to tile names.
    - A set of available colors from 'available-color' facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify all goal conditions of the form '(painted T C)'.
    2. Identify all facts in the current state of the form '(painted T C)'.
    3. Determine the set of 'unsatisfied goals': These are the '(painted T C)'
       goal facts that are not present in the current state.
    4. If the set of unsatisfied goals is empty, the state is a goal state,
       and the heuristic value is 0.
    5. Initialize the heuristic value `h` to 0.
    6. Add the number of unsatisfied goals to `h`. This accounts for the
       minimum number of paint actions required.
    7. Identify the set of colors required by the unsatisfied goals.
    8. Identify the set of colors currently held by robots in the state.
    9. Add the number of required colors that are not currently held by any
       robot to `h`. This estimates the minimum number of color change actions
       needed across the robot fleet.
    10. For each unsatisfied goal `(T, C)`:
        a. Get the coordinates `(r, c)` of the tile `T`.
        b. Determine the set of required robot painting positions (coordinates)
           adjacent to `T`: `{(r+1, c), (r-1, c), (r, c+1), (r, c-1)}`,
           filtering out positions that do not correspond to existing tiles
           in the grid using the `coords_tile` map.
        c. Find the current location (coordinates) of each robot from the
           state facts '(robot-at R L)'.
        d. Calculate the minimum Manhattan distance from *any* robot's current
           location to *any* of the required robot painting positions for tile `T`.
        e. Add this minimum distance to `h`. This estimates the movement cost
           associated with painting tile `T`.
    11. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic with static task information.

        @param task: The planning task object.
        """
        self.task = task
        self.tile_coords = {}
        self.coords_tile = {}
        self.available_colors = set()
        self._parse_static_info()

    def _parse_static_info(self):
        """Parses static facts and other task info to build internal data structures."""
        all_tiles = set()
        # Extract all tile names from all facts in the task
        for fact_str in self.task.facts:
             parts = fact_str.strip('()').split()
             for part in parts:
                 if part.startswith('tile_'):
                     all_tiles.add(part)

        # Build tile_coords and coords_tile maps
        for tile_name in all_tiles:
            coords = parse_tile_name(tile_name)
            if coords is not None:
                self.tile_coords[tile_name] = coords
                self.coords_tile[coords] = tile_name

        # Extract available colors
        for fact_str in self.task.static:
            parts = fact_str.strip('()').split()
            if len(parts) == 2 and parts[0] == 'available-color':
                self.available_colors.add(parts[1])

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @return: The estimated cost to reach the goal.
        """
        # 1. Identify goal painted tiles
        goal_painted_tiles = set()
        for goal_fact in self.task.goals:
            parts = goal_fact.strip('()').split()
            if len(parts) == 3 and parts[0] == 'painted':
                goal_painted_tiles.add((parts[1], parts[2])) # (tile, color)

        # 2. Identify current painted tiles
        current_painted_tiles = set()
        for fact in state:
            parts = fact.strip('()').split()
            if len(parts) == 3 and parts[0] == 'painted':
                current_painted_tiles.add((parts[1], parts[2])) # (tile, color)

        # 3. Determine unsatisfied goals
        unsatisfied_goals = goal_painted_tiles - current_painted_tiles

        # 4. If goal reached, return 0
        if not unsatisfied_goals:
            return 0

        # 5. Initialize heuristic
        h = 0

        # 6. Add paint actions cost
        h += len(unsatisfied_goals)

        # 7. Identify required colors
        required_colors = {color for (tile, color) in unsatisfied_goals}

        # 8. Identify held colors and robot locations
        held_colors = set()
        robot_locations_coords = {} # robot_name -> (r, c)
        for fact in state:
            parts = fact.strip('()').split()
            if len(parts) == 3:
                if parts[0] == 'robot-at':
                    robot_name = parts[1]
                    tile_name = parts[2]
                    if tile_name in self.tile_coords:
                         robot_locations_coords[robot_name] = self.tile_coords[tile_name]
                    # else: robot is at an unknown location? Should not happen in valid states.
                elif parts[0] == 'robot-has':
                    held_colors.add(parts[2])

        # 9. Add color change cost
        # Estimate one color change for each required color not currently held by any robot
        h += len(required_colors - held_colors)

        # 10. Add movement cost
        movement_cost = 0
        for (tile_to_paint, required_color) in unsatisfied_goals:
            if tile_to_paint not in self.tile_coords:
                 # Tile to paint is not a known tile. Problem likely unsolvable.
                 return 1000000 # Large finite number

            (r, c) = self.tile_coords[tile_to_paint]

            # Determine required robot painting positions (coordinates)
            # Robot needs to be at (r_req, c_req) to paint (r, c)
            # paint_up: robot at (r+1, c) paints (r, c)
            # paint_down: robot at (r-1, c) paints (r, c)
            # paint_left: robot at (r, c+1) paints (r, c)
            # paint_right: robot at (r, c-1) paints (r, c)
            potential_req_coords = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
            required_robot_loc_coords = {
                coords for coords in potential_req_coords if coords in self.coords_tile
            }

            if not required_robot_loc_coords:
                 # This tile cannot be painted from any adjacent tile that exists in the grid.
                 # Problem likely unsolvable.
                 return 1000000 # Large finite number

            min_dist_to_paint_pos_for_this_tile = float('inf')

            # Find the minimum distance from any robot to any required painting position
            if not robot_locations_coords:
                 # No robots found? Problem likely unsolvable.
                 return 1000000 # Large finite number

            for robot_coords in robot_locations_coords.values():
                 for req_coords in required_robot_loc_coords:
                     dist = manhattan_distance(robot_coords, req_coords)
                     min_dist_to_paint_pos_for_this_tile = min(min_dist_to_paint_pos_for_this_tile, dist)

            # If min_dist_to_paint_pos_for_this_tile is still inf, it means no robot can reach any required position.
            # This could happen if the grid is disconnected or robots are stuck in a way that prevents reaching any paint position.
            if min_dist_to_paint_pos_for_this_tile == float('inf'):
                 # This tile cannot be painted by any robot. Problem likely unsolvable.
                 return 1000000 # Large finite number

            movement_cost += min_dist_to_paint_pos_for_this_tile

        h += movement_cost

        return h
