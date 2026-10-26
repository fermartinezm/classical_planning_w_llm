from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this path is correct

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at-robby rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the total cost to paint all required tiles.
    It sums, for each unpainted goal tile, the minimum estimated cost
    for any robot to reach an adjacent tile, change color if needed, and paint the tile.
    It also detects unsolvable states where a tile is painted with the wrong color.

    # Assumptions
    - The grid structure is defined by `up`, `down`, `left`, `right` predicates in static facts.
    - Tiles needing painting in the goal are initially either `clear` or already painted with the correct color. If a tile is painted with the wrong color according to the goal, the problem is considered unsolvable.
    - Robots are always at a tile and have a color in any reachable state.
    - The cost of changing color is 1.
    - The cost of moving between adjacent tiles is 1 (Manhattan distance on the grid).
    - The cost of painting a tile is 1.
    - The heuristic calculates the minimum cost *per unpainted tile* across all robots and sums these minimums. This is an overestimate (as it doesn't account for shared movement or color changes for nearby tiles/robots) but aims to guide towards making progress on individual goal tiles.

    # Heuristic Initialization
    - Extracts goal painting requirements (`(painted tile color)` facts from the goal).
    - Parses static facts (`up`, `down`, `left`, `right`) to build an adjacency map representing the grid structure, which is used for Manhattan distance calculations.
    - Identifies all robots present in the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal facts of the form `(painted tile color)`. Store these as the target paintings.
    2. Iterate through the target paintings. For each goal tile `T` needing color `C`:
       - Check if the fact `(painted T C)` is already true in the current state. If yes, this goal is satisfied for this tile.
       - If not satisfied, check if tile `T` is painted with any *other* color `C'` in the current state. If this is the case, the problem is likely unsolvable in this domain (as there's no action to unpaint or repaint a tile that isn't clear). If an unsolvable state is detected, return a very high heuristic value (e.g., 1,000,000) to prune this branch.
       - If the tile is neither painted correctly nor wrongly painted, it must be `clear` (assuming valid state representation). Add this tile `T` and its needed color `C` to a list of "unpainted goal tiles".
    3. If the list of unpainted goal tiles is empty after checking all goal paintings, the heuristic value is 0 (the goal state has been reached).
    4. If there are unpainted goal tiles, get the current location and color for each robot from the state facts.
    5. Initialize the total heuristic cost to 0.
    6. For each unpainted goal tile `T` needing color `C`:
       a. Calculate the minimum estimated cost for *any* robot to successfully paint this specific tile `T`.
          - For each robot `R` with its current location `R_loc` and color `R_color`:
            - Estimate the cost to change color: 1 if `R_color` is not the needed color `C`, otherwise 0.
            - Estimate the cost to move to a tile adjacent to `T`: Calculate the minimum Manhattan distance from the robot's current location `R_loc` to any tile that is adjacent to `T` according to the grid structure (using the pre-calculated adjacency map and parsing tile names into row/column coordinates).
            - The cost to paint the tile is 1.
            - The total estimated cost for robot `R` to paint tile `T` is the sum of the color change cost, the movement cost, and the paint cost.
          - Find the minimum of these total estimated costs over all available robots. This represents the minimum effort required by the "best" robot to paint tile `T`.
       b. Add this minimum cost calculated for tile `T` to the total heuristic cost.
       c. If the minimum cost for any unpainted tile is infinite (e.g., due to a disconnected grid preventing access), the problem is likely unsolvable. Return a high heuristic value (e.g., 1,000,000).
    7. Return the accumulated total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and robot information.
        """
        self.goals = task.goals

        # Store goal paintings: {tile: color}
        self.goal_paintings = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted' and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color

        # Build adjacency map from static facts: {tile: [adjacent_tiles]}
        # This map stores direct neighbors based on up/down/left/right relations.
        self.adj_map = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right'] and len(parts) == 3:
                tile1, tile2 = parts[1], parts[2]
                # Relations are symmetric in a grid
                self.adj_map.setdefault(tile1, []).append(tile2)
                self.adj_map.setdefault(tile2, []).append(tile1)

        # Identify robots from initial state facts
        self.robots = set()
        for fact in task.initial_state:
             parts = get_parts(fact)
             # Look for facts that involve a robot object
             if parts and parts[0] in ['robot-at', 'robot-has', 'free-color']:
                 if len(parts) > 1: # Ensure there's an object parameter
                     self.robots.add(parts[1])
        self.robots = list(self.robots) # Convert to list for consistent ordering

    def parse_tile_name(self, tile_name):
        """Parses a tile name like 'tile_r_c' into a (row, col) tuple."""
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            try:
                row = int(parts[1])
                col = int(parts[2])
                return (row, col)
            except ValueError:
                return None # Not a valid tile name format
        return None # Not a valid tile name format

    def dist_manhattan(self, coord1, coord2):
        """Calculates Manhattan distance between two (row, col) coordinates."""
        r1, c1 = coord1
        r2, c2 = coord2
        return abs(r1 - r2) + abs(c1 - c2)

    def min_dist_to_adjacent(self, loc1, loc2):
        """
        Calculates the minimum Manhattan distance from loc1 to any tile
        adjacent to loc2. Returns float('inf') if loc2 has no adjacent tiles
        or if loc1 is not a valid tile name.
        """
        coord1 = self.parse_tile_name(loc1)
        if coord1 is None:
             # Robot location is not a valid tile name - should not happen in valid states
             return float('inf')

        adjacent_locs2 = self.adj_map.get(loc2, [])
        if not adjacent_locs2:
            # loc2 has no adjacent tiles (e.g., isolated tile)? Should not happen for goal tiles in a grid.
            return float('inf')

        min_dist = float('inf')
        for adj_loc2 in adjacent_locs2:
            coord2 = self.parse_tile_name(adj_loc2)
            if coord2 is not None:
                min_dist = min(min_dist, self.dist_manhattan(coord1, coord2))

        return min_dist # Can still be inf if no adjacent tiles have valid names

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        unpainted_goals = {} # {tile: needed_color}

        # Check goal paintings against current state to find unpainted goals
        # and detect unsolvable states (wrongly painted tiles).
        for tile, needed_color in self.goal_paintings.items():
            is_painted_correctly = f'(painted {tile} {needed_color})' in state

            if not is_painted_correctly:
                # This tile needs to be painted with needed_color
                # Check if it's currently painted with the wrong color
                is_wrongly_painted = False
                for fact in state:
                    parts = get_parts(fact)
                    if parts[0] == 'painted' and len(parts) == 3 and parts[1] == tile and parts[2] != needed_color:
                        is_wrongly_painted = True
                        break

                if is_wrongly_painted:
                    # Tile is painted with the wrong color - problem is likely unsolvable
                    # Return a very high heuristic value to prune this branch
                    return 1000000 # Indicate unsolvability

                # If not painted correctly and not wrongly painted, it must be clear (or an issue with state representation)
                # Assume it's clear and needs painting
                unpainted_goals[tile] = needed_color

        if not unpainted_goals:
            return 0 # Goal reached

        # Get robot states (location and color) from the current state
        robot_states = {} # {robot_name: {'loc': tile_name, 'color': color_name}}
        for robot in self.robots:
             loc = None
             color = None
             # Find robot's location and color in the current state
             for fact in state:
                 if match(fact, 'robot-at', robot, '*'):
                     loc = get_parts(fact)[2]
                 elif match(fact, 'robot-has', robot, '*'):
                     color = get_parts(fact)[2]
             # Store state. Assume loc and color are always present for active robots in valid states.
             robot_states[robot] = {'loc': loc, 'color': color}


        total_heuristic_cost = 0

        # Sum minimum cost for each unpainted goal tile across all robots
        for tile, needed_color in unpainted_goals.items():
            min_cost_for_tile = float('inf')

            for robot_name, robot_state in robot_states.items():
                robot_loc = robot_state['loc']
                robot_color = robot_state['color']

                # If robot state is incomplete (shouldn't happen in valid states) or robot cannot paint
                if robot_loc is None or robot_color is None:
                     continue # Cannot use this robot for this tile

                # Cost to change color
                color_cost = 0
                if robot_color != needed_color:
                    color_cost = 1 # Assume 1 action to change color

                # Cost to move to adjacent tile
                movement_cost = self.min_dist_to_adjacent(robot_loc, tile)

                # Total cost for this robot to paint this tile
                # movement_cost can be inf if no adjacent tiles found for the target tile
                if movement_cost == float('inf'):
                     cost = float('inf')
                else:
                     cost = color_cost + movement_cost + 1 # +1 for paint action

                min_cost_for_tile = min(min_cost_for_tile, cost)

            # If min_cost_for_tile is still inf, it means no robot can reach/paint this tile.
            # This might indicate unsolvability or a disconnected part of the grid.
            # Return a high number if any required tile is unreachable.
            if min_cost_for_tile == float('inf'):
                 return 1000000 # Indicate unsolvability

            total_heuristic_cost += min_cost_for_tile

        # The total heuristic cost is finite if all unpainted tiles are reachable.
        return total_heuristic_cost
