import re
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Splits a PDDL fact string into its predicate and arguments."""
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

# Helper function to parse tile names like 'tile_row_col'
def parse_tile_name(tile_name):
    """Parses a tile name string 'tile_row_col' into (row, col) integer coordinates."""
    match = re.match(r'tile_(\d+)_(\d+)', tile_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    # Assuming valid tile names based on problem description format.
    # Return None or raise error if format is unexpected.
    return None

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        up the estimated minimum cost for each unpainted goal tile. For each
        unpainted goal tile, it calculates the minimum cost among all robots
        to acquire the correct color and reach a tile adjacent to the target
        tile, plus the cost of the paint action. This provides an estimate
        of the effort required for each individual painting task, summed over
        all tasks.

    Assumptions:
        - All action costs are unit (cost 1).
        - The tile names follow the pattern 'tile_row_col' allowing coordinate
          extraction.
        - The grid structure (adjacency) is defined by the static 'up', 'down',
          'left', 'right' predicates.
        - Movement cost between adjacent tiles is 1.
        - Manhattan distance is used as an estimate for movement cost between
          tiles, ignoring dynamic obstacles ('clear' predicate) for simplicity
          and efficiency. This means the heuristic is not admissible.
        - Tiles that are goal tiles but are not 'clear' in the current state
          and not painted with the goal color are considered unsolvable dead ends
          because there is no action to unpaint or clear a painted tile.
        - Robots must have a color ('robot-has') to be able to change color
          ('change_color'). Robots starting without a color cannot acquire one.
        - 'available-color' indicates colors a robot *can* change into if it
          already holds a color.

    Heuristic Initialization:
        In the constructor, the heuristic pre-processes static information from
        `task.static`:
        - It identifies all available colors from 'available-color' facts and
          stores them in `self.available_colors`.
        - It parses tile names from adjacency facts ('up', 'down', etc.) to
          build a mapping from tile names to (row, col) integer coordinates,
          stored in `self.tile_coords`.
        - It builds an adjacency map for tiles based on 'up', 'down', 'left',
          'right' facts, storing it in `self.tile_neighbors`. This map represents
          the static grid connectivity.
        - It stores the goal facts from `task.goals`.

    Step-By-Step Thinking for Computing Heuristic:
        The heuristic function `__call__(self, node)` computes the heuristic
        value for a given state (`node.state`).
        1. Initialize the total heuristic value `h` to 0.
        2. Extract dynamic state information: Iterate through the facts in the
           current `state` to find robot locations (`robot-at`), robot colors
           (`robot-has`), clear tiles (`clear`), and painted tiles (`painted`).
           Store these in dictionaries/sets (`robot_locations`, `robot_colors`,
           `clear_tiles`, `painted_tiles`).
        3. Check for basic unsolvability: If there are no robots recorded in
           `robot_locations`, the state is likely unsolvable as painting requires
           a robot; return infinity (`float('inf')`).
        4. Identify unpainted goal tiles that need painting: Iterate through the
           stored goal facts (`self.goals`). For each goal fact
           `(painted tile_X color_Y)`:
           a. Check if the fact `(painted tile_X color_Y)` is already true in the
              current state (`painted_tiles`). If yes, this goal is satisfied;
              continue to the next goal.
           b. If the goal is not satisfied, the tile `tile_X` needs to be painted
              with `color_Y`. To be paintable, `tile_X` must be `clear`. Check if
              `(clear tile_X)` is true in `clear_tiles`.
           c. If `(clear tile_X)` is false, the tile is not clear and not painted
              with the goal color (it must be painted with a different color).
              Since there's no way to unpaint, this state is likely a dead end
              or unsolvable; return infinity.
           d. If `(clear tile_X)` is true, add the tuple `(tile_X, color_Y)` to a
              list `unpainted_goals`.
        5. If the list of `unpainted_goals` is empty, all goal tiles are painted
           correctly; return `h = 0`.
        6. For each `(tile_X, color_Y)` in the `unpainted_goals` list:
           a. Initialize `min_robot_paint_cost` for this specific tile to infinity.
           b. Find the minimum cost for *any* robot to paint `tile_X` with `color_Y`.
              Iterate through each robot `R` with its current location `tile_R`
              and color `color_R_held` (obtained from `robot_locations` and
              `robot_colors`).
              i. Calculate the estimated color change cost: If `color_R_held` is
                 already `color_Y`, the cost is 0. If `color_R_held` is different
                 from `color_Y` and `color_Y` is in `self.available_colors`, the
                 cost is 1 (for the `change_color` action). If the robot has no
                 color initially or `color_Y` is not available, this robot cannot
                 acquire the target color; consider its cost infinity for this tile
                 and move to the next robot.
              ii. If the color cost is not infinity, calculate the estimated
                  movement cost. The robot needs to reach a tile adjacent to
                  `tile_X` to paint it. Calculate the minimum Manhattan distance
                  from the robot's current tile `tile_R` to any tile `tile_Adj`
                  that is a neighbor of `tile_X` (using `self.tile_coords` and
                  `self.tile_neighbors`). Let this be `min_move_cost_to_neighbor`.
                  If `tile_X` has no neighbors in the map (shouldn't happen in a
                  connected grid), consider the cost infinity.
              iii. The total estimated cost for robot `R` to paint `tile_X` is
                   `color_cost + min_move_cost_to_neighbor + 1` (for the paint
                   action itself).
              iv. Update `min_robot_paint_cost` for `tile_X` with the minimum
                  cost found among all robots.
           c. After checking all robots, if `min_robot_paint_cost` for `tile_X`
              is still infinity, it means no robot can paint this tile; the state
              is unsolvable. Return infinity.
           d. Add the calculated `min_robot_paint_cost` for `tile_X` to the
              `total_heuristic`.
        7. Return the final `total_heuristic` value.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.tile_coords = {}
        self.tile_neighbors = {}
        self.available_colors = set()

        # Pre-process static facts to build grid structure and available colors
        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == 'available-color':
                self.available_colors.add(parts[1])
            elif predicate in ['up', 'down', 'left', 'right']:
                tile1_name, tile2_name = parts[1], parts[2]

                # Parse coordinates and store
                coords1 = parse_tile_name(tile1_name)
                coords2 = parse_tile_name(tile2_name)
                if coords1:
                    self.tile_coords[tile1_name] = coords1
                if coords2:
                    self.tile_coords[tile2_name] = coords2

                # Store adjacency (undirected graph)
                if tile1_name not in self.tile_neighbors:
                    self.tile_neighbors[tile1_name] = []
                if tile2_name not in self.tile_neighbors:
                    self.tile_neighbors[tile2_name] = []
                self.tile_neighbors[tile1_name].append(tile2_name)
                self.tile_neighbors[tile2_name].append(tile1_name)

        # Remove duplicates from neighbor lists (adjacency is symmetric)
        for tile in self.tile_neighbors:
            self.tile_neighbors[tile] = list(set(self.tile_neighbors[tile]))

    def __call__(self, node):
        state = node.state

        # Extract dynamic state information
        robot_locations = {}
        robot_colors = {}
        clear_tiles = set()
        painted_tiles = {} # {tile_name: color}

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'robot-at':
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif predicate == 'robot-has':
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif predicate == 'clear':
                clear_tiles.add(parts[1])
            elif predicate == 'painted':
                tile, color = parts[1], parts[2]
                painted_tiles[tile] = color

        # Check if any robot exists (should always be at least one in solvable problems)
        if not robot_locations:
            # Cannot paint without robots
            return float('inf')

        # Identify unpainted goal tiles that need painting
        unpainted_goals = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile_X, color_Y = parts[1], parts[2]
                
                # Check if the goal tile is already painted correctly
                if tile_X in painted_tiles and painted_tiles[tile_X] == color_Y:
                    continue # Goal already satisfied for this tile

                # If not painted correctly, it must be clear to be paintable
                if tile_X not in clear_tiles:
                    # Tile is not clear and not painted with the goal color.
                    # This state is likely a dead end or unsolvable as there's
                    # no action to unpaint/clear a painted tile.
                    return float('inf')

                # Tile needs painting and is clear
                unpainted_goals.append((tile_X, color_Y))

        # If all goal tiles are painted correctly, heuristic is 0
        if not unpainted_goals:
            return 0

        total_heuristic = 0

        # Calculate cost for each unpainted goal tile independently
        for tile_X, color_Y in unpainted_goals:
            min_robot_paint_cost = float('inf')

            # Ensure the target tile exists in our precomputed map (should if it's a goal tile)
            if tile_X not in self.tile_coords:
                 # Invalid tile name or map building issue
                 return float('inf')

            # Find the minimum cost for any robot to paint this tile
            for robot_R, tile_R in robot_locations.items():
                color_R_held = robot_colors.get(robot_R)

                # Robot must have a color to be able to change it or paint
                if color_R_held is None:
                     continue # This robot cannot paint

                # Calculate color change cost
                color_cost = 0
                if color_R_held != color_Y:
                    if color_Y in self.available_colors:
                        color_cost = 1 # change_color action needed
                    else:
                        # Robot cannot acquire the target color
                        continue # This robot cannot paint this tile

                # Calculate minimum move cost to reach a tile adjacent to tile_X
                min_move_cost_to_neighbor = float('inf')
                
                # Ensure robot's current tile exists in our map
                if tile_R not in self.tile_coords:
                     # Should not happen, but defensive check
                     continue

                (r_R, c_R) = self.tile_coords[tile_R]

                # Ensure the target tile has neighbors in our map (should if it's a valid tile)
                if tile_X not in self.tile_neighbors:
                     # Invalid tile name or map building issue
                     return float('inf')

                for tile_Adj in self.tile_neighbors[tile_X]:
                    # Ensure neighbor tile exists in our map
                    if tile_Adj not in self.tile_coords:
                         continue

                    (r_Adj, c_Adj) = self.tile_coords[tile_Adj]
                    # Manhattan distance from robot's current tile to the adjacent tile
                    dist = abs(r_R - r_Adj) + abs(c_R - c_Adj)
                    min_move_cost_to_neighbor = min(min_move_cost_to_neighbor, dist)

                # If no adjacent tiles were found or reachable (shouldn't happen in a connected grid)
                if min_move_cost_to_neighbor == float('inf'):
                     continue # This robot cannot paint this tile

                # Total estimated cost for this robot to paint this tile
                # = color_cost + move_cost_to_neighbor + paint_action_cost (1)
                robot_paint_cost = color_cost + min_move_cost_to_neighbor + 1
                min_robot_paint_cost = min(min_robot_paint_cost, robot_paint_cost)

            # If no robot can paint this tile after checking all robots, the state is unsolvable
            if min_robot_paint_cost == float('inf'):
                return float('inf')

            # Add the minimum cost for this tile to the total heuristic
            total_heuristic += min_robot_paint_cost

        return total_heuristic
