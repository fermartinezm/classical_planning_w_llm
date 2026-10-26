from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assuming fact is always in the format '(predicate arg1 ...)'
    fact = fact.strip()
    if fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    # Should not happen for valid PDDL facts in state/goals/static
    return fact.split()

def get_coords(tile_name):
    """Extract (row, col) from tile name like 'tile_R_C'."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            return None # Not a standard tile name format
    return None # Not a tile name starting with 'tile_'

def manhattan_distance(coords1, coords2):
    """Calculate Manhattan distance between two (row, col) tuples."""
    if coords1 is None or coords2 is None:
        # This indicates an issue or unreachable location, treat as infinite distance
        return float('inf')
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the total number of actions required to paint all
    goal tiles with their target colors. It sums the estimated cost for each
    individual unpainted goal tile, plus the cost for changing robot colors
    if necessary. The cost for each tile includes painting, moving a robot
    to the tile's location, and clearing the tile if a robot is on it.

    # Assumptions
    - Tile names follow the format 'tile_R_C' where R and C are integers representing
      row and column coordinates.
    - The grid structure implied by 'up', 'down', 'left', 'right' predicates
      corresponds to movement between adjacent cells in this grid. Manhattan distance
      is used as a lower bound for movement cost.
    - A tile becomes 'clear' only when a robot moves off it. Painted tiles are
      not 'clear'.
    - Goal tiles are initially either clear or already painted with the correct color.
      If a goal tile is painted with the wrong color in the current state, the
      heuristic returns a large value assuming the state is unsolvable.
    - The cost of moving between adjacent tiles is 1. The cost of changing color is 1.
      The cost of painting is 1.

    # Heuristic Initialization
    - Parses all tile names found in the static facts (which define the grid)
      and extracts their (row, col) coordinates based on the 'tile_R_C' format.
    - Stores the required color for each goal tile by parsing the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal tiles that are not currently painted with their required color.
       Simultaneously, check if any goal tile is painted with a *different* color.
       If a goal tile has the wrong color, return a large heuristic value (assuming unsolvability).
    2. Identify the current location of each robot using the `robot-at` facts.
    3. Identify the color currently held by each robot using the `robot-has` facts.
    4. Create a set of tiles currently occupied by robots.
    5. Determine the set of unique colors required by the unpainted goal tiles.
    6. Determine the set of unique colors currently held by the robots.
    7. Initialize the total heuristic cost to 0.
    8. Calculate the color change cost: For each required color (from step 5) that is not
       currently held by *any* robot (from step 6), add 1 to the total heuristic cost.
       This estimates the cost for one robot to change to that color.
    9. For each unpainted goal tile (identified in step 1):
       a. Add 1 to the total heuristic cost (estimating the cost for the paint action itself).
       b. Calculate the minimum Manhattan distance from any robot's current location
          to the location of the goal tile. Use the pre-calculated tile coordinates.
       c. Add this minimum distance to the total heuristic cost. This estimates the
          minimum number of move actions required for a robot to reach the tile's position.
       d. Check if a robot is currently occupying the goal tile (using the set from step 4).
          If it is, add 1 to the total heuristic cost. This estimates the cost to move
          the blocking robot off the tile so it can become clear and paintable.
    10. The final total heuristic value is the sum of all calculated costs.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting tile coordinates and goal requirements.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Map tile names to (row, col) coordinates.
        self.tile_coords = {}
        # Collect all potential tile names from static facts (up, down, left, right)
        all_potential_tile_names = set()
        for fact in self.static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                all_potential_tile_names.add(parts[1])
                all_potential_tile_names.add(parts[2])

        # Parse coordinates for names matching the 'tile_R_C' format
        for tile_name in all_potential_tile_names:
            coords = get_coords(tile_name)
            if coords is not None:
                self.tile_coords[tile_name] = coords

        # Store goal requirements: map goal tile name to required color.
        self.goal_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted" and len(parts) == 3:
                tile_name, color = parts[1], parts[2]
                self.goal_tiles[tile_name] = color

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Step 1 & 2 & 3 & 4: Identify state facts and unpainted goal tiles ---
        unpainted_goals = []
        painted_tiles = {} # Map tile -> color for painted tiles in current state
        robot_locations = {} # Map robot -> tile location
        robot_colors = {} # Map robot -> color held

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "painted" and len(parts) == 3:
                painted_tiles[parts[1]] = parts[2]
            elif parts[0] == "robot-at" and len(parts) == 3:
                robot_locations[parts[1]] = parts[2]
            elif parts[0] == "robot-has" and len(parts) == 3:
                robot_colors[parts[1]] = parts[2]

        occupied_tiles = set(robot_locations.values())

        for goal_tile, required_color in self.goal_tiles.items():
            if goal_tile in painted_tiles:
                current_color = painted_tiles[goal_tile]
                if current_color != required_color:
                    # Goal tile painted with the wrong color - likely unsolvable
                    return float('inf') # Return a large value

                # Else: Correctly painted, goal met for this tile
            else:
                # Tile is not painted (implies clear or occupied by robot)
                unpainted_goals.append((goal_tile, required_color))

        # If all goal tiles are already painted correctly, the heuristic is 0.
        if not unpainted_goals:
            return 0

        total_cost = 0

        # --- Step 5, 6, 8: Calculate color change cost ---
        needed_colors = {color for _, color in unpainted_goals}
        available_robot_colors = set(robot_colors.values())

        for color in needed_colors:
            if color not in available_robot_colors:
                total_cost += 1 # Cost to change color on one robot

        # --- Step 9: Calculate cost per unpainted goal tile ---
        for tile_name, required_color in unpainted_goals:
            # a. Cost for paint action
            total_cost += 1

            # b. Calculate minimum Manhattan distance from any robot to the tile
            tile_coords = self.tile_coords.get(tile_name)
            min_dist = float('inf')

            if tile_coords is not None:
                for robot_loc in robot_locations.values():
                    robot_coords = self.tile_coords.get(robot_loc)
                    if robot_coords is not None:
                        dist = manhattan_distance(robot_coords, tile_coords)
                        min_dist = min(min_dist, dist)

            # c. Add minimum distance as move cost
            # If there are no robots, min_dist remains inf. This case should ideally
            # not happen in solvable problems unless robots can be created.
            # Assuming at least one robot exists and its location is known.
            if min_dist != float('inf'):
                 total_cost += min_dist
            # else: problem might be unsolvable if no robots can reach the tile,
            # but we don't return inf here as other tiles might still be paintable.
            # The lack of moves will simply not add to the cost for this tile.

            # d. Add cost if a robot is occupying the tile
            if tile_name in occupied_tiles:
                total_cost += 1 # Cost to move the blocking robot off

        return total_cost
