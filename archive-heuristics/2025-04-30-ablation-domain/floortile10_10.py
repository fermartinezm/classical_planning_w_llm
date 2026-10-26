from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper functions (can be defined outside the class but in the same file)

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle cases like "(<predicate>)" or "(<predicate> <arg1> ...)"
    content = fact.strip()[1:-1].strip()
    if not content:
        return []
    # Simple split by space works for this domain's fact structure
    return content.split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Uses fnmatch for wildcard support.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_coords(tile_name):
    """Parses tile name 'tile_r_c' into (row, col) integers."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases like 'tile_abc_1'
            return None
    # Handle cases like 'tile_0_1_extra' or non-'tile_' names
    return None

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles based on their names."""
    if tile1_name == tile2_name:
        return 0

    coords1 = parse_tile_coords(tile1_name)
    coords2 = parse_tile_coords(tile2_name)

    if coords1 is None or coords2 is None:
        # If parsing fails for either tile, we cannot calculate distance using grid coordinates.
        # This might indicate a non-grid tile or an issue.
        # Return a large value to represent a high cost/impossibility via grid moves.
        return 1000000 # Effectively infinity

    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    state by summing the necessary clear and paint actions for unsatisfied
    goal tiles, and adding the estimated cost to move the robot to the nearest
    unsatisfied tile and obtain the correct color for it. It is designed for
    a single-robot scenario.

    # Assumptions
    - Tiles involved in robot location or goal conditions are named in the
      format 'tile_row_col' allowing coordinate parsing for distance calculation.
    - Robot can only hold one color at a time.
    - To change color when holding a different color, the robot must first
      use the current color (e.g., by painting) before picking up a new one.
      This is estimated as 2 actions (paint + pickup).
    - Picking up a color when holding none costs 1 action (pickup).
    - Moving between adjacent tiles costs 1 action. Manhattan distance is used
      as a lower bound for movement cost on the grid inferred from tile names.
    - The problem involves a single robot.

    # Heuristic Initialization
    - Extracts the goal conditions, specifically the required color for each
      goal tile, storing them in `self.goal_colors`.
    - Static facts are not explicitly used beyond the goal definition, as
      tile connectivity (for distance) is inferred from tile names.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all goal conditions of the form `(painted ?t ?c)`. Store the
       required color for each goal tile in `self.goal_colors`. This is done
       during initialization.
    2. In the `__call__` method, find the robot's current location (`robot_tile`)
       and the color it is currently holding (`robot_color`) by iterating
       through the facts in the current state. This heuristic assumes a single robot.
    3. Initialize the heuristic value `h` to 0.
    4. Identify the set of tiles that are goal tiles but are *not* currently
       painted with the correct goal color (`unsatisfied_tiles`). A tile is
       unsatisfied if the goal requires it to be painted with color C, but
       it is currently clear or painted with a different color C'.
    5. Identify the set of tiles that are currently painted with the *wrong*
       color (`tiles_to_clear`). These are tiles where the state has
       `(painted t c_current)` but the goal requires `(painted t c_goal)`
       where `c_current != c_goal`.
    6. Add the number of tiles in `tiles_to_clear` to `h`. This accounts for
       the necessary `clear` action for each such tile.
    7. Add the number of tiles in `unsatisfied_tiles` to `h`. This accounts
       for the necessary `paint` action for each such tile.
    8. If `unsatisfied_tiles` is empty, the goal is reached for all relevant
       tiles, and `h` is 0. Return `h`.
    9. If `unsatisfied_tiles` is not empty (meaning the goal is not yet reached):
       a. Find the tile `t_closest` in `unsatisfied_tiles` that is closest
          (using Manhattan distance based on parsed coordinates) to the robot's
          current location `robot_tile`. We only consider unsatisfied tiles
          whose names can be parsed into coordinates.
       b. Add the Manhattan distance between `robot_tile` and `t_closest` to `h`.
          This estimates the minimum number of `move` actions required to reach
          the first tile needing attention.
       c. Determine the goal color `c_closest_goal` required for `t_closest`
          from the pre-calculated `self.goal_colors`.
       d. Calculate the estimated cost to get the color `c_closest_goal`:
          - If the robot has no color (`robot_color` is None), cost is 1 (pickup).
          - If the robot has a different color (`robot_color != c_closest_goal`),
            cost is estimated as 2 actions (one to use the current color, one
            to pickup the new color).
          - If the robot already has the correct color (`robot_color == c_closest_goal`),
            cost is 0.
       e. Add this calculated color cost to `h`.
    10. Return the final value of `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        # Map goal tile to its required color
        self.goal_colors = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_colors[tile] = color

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Find robot location and color (assuming a single robot)
        robot_tile = None
        robot_color = None
        robot_name = None # Store robot name to find its color

        # Find the robot's location and name first
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot_name = parts[1]
                robot_tile = parts[2]
                break # Assuming one robot

        # If robot location found, find its color
        if robot_name:
             for fact in state:
                 parts = get_parts(fact)
                 if parts[0] == "robot-has" and parts[1] == robot_name:
                     robot_color = parts[2]
                     break

        # If no robot location found, check if goal is already met.
        # Otherwise, the state might be invalid or unreachable.
        if robot_tile is None:
             # Check if the state satisfies all goal conditions
             if self.is_goal(state):
                 return 0
             else:
                 # Should not happen in valid problem states unless robot disappears
                 # Return infinity to prune this path
                 return float('inf')

        # 2. Identify unsatisfied goal tiles and tiles needing clearing
        unsatisfied_tiles = set()
        tiles_to_clear = set()
        current_painted_colors = {} # Map tile -> current color if painted

        # Build map of currently painted tiles and their colors
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "painted":
                 tile, color = parts[1], parts[2]
                 current_painted_colors[tile] = color

        # Check each goal tile
        for goal_tile, goal_color in self.goal_colors.items():
            is_goal_satisfied_for_tile = False
            if goal_tile in current_painted_colors:
                current_color = current_painted_colors[goal_tile]
                if current_color == goal_color:
                    is_goal_satisfied_for_tile = True
                else:
                    # Painted with wrong color
                    tiles_to_clear.add(goal_tile)

            # If goal is not satisfied for this tile (either clear or wrong color)
            if not is_goal_satisfied_for_tile:
                 unsatisfied_tiles.add(goal_tile)

        # 3. Calculate base heuristic from clear and paint actions
        h = len(tiles_to_clear) # Cost for clearing wrong tiles
        h += len(unsatisfied_tiles) # Cost for painting unsatisfied tiles

        # 4. Add robot state costs if there are unsatisfied tiles
        if not unsatisfied_tiles:
            # Goal reached (all goal tiles are painted correctly)
            # The heuristic is 0 iff all goal tiles specified in the problem
            # are painted with the correct color. It ignores other state facts.
            return 0

        # Find the closest unsatisfied tile whose name can be parsed
        closest_tile = None
        min_dist = float('inf')

        robot_coords = parse_tile_coords(robot_tile)

        # Only proceed if robot location is parseable (expected for grid domains)
        if robot_coords is not None:
            for tile in unsatisfied_tiles:
                tile_coords = parse_tile_coords(tile)
                if tile_coords is not None: # Only consider parseable unsatisfied tiles
                    dist = manhattan_distance(robot_tile, tile) # manhattan_distance handles parsing internally
                    if dist < min_dist:
                        min_dist = dist
                        closest_tile = tile

        # Add movement cost to the closest unsatisfied tile (if found)
        # and the color cost for that tile.
        if closest_tile is not None:
            h += min_dist

            # Add color cost for the closest unsatisfied tile
            needed_color = self.goal_colors[closest_tile]
            color_cost = 0
            if robot_color is None:
                color_cost = 1 # pickup needed_color
            elif robot_color != needed_color:
                # Assume need to paint something with current color (1) + pickup needed_color (1)
                color_cost = 2
            # If robot_color == needed_color, color_cost is 0

            h += color_cost
        # Else: closest_tile is None (either robot_tile unparseable or all unsatisfied tiles unparseable).
        # In this case, h remains the base clear/paint count.

        return h

    def is_goal(self, state):
        """Helper to check if the state is a goal state based on self.goals."""
        # A state is a goal state if all facts in self.goals are present in the state.
        return self.goals <= state
