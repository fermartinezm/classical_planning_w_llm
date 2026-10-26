from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at package1 city1-1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Domain-specific helper function to get tile coordinates
def get_coords(tile_name):
    """Parses a tile name like 'tile_r_c' into integer coordinates (r, c)."""
    parts = tile_name.split('_')
    # Assuming tile names are always in the format 'tile_row_col'
    try:
        row = int(parts[1])
        col = int(parts[2])
        return (row, col)
    except (IndexError, ValueError):
        # Handle unexpected tile name format gracefully
        print(f"Warning: Could not parse tile coordinates from '{tile_name}'.")
        return None # Or raise an error, depending on desired behavior

# Standard helper function for distance calculation
def manhattan_distance(coords1, coords2):
    """Calculates the Manhattan distance between two points (r1, c1) and (r2, c2)."""
    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all goal tiles
    with their required colors. It considers the number of tiles that still need
    to be painted, the number of color changes required to paint them, and an
    estimated cost for the robot's movement to reach those tiles.

    # Assumptions
    - All tiles that need to be painted according to the goal are initially 'clear'
      if they are not already painted with the correct color.
    - Once a tile is painted with the correct color, it remains so and does not
      need repainting or clearing.
    - The grid structure is regular, and tile names follow the 'tile_row_col' format,
      allowing for Manhattan distance calculation.
    - The estimated movement cost is a non-admissible approximation that does not
      strictly account for dynamic 'clear' statuses of tiles during pathfinding.
    - There is only one robot.

    # Heuristic Initialization
    - Extracts the goal conditions to identify the set of tiles that must be painted
      and their required colors. This set `self.goal_painted` is stored for quick lookup.
    - Static facts from `task.static` are not explicitly stored as class members
      because the heuristic logic primarily relies on the goal state and the current
      state, using helper functions (`get_coords`, `manhattan_distance`) that
      implicitly understand the grid structure based on tile naming.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Robot State:** Find the robot's current location (as a tile name string)
        and the color it is currently holding by iterating through the facts in the
        current state. Parse the robot's tile location string into grid coordinates (row, col).
    2.  **Identify Unpainted Goal Tiles:** Determine which tiles specified in the goal
        as `(painted tile_name color)` are *not* currently painted with that color
        in the current state. Collect these as a set of `(tile_name, goal_color)` pairs.
    3.  **Goal Check:** If the set of unpainted goal tiles is empty, the current state
        is a goal state, and the heuristic value is 0.
    4.  **Calculate Heuristic Components (if not a goal state):**
        a.  **Paint Cost:** The most basic cost is performing the paint action for each
            tile that needs painting. This component is simply the total number of
            unpainted goal tiles.
        b.  **Color Change Cost:** The robot can only hold one color at a time. To paint
            tiles requiring different colors, the robot must perform `change_color` actions.
            Count the number of *distinct* colors required by the unpainted goal tiles.
            If the robot's current color is one of these needed colors, it needs to
            switch colors for `(number of distinct needed colors) - 1` other color groups.
            If the robot's current color is *not* one of the needed colors, it needs to
            switch colors for `(number of distinct needed colors)` color groups (first to
            get a needed color, then for each subsequent needed color).
        c.  **Movement Cost:** The robot needs to move to a tile adjacent to each unpainted
            goal tile before it can paint it. Estimate the total movement effort required.
            A simple, non-admissible estimate is used: For each unpainted goal tile,
            calculate the Manhattan distance from the robot's current position to that
            tile's position. Subtract 1 from this distance (since the robot only needs
            to be adjacent, not on the tile itself). Sum these adjusted distances for
            all unpainted goal tiles. Divide this total sum by 2 as a further relaxation,
            assuming that some moves contribute to reaching the vicinity of multiple
            tiles or that movement is relatively less costly than painting/color changes
            in the overall plan structure.
    5.  **Sum Components:** The total heuristic value is the sum of the Paint Cost,
        Color Change Cost, and estimated Movement Cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        # Store goal conditions: set of (tile_name, color) tuples for painted tiles
        self.goal_painted = set()
        for goal in task.goals:
            # Assuming goal facts are always in the form (painted tile_name color)
            if match(goal, "painted", "*", "*"):
                _, tile_name, color = get_parts(goal)
                self.goal_painted.add((tile_name, color))

        # Static facts like (up tile_1_1 tile_0_1), (available-color white), etc.,
        # are available in task.static but not explicitly stored as they are
        # not directly needed for the logic of this specific heuristic calculation.
        # The grid structure they define is implicitly used by helper functions.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings)

        # 1. Identify robot's current location and color
        robot_pos_str = None
        robot_color = None
        # Assuming there is only one robot and its name is 'robot1' or similar,
        # and robot-at and robot-has facts exist in the state.
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                # Extract the second argument which is the tile name
                robot_pos_str = get_parts(fact)[2]
            if match(fact, "robot-has", "*", "*"):
                 # Extract the second argument which is the color name
                robot_color = get_parts(fact)[2]

        # Basic check if robot state is found (should always be in valid states)
        if robot_pos_str is None or robot_color is None:
             # This state is likely malformed or represents an impossible scenario
             # within the domain's action semantics. Return infinity or a large value.
             return float('inf')

        robot_coords = get_coords(robot_pos_str)
        if robot_coords is None:
             # Handle case where robot tile name cannot be parsed
             return float('inf')


        # 2. & 3. Identify currently painted tiles and unpainted goal tiles
        current_painted = set()
        for fact in state:
            if match(fact, "painted", "*", "*"):
                _, tile_name, color = get_parts(fact)
                current_painted.add((tile_name, color))

        # Tiles that are in the goal but not currently painted correctly
        unpainted_goals = self.goal_painted - current_painted

        # 4. Goal check
        if not unpainted_goals:
            return 0 # Goal state reached

        # 5. Calculate heuristic components
        h = 0

        # a. Paint Cost: Each unpainted goal tile needs one paint action.
        num_unpainted = len(unpainted_goals)
        h += num_unpainted

        # b. Color Change Cost: Estimate the number of color switches needed.
        needed_colors = {color for tile, color in unpainted_goals}
        num_needed_colors = len(needed_colors)
        color_cost = 0
        if num_needed_colors > 0:
            if robot_color in needed_colors:
                # Robot has one of the needed colors, needs to switch for the others
                color_cost = num_needed_colors - 1
            else:
                # Robot has a color it doesn't need, must switch to a needed color first,
                # then switch for the remaining needed colors.
                color_cost = num_needed_colors
        h += color_cost

        # c. Movement Cost (estimated): Estimate effort to reach vicinity of unpainted tiles.
        movement_cost_sum = 0
        for tile_name, color in unpainted_goals:
            tile_coords = get_coords(tile_name)
            if tile_coords is None:
                # Handle case where a goal tile name cannot be parsed
                return float('inf')

            dist = manhattan_distance(robot_coords, tile_coords)
            # Robot needs to be adjacent (distance 1) to paint.
            # Cost to get adjacent is max(0, dist - 1).
            movement_cost_sum += max(0, dist - 1)

        # Divide the total estimated movement by 2 as a non-admissible relaxation.
        # This assumes moves can potentially cover distance towards multiple targets
        # or that movement is relatively cheaper.
        h += movement_cost_sum // 2

        return h
