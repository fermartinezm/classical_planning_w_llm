from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Use zip to handle cases where fact parts might be fewer than args (e.g., short facts)
    # fnmatch handles wildcards correctly.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_name(tile_name):
    """Parses 'tile_R_C' into (R, C) integer coordinates."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases where R or C are not integers
            return None
    # Handle cases where the name format is unexpected
    return None

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles based on their names."""
    coords1 = parse_tile_name(tile1_name)
    coords2 = parse_tile_name(tile2_name)
    if coords1 is None or coords2 is None:
        # If tile names cannot be parsed, return a large distance
        return float('inf')
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their correct colors. It considers the cost of painting each tile,
    changing the robot's color, and moving the robot between tiles. It is
    designed for greedy best-first search and is not admissible.

    # Assumptions
    - Tiles are arranged in a grid, and movement cost is approximated by Manhattan distance.
    - The robot can only hold one color at a time.
    - Changing color involves dropping the current color (if any) and picking up the new color.
      Each drop and pickup action costs 1.
    - Tiles needing painting are either 'clear' or painted the wrong color. The heuristic
      counts a 'paint' action for any tile not matching its goal color, implicitly
      assuming preconditions (like 'clear') can be met.
    - The heuristic assumes a single robot is responsible for painting.
    - The heuristic estimates movement cost as the distance to the closest unpainted tile
      plus the number of remaining unpainted tiles minus one (for moves between them).

    # Heuristic Initialization
    - Extracts the goal painting requirements (tile and target color) from the task's goals.
      This creates a dictionary mapping tile names to their required goal colors.
    - Static facts (like grid structure or available colors) are not explicitly stored
      as the grid structure is inferred from tile names for Manhattan distance calculation.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Extract Relevant Information from the State:
       - Identify which tiles are currently painted and their colors.
       - Identify which tiles are currently 'clear'.
       - Determine the robot's current location.
       - Determine the color the robot is currently holding (if any).

    2. Identify Tiles Needing Painting:
       - Compare the current state of each tile with the goal painting requirements.
       - A tile needs painting if the goal specifies a color for it, and the tile is
         currently either 'clear' or painted with a different color than the goal requires.
       - Store these tiles and their required goal colors.

    3. Calculate Base Cost (Paint Actions):
       - If no tiles need painting, the state is a goal state (or satisfies all painting goals),
         so the heuristic value is 0.
       - Otherwise, initialize the heuristic value `h` with the count of tiles needing painting.
         This accounts for the minimum number of 'paint' actions required.

    4. Calculate Color Change Cost:
       - Determine the set of distinct colors required by the tiles needing painting.
       - Consider the robot's current color:
         - If the robot has no color, it needs to perform a 'pickup' action for the first color it needs.
         - If the robot has a color that is *not* among the needed colors, it must 'drop' the current color.
         - For each *subsequent* distinct color needed after the first one is acquired, the robot must
           'drop' the current color and 'pickup' the new color (cost 2 per switch).
       - Sum these estimated color change costs and add them to `h`.

    5. Calculate Movement Cost:
       - Find the robot's current location.
       - Calculate the Manhattan distance from the robot's current location to every tile
         that needs painting.
       - Find the minimum of these distances. Add this minimum distance to `h`. This estimates
         the cost to move the robot to the first tile it needs to paint.
       - Add the number of tiles needing painting minus one to `h`. This is a simplified
         estimate of the moves required to travel between the remaining tiles after reaching the first one.

    6. Return the Total Heuristic Value:
       - The final heuristic value is the sum of the base cost (paint actions),
         the estimated color change cost, and the estimated movement cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        # Extract goal paintings: {tile_name: color_name}
        self.goal_paintings = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color

        # Static facts are not explicitly needed for this heuristic's calculation
        # beyond the assumption that tile names allow coordinate parsing.
        # self.static_facts = task.static

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Extract Relevant Information from the State
        current_paintings = {}
        # current_clear = set() # Not strictly needed for this heuristic logic
        robot_name = None
        robot_loc = None
        robot_color = None # Can be None if robot has no color

        # Assuming a single robot or focusing on the first one found
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot_name = parts[1]
                robot_loc = parts[2]
            elif parts[0] == "robot-has":
                 # Assume this fact belongs to the same robot found via robot-at, or the primary robot
                 if robot_name is None or robot_name == parts[1]:
                     robot_name = parts[1]
                     robot_color = parts[2]
            elif parts[0] == "painted":
                current_paintings[parts[1]] = parts[2]
            # elif parts[0] == "clear":
            #     current_clear.add(parts[1])


        # 2. Identify Tiles Needing Painting
        tiles_to_paint = {} # {tile_name: goal_color}
        for tile, goal_color in self.goal_paintings.items():
            if tile not in current_paintings or current_paintings[tile] != goal_color:
                 # Tile is not painted correctly (either clear or wrong color)
                 tiles_to_paint[tile] = goal_color

        # 3. Calculate Base Cost (Paint Actions)
        if not tiles_to_paint:
            return 0 # Goal state or all painting goals met

        h = 0
        # Add cost for paint actions (1 per tile needing painting)
        h += len(tiles_to_paint)

        # 4. Calculate Color Change Cost
        needed_colors = set(tiles_to_paint.values())
        n_distinct_needed_colors = len(needed_colors)
        color_h = 0

        if n_distinct_needed_colors > 0:
            if robot_color is None:
                # Needs to pickup the first needed color
                color_h += 1 # Pickup cost
                # Needs drop/pickup for subsequent colors
                if n_distinct_needed_colors > 1:
                    color_h += 2 * (n_distinct_needed_colors - 1)
            elif robot_color in needed_colors:
                # Robot has a useful color, doesn't need initial pickup/drop for this specific color
                # Needs drop/pickup for subsequent colors (all needed colors except the one it has)
                if n_distinct_needed_colors > 1:
                    color_h += 2 * (n_distinct_needed_colors - 1)
            else: # robot_color is not None and not in needed_colors
                # Robot has a useless color
                color_h += 1 # Drop current useless color
                color_h += 1 # Pickup first needed color
                # Needs drop/pickup for subsequent colors
                if n_distinct_needed_colors > 1:
                    color_h += 2 * (n_distinct_needed_colors - 1)

        h += color_h

        # 5. Calculate Movement Cost
        if robot_loc: # Ensure robot location is known
            min_dist_to_first_tile = float('inf')
            for tile in tiles_to_paint:
                dist = manhattan_distance(robot_loc, tile)
                min_dist_to_first_tile = min(min_dist_to_first_tile, dist)

            if min_dist_to_first_tile != float('inf'):
                 h += min_dist_to_first_tile # Move to the closest needed tile
                 # Add cost for moving between the remaining tiles (simplified)
                 if len(tiles_to_paint) > 1:
                     h += len(tiles_to_paint) - 1 # Minimum moves to visit N tiles after reaching the first one is N-1

        # 6. Return the estimated total cost
        return h
