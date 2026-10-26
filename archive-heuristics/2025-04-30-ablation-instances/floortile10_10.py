from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts defensively
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        # Return empty list for invalid format
        return []
    return fact[1:-1].split()

# Helper function to match PDDL facts (similar to the example)
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(painted tile_1_2 black)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function to parse tile names like 'tile_row_col'
def parse_tile_name(tile_name):
    """Parses a tile name string 'tile_row_col' into a (row, col) tuple of integers."""
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            # Handle unexpected tile name format - return None or raise error
            # print(f"Warning: Unexpected tile name format: {tile_name}")
            return None
    except ValueError:
        # Handle cases where row/col are not integers - return None or raise error
        # print(f"Warning: Could not parse row/col from tile name: {tile_name}")
        return None

# Helper function to calculate Manhattan distance between two tiles
def manhattan_distance(tile_name1, tile_name2):
    """Calculates the Manhattan distance between two tiles given their names."""
    coords1 = parse_tile_name(tile_name1)
    coords2 = parse_tile_name(tile_name2)
    if coords1 is None or coords2 is None:
        # Indicate invalid distance if parsing failed
        return float('inf')
    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are not yet correctly painted. It considers the number of tiles needing
    painting, the number of distinct colors required, and an estimate of the
    movement cost based on Manhattan distance. It is designed for greedy best-first
    search and is not necessarily admissible.

    # Assumptions
    - The goal is defined by a set of `(painted tile color)` facts.
    - Tiles not specified as `painted` in the goal should remain `clear`.
    - Tiles are arranged in a grid structure, and their names follow the format `tile_row_col`.
    - Movement actions (`move_up`, `move_down`, `move_left`, `move_right`) have a cost of 1.
    - Painting actions (`paint_up`, etc.) have a cost of 1.
    - Color change action (`change_color`) has a cost of 1.
    - Tiles that are currently painted with the wrong color cannot be fixed within the provided domain actions. We assume solvable instances do not require fixing wrongly painted tiles.
    - There is only one robot, named 'robot1'.

    # Heuristic Initialization
    - Extracts the set of goal `(painted tile color)` facts from the task definition.
    - Static facts like adjacency are not explicitly stored as Manhattan distance is derived from tile names.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Unpainted Goal Tiles:
       - Iterate through the goal facts stored during initialization and the current state facts.
       - Create a set `tiles_to_paint` containing `(tile, color)` tuples for all tiles
         that are specified as `(painted tile color)` in the goal but are *not*
         present as `(painted tile color)` in the current state.

    2. Calculate Base Cost (Paint Actions):
       - The number of tiles in `tiles_to_paint` represents the minimum number of
         `paint_` actions required (one for each tile). Add `len(tiles_to_paint)`
         to the total heuristic cost.

    3. Estimate Color Change Cost:
       - Identify the set of distinct colors required for the tiles in `tiles_to_paint`.
       - Find the robot's current color by examining the state facts (`robot-has`).
       - The estimated color change cost is the number of distinct colors needed.
         If the robot's current color is one of the needed colors, subtract 1 from
         this count, as the robot already possesses one useful color. This assumes
         an optimal strategy of painting all tiles of one color before switching.
         Add this calculated cost (ensuring it's non-negative) to the total.

    4. Estimate Movement Cost:
       - Find the robot's current tile location by examining the state facts (`robot-at`).
       - For each tile `T` in `tiles_to_paint`, calculate the Manhattan distance
         between the robot's current tile and `T` using the tile name coordinates.
       - Sum these Manhattan distances. This sum serves as a simple, non-admissible
         estimate of the total movement required to get the robot *in the vicinity*
         of all tiles that need painting. Add this sum to the total heuristic cost.

    5. Return Total Cost:
       - The sum of the costs from steps 2, 3, and 4 is the heuristic value for the state.
       - If `tiles_to_paint` is empty, it means all goal painted facts are met,
         so the state is a goal state (with respect to painting), and the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal painted facts.
        """
        # Store goal painted facts for quick lookup
        self.goal_painted_facts = set()
        for goal_fact in task.goals:
            if match(goal_fact, "painted", "*", "*"):
                self.goal_painted_facts.add(goal_fact)

        # Static facts are not explicitly needed for this heuristic's calculation,
        # as tile coordinates are derived from names and robot state is dynamic.
        # self.static_facts = task.static

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach a goal state from the current state.
        """
        state = node.state  # Current world state (frozenset of fact strings)

        # 1. Identify Unpainted Goal Tiles and Robot State
        current_painted_facts = set()
        robot_tile = None
        robot_color = None

        for fact in state:
            if match(fact, "painted", "*", "*"):
                current_painted_facts.add(fact)
            elif match(fact, "robot-at", "robot1", "*"):
                # Assuming only one robot named 'robot1'
                parts = get_parts(fact)
                if len(parts) == 3:
                     robot_tile = parts[2]
            elif match(fact, "robot-has", "robot1", "*"):
                 # Assuming only one robot named 'robot1'
                 parts = get_parts(fact)
                 if len(parts) == 3:
                      robot_color = parts[2]

        # Tiles that are in the goal but not currently painted correctly
        tiles_to_paint = set() # Stores (tile_name, color) tuples
        colors_needed = set()

        for goal_fact in self.goal_painted_facts:
            if goal_fact not in current_painted_facts:
                # This tile needs to be painted
                parts = get_parts(goal_fact)
                if len(parts) == 3: # Should be (painted tile color)
                    tile_name = parts[1]
                    color = parts[2]
                    tiles_to_paint.add((tile_name, color))
                    colors_needed.add(color)

        # If no tiles need painting, we are in a goal state (wrt painting)
        if not tiles_to_paint:
            return 0

        total_cost = 0

        # 2. Calculate Base Cost (Paint Actions)
        # Each unpainted tile needs at least one paint action
        total_cost += len(tiles_to_paint)

        # 3. Estimate Color Change Cost
        # Estimate the number of color changes needed.
        # At least one change is needed for each distinct color,
        # unless the robot already has one of the needed colors.
        num_colors_needed = len(colors_needed)
        color_cost = num_colors_needed
        if robot_color in colors_needed:
             # If the robot already has one of the needed colors,
             # it might save one change operation assuming optimal grouping.
             color_cost -= 1
        # Ensure color_cost is non-negative
        color_cost = max(0, color_cost)
        total_cost += color_cost

        # 4. Estimate Movement Cost
        movement_cost = 0
        if robot_tile: # Ensure robot location is known from the state
            for tile_name, _ in tiles_to_paint:
                # Sum of Manhattan distances from robot to each unpainted tile.
                # This is an overestimate of total travel but guides towards relevant areas.
                movement_cost += manhattan_distance(robot_tile, tile_name)

        # 5. Return Total Cost
        return total_cost
