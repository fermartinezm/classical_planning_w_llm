from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Import math for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def parse_tile_name(tile_name):
    """
    Parses a tile name in the format 'tile_row_col' into (row, col) integers.
    Returns None if the format is unexpected.
    """
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases where row/col are not integers
            return None
    return None # Handle cases where name format is not 'tile_r_c'

def manhattan_distance(tile1_name, tile2_name):
    """
    Calculates the Manhattan distance between two tiles based on their names
    'tile_r_c'. Returns a large value if parsing fails or names are invalid.
    """
    coords1 = parse_tile_name(tile1_name)
    coords2 = parse_tile_name(tile2_name)

    if coords1 is None or coords2 is None:
        # If parsing fails for either tile, they are likely not connected
        # or the names are malformed. Return a large value.
        # Check for identity first as a fallback for non-standard names if needed,
        # but assuming 'tile_r_c' format is standard for tiles.
        if tile1_name == tile2_name:
             return 0
        return 1000000 # A large integer representing high cost

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint or recolor
    all goal tiles that are not currently in their goal state. It sums the estimated
    cost for each individual tile that is not yet painted with the correct color.
    The cost for a single tile is estimated as 1 (for the paint/recolor action itself)
    plus the minimum cost for a robot to reach the tile and acquire the necessary color.

    # Assumptions
    - Tile names are consistently in the format 'tile_row_col', allowing coordinate extraction for Manhattan distance calculation.
    - Manhattan distance provides a reasonable, though potentially non-admissible, estimate of movement cost between tiles.
    - A robot can only hold one color at a time.
    - The cost for a robot to acquire a needed color is simplified: it costs 0 if the robot already has the color, and 1 otherwise (ignoring complex interactions with 'available-color' and other robots/tiles). This is a key non-admissible simplification.
    - The 'clear' predicate indicates a tile is unpainted and needs painting.
    - The 'painted' predicate indicates a tile has a color. If this color is not the goal color, the tile needs recoloring.

    # Heuristic Initialization
    - The constructor extracts the goal conditions, specifically identifying which tile needs to be painted with which color. This information is stored in `self.goal_tiles`.
    - Static facts (like adjacency relations 'up', 'down', 'left', 'right', and 'available-color') are available in `task.static` but are not explicitly stored or used in this simplified heuristic beyond the implicit grid structure assumed by the distance calculation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the current state satisfies all goal conditions. If yes, the heuristic value is 0.
    2. If the state is not a goal state, initialize the total heuristic cost to 0.
    3. Identify the current location and held color for each robot present in the state. Store this information.
    4. Iterate through each tile and its desired color as specified in the goal conditions (`self.goal_tiles`).
    5. For the current goal tile (`tile_T`) and its desired color (`color_goal`):
        a. Check if the state contains the fact `(painted tile_T color_goal)`.
        b. If this fact is present, the goal for this tile is satisfied. Add 0 to the total heuristic for this tile and move to the next goal tile.
        c. If the fact is *not* present, the tile needs to be painted or recolored with `color_goal`. Add 1 to the total heuristic for the paint/recolor action itself.
        d. Estimate the minimum cost required for *any* robot to reach `tile_T` and possess `color_goal`.
            i. Initialize a variable `min_robot_cost_for_tile` to infinity.
            ii. For each robot identified in step 3:
                - Get the robot's current location (`tile_R`) and held color (`color_R`).
                - Calculate the movement cost as the Manhattan distance between `tile_R` and `tile_T`.
                - Calculate the color acquisition cost: If `color_R` is `color_goal`, the cost is 0. Otherwise (if the robot has a different color or no color), the cost is 1 (simplified assumption for acquiring `color_goal`).
                - The total cost for this specific robot to handle `tile_T` is `movement_cost + color_acquisition_cost`.
                - Update `min_robot_cost_for_tile` with the minimum cost found so far across all robots.
            iii. Add `min_robot_cost_for_tile` to the total heuristic.
    6. After processing all goal tiles, the final `total_cost` is the heuristic value for the given state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals  # Goal conditions (frozenset of strings).

        # Extract goal locations and colors for each tile from the goal facts.
        # Maps tile_name (string) -> goal_color (string)
        self.goal_tiles = {}
        for goal in self.goals:
            # Check if the goal fact is a 'painted' predicate
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_tiles[tile] = color

        # Static facts are available in task.static but not explicitly used
        # in this simplified heuristic beyond the implicit grid structure.

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach a goal state from the current state.
        """
        state = node.state  # Current world state (frozenset of strings).

        # If the state is a goal state, the heuristic is 0.
        if self.goals <= state:
             return 0

        # Track robot locations and colors in the current state
        # Maps robot_name (string) -> {'location': tile_name (string), 'color': color_name (string) or None}
        robot_info = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "robot-at" and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                if robot not in robot_info:
                    robot_info[robot] = {'location': None, 'color': None}
                robot_info[robot]['location'] = tile
            elif parts and parts[0] == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                if robot not in robot_info:
                    robot_info[robot] = {'location': None, 'color': None}
                robot_info[robot]['color'] = color

        total_cost = 0  # Initialize the total heuristic cost.

        # Iterate through each tile that needs to be painted/recolored according to the goal
        for goal_tile, goal_color in self.goal_tiles.items():
            # Check if the tile is already painted with the correct color in the current state
            is_goal_painted = f"(painted {goal_tile} {goal_color})" in state

            if not is_goal_painted:
                # This tile is not in its goal state, so it needs attention.
                # Add the cost of the paint/recolor action itself.
                total_cost += 1

                # Estimate the minimum cost to get a robot with the right color to this tile.
                min_robot_cost_for_tile = math.inf # Use infinity from math module

                # Consider each robot as a candidate to paint this tile
                for robot_name, info in robot_info.items():
                    robot_location = info.get('location')
                    robot_color = info.get('color')

                    # A robot must have a known location to be considered
                    if robot_location is None:
                         continue

                    # Calculate the movement cost for this robot to reach the goal tile
                    movement_cost = manhattan_distance(robot_location, goal_tile)

                    # Calculate the color acquisition cost (simplified assumption)
                    color_cost = 0
                    if robot_color != goal_color:
                         # If the robot doesn't have the required color, assume it costs 1 action to acquire it.
                         # This simplifies the logic of dropping current color and picking up the new one,
                         # and ignores the 'available-color' predicate.
                         color_cost = 1

                    # The total estimated cost for this robot to handle this specific tile
                    robot_total_cost = movement_cost + color_cost

                    # Update the minimum cost found across all robots for this tile
                    min_robot_cost_for_tile = min(min_robot_cost_for_tile, robot_total_cost)

                # Add the minimum robot cost required for this tile to the total heuristic.
                # If no robots were found or usable (min_robot_cost_for_tile remains infinity),
                # this might indicate an unsolvable state or parsing issue.
                # Adding infinity would make the total heuristic infinity, which is appropriate
                # for potentially unsolvable subproblems from the heuristic's perspective.
                total_cost += min_robot_cost_for_tile

        return total_cost

