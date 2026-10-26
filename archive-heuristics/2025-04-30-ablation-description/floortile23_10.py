# Required imports
import math # For infinity

# Helper functions (can be defined outside the class)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assumes fact is a string like "(predicate arg1 arg2)"
    # Returns a list of strings [predicate, arg1, arg2]
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def get_coords(tile_name):
    """Parse tile name 'tile_row_col' into (row, col) tuple."""
    # Assumes tile names are always in the format tile_R_C
    # Example: 'tile_1_5' -> (1, 5)
    parts = tile_name.split('_')
    if len(parts) != 3 or parts[0] != 'tile':
        # Handle unexpected format, although problem assumes standard naming
        # Return something that results in large distance or error
        return (-1, -1) # Indicate parsing failure
    try:
        row = int(parts[1])
        col = int(parts[2])
        return (row, col)
    except ValueError:
        # Handle non-integer row/col, unlikely in valid problems
        return (-1, -1) # Indicate parsing failure


def manhattan_distance(tile1, tile2):
    """Calculate Manhattan distance between two tiles using their names."""
    r1, c1 = get_coords(tile1)
    r2, c2 = get_coords(tile2)
    # Check for invalid coordinates resulting from parsing errors
    if r1 == -1 or r2 == -1:
        return math.inf # Cannot calculate distance for invalid tiles
    return abs(r1 - r2) + abs(c1 - c2)

# Define the heuristic class
# Assuming Heuristic base class exists and this class will inherit from it
# from heuristics.heuristic_base import Heuristic
class floortileHeuristic: # Inherit from Heuristic if needed by the framework
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    The heuristic estimates the cost to paint all goal tiles with the correct color.
    It sums the estimated cost for each goal tile that is not yet painted correctly.
    The estimated cost for a single tile includes:
    1. The paint action itself (cost 1).
    2. The cost to change the robot's color if needed (estimated cost 1).
    3. The minimum movement cost for a robot to reach a tile adjacent to the target tile (estimated using Manhattan distance).

    # Assumptions
    - A tile painted with the wrong color makes the problem unsolvable.
    - Robots always hold a color (never in a 'free-color' state) and can change to any available color.
    - Changing color costs 1 action if the target color is available.
    - Movement cost is estimated using Manhattan distance on the grid, ignoring obstacles (other robots or painted tiles). This is a relaxation.
    - Any robot can be used to paint any tile. The heuristic considers the minimum cost over all robots.
    - A robot must be on a tile adjacent to the target tile to paint it.
    - Goal tiles are always represented in the state facts and are either 'clear' or 'painted'.
    - All colors required by the goal are available colors.

    # Heuristic Initialization
    - Extract goal conditions (`painted` facts).
    - Build an adjacency map for tiles based on `up`, `down`, `left`, `right` static facts to find adjacent tiles.
    - Identify available colors. Check if all goal colors are available. If not, the problem is unsolvable from the start.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Parse the current state to find:
       - Robot locations (`robot-at`).
       - Robot held colors (`robot-has`).
       - Painted tiles and their colors (`painted`).
       - Clear tiles (`clear`).
    3. Iterate through each goal fact `(painted ?tile ?goal_color)` extracted during initialization:
       - Check if `goal_color` is in the available colors. If not, return infinity (checked during init, but defensive check here).
       - Determine the current status of the goal tile (painted with which color, or clear).
       - If painted correctly: This goal is met. Continue.
       - If painted with the wrong color: Problem is unsolvable. Return infinity.
       - If clear: The tile needs painting. Calculate the estimated cost for this tile:
           - Initialize tile cost `cost_tile = 1` (for the paint action).
           - Determine if a color change is needed: Check if any robot currently holds `goal_color`. If not, add 1 to `cost_tile`.
           - Estimate movement cost: Find the minimum Manhattan distance from any robot's current location to any tile adjacent to `?tile`. Add this minimum distance to `cost_tile`.
           - Add `cost_tile` to the total heuristic cost.
         - If the goal tile is neither painted nor clear in the state facts (implies an invalid state or missing facts): Return infinity.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals  # Goal conditions (frozenset of strings)
        static_facts = task.static  # Static facts (frozenset of strings)

        # Extract goal painted facts: set of (tile, color) tuples
        self.goal_painted_facts = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts and parts[0] == "painted" and len(parts) == 3:
                 self.goal_painted_facts.add((parts[1], parts[2]))

        # Build adjacency map: tile -> list of adjacent tiles
        self.adj = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] in ["up", "down", "left", "right"] and len(parts) == 3:
                tile1, tile2 = parts[1], parts[2]
                self.adj.setdefault(tile1, []).append(tile2)
                self.adj.setdefault(tile2, []).append(tile1) # Adjacency is symmetric

        # Identify available colors
        self.available_colors = {
            parts[1]
            for fact in static_facts
            for parts in [get_parts(fact)]
            if parts and parts[0] == "available-color" and len(parts) == 2
        }

        # Pre-check if any goal color is not available. If so, the problem is unsolvable.
        # This check could also be done per tile in __call__, but doing it once here is efficient.
        self._is_initially_unsolvable = False
        for _, goal_color in self.goal_painted_facts:
            if goal_color not in self.available_colors:
                self._is_initially_unsolvable = True
                break


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        # If problem was determined unsolvable during initialization
        if self._is_initially_unsolvable:
            return math.inf

        state = node.state  # Current world state (frozenset of strings)

        # Parse current state
        robot_locations = {} # robot -> tile
        robot_colors = {}    # robot -> color
        painted_tiles = {}   # tile -> color
        clear_tiles = set()  # set of clear tiles
        all_tiles_in_state = set() # Keep track of all tiles mentioned in state facts

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == "robot-at" and len(parts) == 3:
                robot, tile = parts[1:]
                robot_locations[robot] = tile
                all_tiles_in_state.add(tile)
            elif parts[0] == "robot-has" and len(parts) == 3:
                robot, color = parts[1:]
                robot_colors[robot] = color
            elif parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1:]
                painted_tiles[tile] = color
                all_tiles_in_state.add(tile)
            elif parts[0] == "clear" and len(parts) == 2:
                tile = parts[1]
                clear_tiles.add(tile)
                all_tiles_in_state.add(tile)

        total_cost = 0

        # Check each goal painted fact
        for goal_tile, goal_color in self.goal_painted_facts:

            # We already checked if goal_color is available in __init__

            # Determine the current status of the goal tile
            current_painted_color = painted_tiles.get(goal_tile)
            is_clear = goal_tile in clear_tiles
            is_tile_mentioned_in_state = goal_tile in all_tiles_in_state

            if current_painted_color == goal_color:
                continue # Goal met for this tile

            if current_painted_color is not None and current_painted_color != goal_color:
                # Tile is painted with the wrong color
                return math.inf # Unsolvable

            if is_clear:
                # Tile needs painting
                cost_tile = 1 # Paint action

                # Color change cost
                # Check if any robot currently has the goal color
                any_robot_has_goal_color = any(color == goal_color for color in robot_colors.values())

                if not any_robot_has_goal_color:
                    # Assume one robot needs to change color. Cost = 1.
                    # This assumes there is a robot that can change color
                    # (i.e., it holds *some* color and goal_color is available).
                    # We assume robots always hold some color based on examples.
                    # We already checked goal_color is available.
                    cost_tile += 1

                # Estimated movement cost
                min_move_cost = math.inf

                # Get adjacent tiles for the goal tile
                adjacent_tiles = self.adj.get(goal_tile, [])

                if not adjacent_tiles or not robot_locations:
                     # Cannot paint if no adjacent tiles or no robots
                     return math.inf

                # Find the minimum distance from any robot to any adjacent tile
                for robot_loc in robot_locations.values():
                    for adj_tile in adjacent_tiles:
                         # Note: This Manhattan distance ignores the 'clear' precondition for movement.
                         # It's a relaxed distance.
                         dist = manhattan_distance(robot_loc, adj_tile)
                         min_move_cost = min(min_move_cost, dist)

                if min_move_cost == math.inf:
                     # Should not happen if adjacent_tiles and robot_locations are not empty,
                     # and get_coords/manhattan_distance work correctly.
                     return math.inf

                cost_tile += min_move_cost
                total_cost += cost_tile

            elif not is_tile_mentioned_in_state:
                 # Goal tile is not mentioned at all in the state facts.
                 # This implies it cannot be reached or interacted with. Unsolvable.
                 return math.inf
            # else: The tile is not painted correctly, not painted wrong, and not clear.
            # This case should not happen in a valid state based on domain predicates.
            # If is_tile_mentioned_in_state is True, and it's not painted correctly,
            # it must be either painted wrong (handled) or clear (handled).
            # So this 'else' branch should theoretically be unreachable in valid states.


        # Heuristic is 0 only if all goal painted facts are satisfied.
        # This happens if the loop finishes without returning inf and total_cost remains 0.
        return total_cost
