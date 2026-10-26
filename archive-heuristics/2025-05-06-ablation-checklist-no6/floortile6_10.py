from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math

# Utility functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def parse_tile_name(tile_name):
    """Parses tile name 'tile_X_Y' into (row, col) tuple."""
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            return (int(parts[1]), int(parts[2]))
        else:
            # Handle unexpected tile name format
            return None
    except ValueError:
        # Handle non-integer row/col
        return None

def manhattan_distance(pos1, pos2):
    """Calculates Manhattan distance between two (row, col) positions."""
    if pos1 is None or pos2 is None:
        return float('inf')
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost to reach the goal by summing:
    1. The number of unpainted goal tiles.
    2. The number of required colors not currently held by any robot.
    3. The sum of minimum Manhattan distances from any robot to a neighbor
       of each unpainted goal tile.

    This heuristic is not admissible but aims to guide greedy best-first search
    by prioritizing states where more goal tiles are painted, robots have
    necessary colors, and robots are closer to unpainted goal tiles.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        We assume tile names follow the 'tile_row_col' format for distance calculation.
        """
        self.goals = task.goals  # Goal conditions (e.g., (painted tile_1_1 white))

        # Extract goal painting requirements: map tile -> required_color
        self.goal_paintings = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'painted' and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_paintings[tile] = color
            # Ignore other potential goal types if any exist, as per domain definition.

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        Returns float('inf') if the state is a dead end (a goal tile is painted
        with the wrong color).
        Returns 0 if the state is a goal state.
        Otherwise, returns a non-negative integer estimate.
        """
        state = node.state

        # Check if the goal is already reached
        if self.goals <= state:
            return 0

        unpainted_goal_tiles = {} # tile -> required_color
        painted_tiles_in_state = {} # tile -> color
        robot_locations = {} # robot -> tile
        robot_colors = set() # set of colors robots currently hold

        # Extract relevant information from the current state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'painted' and len(parts) == 3:
                tile, color = parts[1], parts[2]
                painted_tiles_in_state[tile] = color
            # 'clear' facts are implicitly handled: if a goal tile isn't painted,
            # we assume it's clear and needs painting.
            elif predicate == 'robot-at' and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif predicate == 'robot-has' and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors.add(color)

        # Identify unpainted goal tiles and check for dead ends
        for goal_tile, goal_color in self.goal_paintings.items():
            if goal_tile in painted_tiles_in_state:
                current_color = painted_tiles_in_state[goal_tile]
                if current_color != goal_color:
                    # Tile is painted with the wrong color - dead end
                    return float('inf')
                # Else: Tile is painted with the correct color, goal met for this tile
            else:
                # Tile is not painted. It must be clear to be paintable.
                # Add it to the list of tiles that still need painting.
                unpainted_goal_tiles[goal_tile] = goal_color

        # If all goal tiles are already painted correctly, goal reached (redundant check)
        if not unpainted_goal_tiles:
             return 0

        # --- Calculate Heuristic Components ---

        # 1. Number of paint actions needed
        # Each unpainted goal tile requires at least one paint action.
        paint_cost = len(unpainted_goal_tiles)

        # 2. Color change cost
        # Count how many required colors are not currently held by any robot.
        # Assumes one robot can acquire all missing colors sequentially.
        required_colors = set(unpainted_goal_tiles.values())
        missing_colors = required_colors - robot_colors
        color_cost = len(missing_colors)

        # 3. Movement cost
        # Sum of minimum distances from any robot to a tile adjacent to each
        # unpainted goal tile. This is an approximation.
        total_movement_cost = 0
        robot_pos_list = [parse_tile_name(loc) for loc in robot_locations.values()]

        # If there are unpainted tiles but no robots, it's unsolvable
        if unpainted_goal_tiles and not robot_pos_list:
             return float('inf')

        for tile, required_color in unpainted_goal_tiles.items():
            tile_pos = parse_tile_name(tile)
            if tile_pos is None:
                 # Cannot parse tile name format, indicate problem
                 return float('inf')

            min_dist_to_neighbor = float('inf')

            for robot_pos in robot_pos_list:
                dist = manhattan_distance(robot_pos, tile_pos)
                # Distance to reach a tile adjacent to the target tile.
                # If robot is at target tile (dist=0), needs 1 move to neighbor.
                # If robot is 1 away (dist=1), might be at neighbor (0 moves needed).
                # If robot is >1 away (dist>1), needs at least dist-1 moves to neighbor.
                dist_to_neighbor = max(0, dist - 1)
                min_dist_to_neighbor = min(min_dist_to_neighbor, dist_to_neighbor)

            # If min_dist_to_neighbor is still inf, something went wrong (e.g., no robots)
            if min_dist_to_neighbor == float('inf'):
                 return float('inf')

            total_movement_cost += min_dist_to_neighbor

        # Total heuristic value is the sum of the components
        heuristic_value = paint_cost + color_cost + total_movement_cost

        return heuristic_value
