import re
from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

# Helper functions from Logistics example (useful for parsing PDDL facts)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args has a wildcard at the end
    if len(parts) != len(args) and args[-1] != '*':
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all
    goal tiles with the correct color. It considers the cost of painting,
    changing colors, and moving robots.

    # Assumptions
    - Tiles are arranged in a grid, and their names follow the pattern 'tile_R_C'.
    - Movement is restricted to adjacent clear tiles (up, down, left, right).
    - A robot must be adjacent to a tile and hold the correct color to paint it.
    - Painted tiles are not clear and cannot be moved onto or repainted.
    - The goal is only concerned with certain tiles being painted with specific colors.

    # Heuristic Initialization
    - Parse the grid structure (adjacency and coordinates) from static facts.
    - Store goal conditions.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For a given state, the heuristic estimates the remaining cost as follows:

    1.  Identify all goal predicates `(painted ?tile ?color)` that are not
        satisfied in the current state AND the tile is currently `clear`.
        These are the tiles that still need to be painted. Let this set be U.
    2.  If U is empty, the goal is reached, heuristic is 0.
    3.  The minimum number of paint actions required is `|U|`. Add `|U|` to the heuristic.
    4.  Identify the set of unique colors required for the tiles in U.
    5.  For each required color, check if *any* robot currently holds that color.
        If a required color is not held by *any* robot, at least one `change_color`
        action is needed to acquire that color for the first time. Count the number
        of such colors and add this count to the heuristic.
    6.  Estimate the movement cost. For each tile T in U:
        a.  Find the minimum Manhattan distance from *any* robot's current location
            to *any* tile adjacent to T.
        b.  Sum these minimum distances over all tiles T in U. Add this sum to the heuristic.
        This step overestimates movement if one robot's path covers the movement
        needs for multiple tiles, but provides a useful gradient for greedy search.

    The total heuristic is the sum of costs from steps 3, 5, and 6.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the grid structure from static facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        self.tile_coords = {}
        self.adj_tiles = defaultdict(list)

        # Regex to parse tile names like 'tile_R_C'
        tile_name_pattern = re.compile(r'tile_(\d+)_(\d+)')

        # Build tile coordinates and adjacency list from static facts
        all_objects = set()
        for fact in task.initial_state | task.goals | static_facts:
             parts = get_parts(fact)
             for part in parts:
                 all_objects.add(part)

        for obj in all_objects:
            match_coords = tile_name_pattern.match(obj)
            if match_coords:
                row, col = int(match_coords.group(1)), int(match_coords.group(2))
                self.tile_coords[obj] = (row, col)

        # Build adjacency list from directional facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right']:
                # Directional facts are (direction tile1 tile2) meaning tile1 is [direction] of tile2
                # e.g., (up tile_1_1 tile_0_1) means tile_1_1 is UP from tile_0_1
                # So tile_1_1 and tile_0_1 are adjacent.
                tile1, tile2 = parts[1], parts[2]
                if tile1 in self.tile_coords and tile2 in self.tile_coords:
                    self.adj_tiles[tile1].append(tile2)
                    self.adj_tiles[tile2].append(tile1) # Adjacency is symmetric

        # Remove duplicates from adjacency lists
        for tile in self.adj_tiles:
            self.adj_tiles[tile] = list(set(self.adj_tiles[tile]))


    def get_manhattan_distance(self, tile1, tile2):
        """Calculate Manhattan distance between two tiles using their coordinates."""
        if tile1 not in self.tile_coords or tile2 not in self.tile_coords:
            # Should not happen in valid problems, but handle defensively
            return float('inf')
        r1, c1 = self.tile_coords[tile1]
        r2, c2 = self.tile_coords[tile2]
        return abs(r1 - r2) + abs(c1 - c2)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify unpainted goal tiles that are currently clear
        unpainted_goal_tiles = set() # Stores (tile_name, color) tuples
        needed_colors = set()
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                _, tile, color = get_parts(goal)
                # Check if the tile is NOT painted with the correct color in the current state
                # and is currently clear (required for painting).
                # If it's painted with the wrong color, it's likely a dead end state
                # given the domain rules (no unpaint action, paint requires clear).
                # We only consider goal tiles that are currently clear.
                if f"(painted {tile} {color})" not in state and f"(clear {tile})" in state:
                     unpainted_goal_tiles.add((tile, color))
                     needed_colors.add(color)

        # 2. If U is empty, goal is reached
        if not unpainted_goal_tiles:
            return 0

        total_cost = 0

        # 3. Add cost for paint actions
        total_cost += len(unpainted_goal_tiles)

        # 4. & 5. Add cost for changing colors if needed
        held_colors = set()
        robot_locations = {}
        for fact in state:
            if match(fact, "robot-has", "*", "*"):
                _, robot, color = get_parts(fact)
                held_colors.add(color)
            elif match(fact, "robot-at", "*", "*"):
                 _, robot, location = get_parts(fact)
                 robot_locations[robot] = location

        colors_to_acquire = needed_colors - held_colors
        total_cost += len(colors_to_acquire)

        # 6. Estimate movement cost
        total_movement_cost = 0
        for tile, color in unpainted_goal_tiles:
            min_dist_to_adj = float('inf')

            # Find the minimum distance from any robot to any tile adjacent to the target tile
            for robot, robot_loc in robot_locations.items():
                if tile in self.adj_tiles: # Ensure the tile is in our grid graph
                    for adj_tile in self.adj_tiles[tile]:
                        dist = self.get_manhattan_distance(robot_loc, adj_tile)
                        min_dist_to_adj = min(min_dist_to_adj, dist)
                # If tile is not in adj_tiles (e.g., isolated tile?), assume infinite distance
                else:
                    min_dist_to_adj = float('inf')
                    break # No need to check other adjacent tiles for this isolated tile

            if min_dist_to_adj == float('inf'):
                 # This tile is unreachable by any robot, potentially a dead end
                 # A very high heuristic value might be appropriate, but let's
                 # stick to summing finite costs for reachable parts.
                 # If a tile is unreachable, the problem might be unsolvable,
                 # but the heuristic should still provide a value. Summing finite
                 # distances for reachable tiles is better than infinity unless
                 # we explicitly detect unsolvability. Let's assume reachable.
                 pass # If unreachable, min_dist_to_adj remains inf, adding inf to total_movement_cost

            total_movement_cost += min_dist_to_adj

        total_cost += total_movement_cost

        return total_cost

