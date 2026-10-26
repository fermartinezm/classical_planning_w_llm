from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # For float('inf')

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential non-string input gracefully
    if not isinstance(fact, str):
        return []
    # Find the first and last parentheses
    l_paren = fact.find('(')
    r_paren = fact.rfind(')')
    if l_paren == -1 or r_paren == -1 or l_paren >= r_paren:
        return [] # Invalid format

    # Extract content between parentheses and split
    content = fact[l_paren + 1 : r_paren].strip()
    if not content:
        return []
    return content.split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args for a valid match
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are not yet correctly painted. It sums the cost of painting each required
    tile, the cost of acquiring necessary colors, and an estimated movement cost.

    # Assumptions
    - Tiles are arranged in a grid, and their names follow the format 'tile_R_C'
      where R is the row and C is the column (0-indexed or 1-indexed, consistent).
      The heuristic assumes R and C can be parsed as integers.
    - Movement between adjacent tiles costs 1 action.
    - Painting a tile costs 1 action.
    - Changing color costs 1 action.
    - A tile can only be painted if it is 'clear'. The heuristic assumes that
      if a goal tile is currently painted with the *wrong* color, the problem
      is unsolvable in this domain (as there's no unpaint/repaint action).
      If a goal tile is 'clear' or correctly painted, it is considered solvable
      with respect to its initial state. The heuristic ignores the 'clear'
      precondition for movement cost estimation, treating the grid as fully traversable
      in terms of distance, but checks the 'painted' state for unsolvability.

    # Heuristic Initialization
    - Extracts the required color for each goal tile from the task's goal conditions.
    - Sets up helper methods for parsing tile coordinates and calculating Manhattan distance.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal tiles that are not currently painted with the correct color.
       These are the 'unmet goal tiles'.
    2. Check for unsolvability: If any unmet goal tile is currently painted with *any*
       color (that is not the correct goal color), the problem is considered unsolvable
       and the heuristic returns infinity.
    3. If there are no unmet goal tiles, the state is a goal state, and the heuristic is 0.
    4. Initialize the heuristic value `h` to 0.
    5. Calculate Color Acquisition Cost:
       - Determine the set of colors required by the unmet goal tiles.
       - Determine the set of colors currently held by any robot.
       - Count the number of required colors that are *not* held by any robot. Add this count to `h`.
       - This assumes one robot acquiring a color is sufficient, even if multiple robots need it.
    6. Calculate Painting Cost:
       - Add the total number of unmet goal tiles to `h`. Each tile needs one paint action.
    7. Calculate Movement Cost:
       - For each robot, find its current location (tile).
       - For each unmet goal tile:
         - Calculate the minimum number of moves required for *any* robot to reach a tile adjacent to this unmet goal tile. This is estimated as `max(0, Manhattan distance - 1)`.
         - Add this minimum movement cost for the current unmet tile to `h`.
       - This step sums the minimum movement cost for each tile independently, which is an overestimation but provides a gradient towards states where robots are closer to unpainted tiles.
    8. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals  # Goal conditions.

        # Store goal locations and required colors for each tile.
        self.goal_paintings = {}
        # Goals can be a single predicate or a conjunction (and ...)
        if isinstance(self.goals, str) and match(self.goals, "painted", "*", "*"):
             _, tile, color = get_parts(self.goals)
             self.goal_paintings[tile] = color
        elif isinstance(self.goals, list) and self.goals[0] == 'and':
             for sub_goal in self.goals[1:]:
                 if match(sub_goal, "painted", "*", "*"):
                     _, tile, color = get_parts(sub_goal)
                     self.goal_paintings[tile] = color
        # Handle other potential goal formats if necessary, but assuming painted predicates are the main goals.


    def get_tile_coords(self, tile_name):
        """Parses tile name 'tile_R_C' into (R, C) integer coordinates."""
        try:
            parts = tile_name.split('_')
            # Expecting format like 'tile_0_1'
            if len(parts) == 3 and parts[0] == 'tile':
                # Assuming R is parts[1] and C is parts[2]
                return (int(parts[1]), int(parts[2]))
            else:
                # print(f"Warning: Unexpected tile name format: {tile_name}")
                return None
        except ValueError:
            # print(f"Warning: Could not parse integer coordinates from tile name: {tile_name}")
            return None

    def manhattan_distance(self, coords1, coords2):
        """Calculates Manhattan distance between two (R, C) coordinate tuples."""
        if coords1 is None or coords2 is None:
            return float('inf') # Indicate inability to calculate distance
        return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings).

        # 1. Identify unmet goal tiles and check for unsolvability
        unmet_goals = {} # {tile_name: goal_color}
        for tile, goal_color in self.goal_paintings.items():
            is_correctly_painted = False
            is_wrongly_painted = False

            # Check current state for the tile's painted status
            for fact in state:
                if fact.startswith('(painted '):
                    parts = get_parts(fact)
                    if len(parts) == 3:
                        _, painted_tile, painted_color = parts
                        if painted_tile == tile:
                            if painted_color == goal_color:
                                is_correctly_painted = True
                            else:
                                is_wrongly_painted = True
                            break # Found the painted status for this tile

            if not is_correctly_painted:
                 # If it's not correctly painted, it's an unmet goal
                 unmet_goals[tile] = goal_color
                 # If it's wrongly painted, the problem is unsolvable
                 if is_wrongly_painted:
                     return float('inf')


        # 3. If no unmet goals, return 0
        if not unmet_goals:
            return 0

        h = 0

        # 4. Calculate Color Acquisition Cost
        needed_colors = set(unmet_goals.values())
        held_colors = set()
        for fact in state:
             if fact.startswith('(robot-has '):
                 parts = get_parts(fact)
                 if len(parts) == 3:
                     # fact is like '(robot-has robot1 white)'
                     held_colors.add(parts[2]) # Add the color

        colors_to_acquire = needed_colors - held_colors
        h += len(colors_to_acquire)

        # 5. Calculate Painting Cost
        h += len(unmet_goals)

        # 6. Calculate Movement Cost
        robot_locations = {} # {robot_name: tile_name}
        for fact in state:
             if fact.startswith('(robot-at '):
                 parts = get_parts(fact)
                 if len(parts) == 3:
                     # fact is like '(robot-at robot1 tile_0_4)'
                     robot_locations[parts[1]] = parts[2] # Map robot name to tile name


        for tile, goal_color in unmet_goals.items():
            target_coords = self.get_tile_coords(tile)
            if target_coords is None:
                 # This tile has an invalid name format, skip or handle error
                 continue

            min_move_cost_for_tile = float('inf')

            # If there are no robots, movement is impossible (shouldn't happen in valid problems)
            if not robot_locations:
                 return float('inf')

            for robot, loc_r in robot_locations.items():
                robot_coords = self.get_tile_coords(loc_r)
                if robot_coords is None:
                    # Robot is at a location with invalid name format, skip or handle error
                    continue

                dist = self.manhattan_distance(robot_coords, target_coords)
                # Minimum moves to get adjacent is distance - 1, clamped at 0
                move_cost = max(0, dist - 1)
                min_move_cost_for_tile = min(min_move_cost_for_tile, move_cost)

            # Add the minimum movement cost required for *some* robot to reach this tile
            # Only add if a valid movement cost was found (i.e., not still inf)
            if min_move_cost_for_tile != float('inf'):
                 h += min_move_cost_for_tile
            # else: This tile might be unreachable if all robot locations are invalid?
            # Assuming valid instance files, this else should not be hit for reachable tiles.


        # 8. Return the total heuristic value
        return h
