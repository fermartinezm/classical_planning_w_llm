from fnmatch import fnmatch
# Assuming Heuristic base class is available in a specific path, e.g.:
# from heuristics.heuristic_base import Heuristic

# Helper functions adapted from provided examples
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(predicate arg1 arg2)" -> ["predicate", "arg1", "arg2"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj1 loc1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the fact has at least as many parts as the pattern args for meaningful comparison
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Domain-specific helper to parse tile coordinates
def get_coords(tile_name):
    """Parses a tile name like 'tile_R_C' into (R, C) integer coordinates."""
    # Example: "tile_0_1" -> ["tile", "0", "1"]
    parts = tile_name.split('_')
    if len(parts) != 3 or parts[0] != 'tile':
         # This format is expected based on problem examples.
         # If it deviates, it's an unexpected input for this domain.
         # Return None to indicate parsing failure.
         return None
    try:
        return (int(parts[1]), int(parts[2]))
    except ValueError:
        # Handle cases where R or C are not valid integers
        return None


# Define the heuristic class
# Inherit from Heuristic base class if available in the environment
# class floortileHeuristic(Heuristic):
class floortileHeuristic: # Use this line if Heuristic base class is not provided directly
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost to paint all required tiles by summing:
    1. Cost for acquiring necessary colors (1 per color if no robot has it).
    2. Cost for painting each unpainted goal tile (1 action per tile).
    3. Cost for moving the closest robot to a position adjacent to each tile
       from which it can be painted (Manhattan distance).

    This heuristic is non-admissible and designed for greedy best-first search.
    It simplifies movement cost by using Manhattan distance and ignoring dynamic
    obstacles (other robots, painted tiles blocking paths).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and grid structure
        from the task's static facts.
        """
        # Store the task object if the base class expects it, or just extract needed info
        # self.task = task # Uncomment if base class needs this
        self.goals = task.goals
        static_facts = task.static

        self.tile_coords = {} # Map tile name (string) to (row, col) (tuple of ints)
        # Map tile name (to be painted, string) to list of tile names (from which it can be painted, list of strings)
        self.paint_from_tiles = {}
        tile_names = set()

        # Parse static facts to build grid structure and paint-from relationships
        # Adjacency facts (up, down, left, right) define the tiles present in the problem
        for fact in static_facts:
            parts = get_parts(fact)
            # Adjacency facts have 3 parts: predicate (direction), tile_y, tile_x
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                # Fact is typically (direction tile_y tile_x)
                # PDDL action paint_X takes parameters (?r - robot ?y - tile ?x - tile ?c - color)
                # Precondition includes (robot-at ?r ?x) and (direction ?y ?x)
                # This means robot is at ?x and paints ?y. So ?x is the 'paint-from' tile for ?y.
                direction, tile_y, tile_x = parts[0], parts[1], parts[2]
                tile_names.add(tile_y)
                tile_names.add(tile_x)
                # If robot is at tile_x, it can paint tile_y
                self.paint_from_tiles.setdefault(tile_y, []).append(tile_x)

        # Parse coordinates for all identified tiles
        for tile_name in tile_names:
            coords = get_coords(tile_name)
            if coords is not None:
                self.tile_coords[tile_name] = coords
            # else: tile_name is invalid, skip it. Any goal or robot tile
            # that is invalid will result in a large heuristic value later.


        # Store goal requirements: {tile_name: color_name}
        self.goal_paintings = {} # Map tile name (string) to required color name (string)
        for goal in self.goals:
            # Goal facts are typically "(painted tile_name color_name)"
            if match(goal, "painted", "?tile", "?color"):
                parts = get_parts(goal)
                if len(parts) == 3:
                    tile_name = parts[1]
                    color_name = parts[2]
                    self.goal_paintings[tile_name] = color_name
            # Add other goal types if necessary, but floortile usually only has painted goals

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.

        The heuristic value is the sum of estimated costs for each unpainted
        goal tile that is currently clear.
        """
        state = node.state # Current world state (frozenset of fact strings).

        # If the goal is already satisfied in the current state, the heuristic is 0.
        if self.goals <= state:
            return 0

        # Identify current robot locations and colors
        robot_locations = {} # {robot_name: tile_name}
        robot_colors = {} # {robot_name: color_name}
        for fact in state:
            if match(fact, "robot-at", "?robot", "?tile"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    robot_name = parts[1]
                    tile_name = parts[2]
                    robot_locations[robot_name] = tile_name
            elif match(fact, "robot-has", "?robot", "?color"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    robot_name = parts[1]
                    color_name = parts[2]
                    robot_colors[robot_name] = color_name

        # Identify unpainted goal tiles that are currently clear.
        # These are the tiles that *need* painting in this state.
        unpainted_goal_tiles = [] # List of (tile_name, color_name) tuples
        for tile_name, required_color in self.goal_paintings.items():
            # A tile needs painting if it's a goal tile and it's currently clear.
            # If it's painted with the wrong color, it cannot be fixed in this domain,
            # implying unsolvability for that goal. We assume solvable instances
            # don't have goal tiles painted incorrectly that are part of the goal.
            if f"(clear {tile_name})" in state:
                 unpainted_goal_tiles.append((tile_name, required_color))
            # If it's already painted with the correct color, it satisfies the goal
            # for this tile and doesn't contribute to the heuristic.

        # If there are no unpainted goal tiles that are clear, but the goal is not met,
        # it implies some goal tiles are painted incorrectly. This state is likely unsolvable.
        # Return a large value to guide the search away from such states.
        if not unpainted_goal_tiles and not (self.goals <= state):
             # Use a large constant instead of float('inf') if required by the search implementation
             return 1000000

        total_cost = 0

        # 1. Estimate cost for acquiring necessary colors.
        # Add 1 for each color required by at least one unpainted goal tile,
        # if no robot currently possesses that color.
        colors_needed = {color for _, color in unpainted_goal_tiles}
        colors_accounted = set() # Track colors for which we've added the change_color cost
        for color in colors_needed:
            # Check if any robot currently has this color
            has_color = any(c == color for c in robot_colors.values())
            if not has_color and color not in colors_accounted:
                # One change_color action is needed for this color by *some* robot.
                # This is a simplification; multiple robots might need the same color sequentially.
                total_cost += 1
                colors_accounted.add(color)

        # 2. + 3. Estimate cost for painting and movement for each unpainted goal tile.
        # For each tile that needs painting:
        #   - Add 1 for the paint action itself.
        #   - Add the minimum Manhattan distance from any robot to any valid position
        #     from which this tile can be painted.
        for tile_name, required_color in unpainted_goal_tiles:
            # Cost for the paint action itself
            total_cost += 1

            min_dist_to_paint_pos = float('inf')
            tile_coords = self.tile_coords.get(tile_name) # Coordinates of the tile to be painted (R_T, C_T)

            # Ensure the tile exists and its coordinates were parsed
            if tile_coords is None:
                 # This tile from the goal was not found in static adjacency facts or parsing failed.
                 # Should not happen in valid problems. Treat as unreachable for this tile.
                 min_dist_to_paint_pos = float('inf')
            else:
                # Get the list of tiles from which tile_name can be painted
                paint_from_tiles = self.paint_from_tiles.get(tile_name, [])

                # Ensure there are valid positions from which this tile can be painted
                if not paint_from_tiles:
                     # This tile cannot be painted from anywhere based on static facts.
                     # Problem likely unsolvable regarding this tile.
                     min_dist_to_paint_pos = float('inf')
                else:
                    # Find the minimum distance from any robot to any valid paint-from tile
                    # If there are no robots, min_dist_to_paint_pos remains inf
                    if not robot_locations:
                         min_dist_to_paint_pos = float('inf')
                    else:
                        for robot_name, robot_tile in robot_locations.items():
                            robot_coords = self.tile_coords.get(robot_tile) # Coordinates of the robot (R_R, C_R)

                            # Ensure the robot's current tile exists and its coordinates were parsed
                            if robot_coords is None:
                                 # Robot is on a tile not found in static adjacency facts or parsing failed.
                                 # Should not happen in valid problems. Skip this robot for this tile.
                                 continue

                            for paint_from_tile in paint_from_tiles:
                                paint_from_coords = self.tile_coords.get(paint_from_tile) # Coordinates of the paint-from tile (R_adj, C_adj)

                                # Ensure the paint-from tile exists and its coordinates were parsed
                                if paint_from_coords is None:
                                     # Paint-from tile not found in static adjacency facts or parsing failed.
                                     # Should not happen in valid problems. Skip this paint-from position.
                                     continue

                                # Calculate Manhattan distance from robot's current position to the paint-from position
                                dist = abs(robot_coords[0] - paint_from_coords[0]) + abs(robot_coords[1] - paint_from_coords[1])
                                min_dist_to_paint_pos = min(min_dist_to_paint_pos, dist)

            # Add the minimum movement cost for this tile.
            # If min_dist_to_paint_pos is still infinity after checking all robots and paint positions,
            # it means this tile is unreachable by any robot. This implies unsolvability for this tile.
            # Add this cost. If it's infinity, the total cost becomes infinity.
            total_cost += min_dist_to_paint_pos

        # If total_cost became infinity due to an unreachable tile, return a large number.
        # Otherwise, return the calculated finite cost.
        # Use a large constant instead of float('inf') if required by the search implementation
        return total_cost if total_cost != float('inf') else 1000000
