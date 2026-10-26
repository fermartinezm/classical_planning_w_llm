# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to parse tile names like 'tile_row_col'
def parse_tile_name(tile_name):
    """Parses a tile name string 'tile_row_col' into a (row, col) tuple of integers."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle unexpected format - should not happen in valid problems
            return None
    # Handle unexpected format - should not happen in valid problems
    return None

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost to paint all goal tiles by summing the minimum cost
    for each unpainted goal tile, considering the nearest robot with the
    correct color (or the nearest robot that can get the color).

    This heuristic is non-admissible and designed for greedy best-first search.
    It calculates the cost for each unpainted goal tile independently, taking
    the minimum cost over all robots. The cost for a single tile includes:
    1. Minimum Manhattan distance for a robot to reach an adjacent tile from
       which the target tile can be painted.
    2. Cost to change color if the robot doesn't have the required color (1 action).
    3. Cost of the paint action (1 action).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        # The base class Heuristic might need task, but its definition isn't provided.
        # Assuming it doesn't require explicit super().__init__(task) based on examples.
        # super().__init__(task)

        self.goals = task.goals
        self.static_facts = task.static

        # Store goal tile-color mapping
        self.goal_colors = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                self.goal_colors[tile] = color

        # Collect all tile names from static facts to validate neighbors
        # This assumes all relevant tiles appear in movement predicates.
        self.all_tiles = set()
        for fact in self.static_facts:
             parts = get_parts(fact)
             # Collect tiles from movement predicates (target and source)
             if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                 self.all_tiles.add(parts[1]) # target tile
                 self.all_tiles.add(parts[2]) # source tile
             # If tiles were listed in :objects, a full PDDL parser would get them.
             # Relying on static predicates is sufficient given the examples.

        # Store available colors
        self.available_colors = set()
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'available-color':
                self.available_colors.add(parts[1])


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Identify unpainted goal tiles
        unpainted_goal_tiles = set()
        for tile, color in self.goal_colors.items():
            # A tile needs painting if the goal requires (painted tile color)
            # and this fact is not true in the current state.
            # We assume that in solvable states, any such tile is currently clear.
            # If it were painted with the wrong color, the problem is likely unsolvable
            # in this domain (no unpaint/repaint). We still count it as needing painting.
            if f'(painted {tile} {color})' not in state:
                 unpainted_goal_tiles.add(tile)


        # If all goal tiles are painted correctly, heuristic is 0
        if not unpainted_goal_tiles:
            return 0

        # Get robot locations and colors from the current state
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot, location = parts[1], parts[2]
                robot_locations[robot] = location
            elif parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        total_heuristic = 0

        # Calculate cost for each unpainted goal tile independently
        for tile in unpainted_goal_tiles:
            required_color = self.goal_colors[tile]

            # Find potential painting locations (adjacent tiles)
            # A robot at tile X can paint tile Y if there's a movement predicate
            # (direction Y X). The set of such X tiles are the neighbors of Y.
            paint_locations = set()
            tile_coords = parse_tile_name(tile)

            if tile_coords is None:
                 # Skip this tile if its name cannot be parsed (unexpected format)
                 # This shouldn't happen for goal tiles in valid problems.
                 continue

            row, col = tile_coords
            # Potential neighbors based on grid structure (Manhattan neighbors)
            potential_neighbors = [
                f'tile_{row-1}_{col}', # Tile above (robot needs to be below to paint up)
                f'tile_{row+1}_{col}', # Tile below (robot needs to be above to paint down)
                f'tile_{row}_{col-1}', # Tile left (robot needs to be right to paint left)
                f'tile_{row}_{col+1}', # Tile right (robot needs to be left to paint right)
            ]

            # Add neighbors that exist in the problem to paint_locations
            # This check is important for tiles on the grid boundaries
            for neighbor_name in potential_neighbors:
                 if neighbor_name in self.all_tiles:
                      paint_locations.add(neighbor_name)

            # If a goal tile has no valid paint locations, it's likely an invalid problem.
            # For a heuristic, we could assign a large cost or skip. Skipping might underestimate.
            # Assuming valid problems, this set is not empty for goal tiles.
            if not paint_locations:
                 # Assign a large penalty? Or assume unsolvable?
                 # For a greedy heuristic, maybe just continue, effectively ignoring this tile?
                 # Let's assume valid problems don't have required tiles that cannot be painted.
                 continue


            min_cost_for_tile = float('inf')

            # Find the minimum cost for any robot to paint this tile
            for robot, robot_loc in robot_locations.items():
                # Get robot's current color. Assume robots always have a color initially.
                # The domain doesn't use 'free-color' in actions.
                robot_color = robot_colors.get(robot)

                # Cost to get the required color
                # Assumes a robot can change color if the required color is available.
                # The change_color action requires the robot to have *some* color to change from.
                # We assume robot_color is never None in valid states.
                color_cost = 0
                if robot_color != required_color:
                    # We need to change color. This costs 1 action.
                    # Precondition: robot-has ?c (current color), available-color ?c2 (required color).
                    # We assume robot_color is a valid color and required_color is available.
                    color_cost = 1


                # Cost to move from robot_loc to the nearest paint_loc using Manhattan distance
                min_move_cost = float('inf')
                robot_coords = parse_tile_name(robot_loc)

                if robot_coords is None:
                     # Skip this robot if its location name cannot be parsed
                     continue

                r_row, r_col = robot_coords

                for paint_loc in paint_locations:
                    paint_loc_coords = parse_tile_name(paint_loc)
                    if paint_loc_coords is None:
                         continue # Skip this paint location if name cannot be parsed

                    pl_row, pl_col = paint_loc_coords
                    dist = abs(r_row - pl_row) + abs(r_col - pl_col)
                    min_move_cost = min(min_move_cost, dist)

                # If min_move_cost is still infinity, it means no valid paint locations were found
                # or robot location could not be parsed. This shouldn't happen in valid problems.
                if min_move_cost == float('inf'):
                     # Skip this robot for this tile if movement cost cannot be calculated
                     continue

                # Total cost for this robot to paint this tile
                # min_move_cost (moves to adjacent tile) + color_cost (change color) + 1 (paint action)
                robot_cost = min_move_cost + color_cost + 1

                min_cost_for_tile = min(min_cost_for_tile, robot_cost)

            # Add the minimum cost found across all robots for this tile
            # If min_cost_for_tile is still infinity, it means no robot could paint it
            # (e.g., no robots in state, or parsing failed for all robots/locations).
            # This indicates an issue with the state/problem.
            # For a heuristic, returning a large value is an option.
            # Let's assume valid states have at least one robot and valid tile names.
            if min_cost_for_tile != float('inf'):
                 total_heuristic += min_cost_for_tile
            # else: problem might be unsolvable or state is malformed, heuristic could be infinity

        return total_heuristic
