from heuristics.heuristic_base import Heuristic

# Helper function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty string or malformed fact gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

# Domain-specific helper functions
def parse_tile_coords(tile_str):
    """Parses a tile string like 'tile_R_C' into integer coordinates (R, C)."""
    if not isinstance(tile_str, str):
        return None
    try:
        parts = tile_str.split('_')
        # Expecting format like 'tile_0_1'
        if len(parts) == 3 and parts[0] == 'tile':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            # Handle unexpected format - this might indicate a problem instance issue
            return None
    except ValueError:
        # Handle cases where R or C are not integers
        return None

def manhattan_distance(tile1_str, tile2_str):
    """Calculates the Manhattan distance between two tiles."""
    coords1 = parse_tile_coords(tile1_str)
    coords2 = parse_tile_coords(tile2_str)
    if coords1 is None or coords2 is None:
        # If parsing failed for either tile, distance is effectively infinite
        return float('inf')
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])

def is_adjacent(tile1_str, tile2_str):
    """Checks if two tiles are adjacent (Manhattan distance is 1)."""
    # Handle cases where parsing fails
    dist = manhattan_distance(tile1_str, tile2_str)
    if dist == float('inf'):
        return False
    return dist == 1

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost to paint all required tiles with the correct colors.
    For each unpainted goal tile, it estimates the minimum cost for any robot
    to reach the tile's vicinity, change color if needed, and paint it.
    Movement cost is approximated by Manhattan distance, ignoring the 'clear'
    constraint for path traversal but considering adjacency for the final paint action.
    Assumes solvable instances do not have goal tiles painted with the wrong color.
    Assumes unpainted goal tiles are clear.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Extract goal tiles and their required colors
        self.goal_tiles = {}
        for goal in self.goals:
            # Goal is typically (painted tile_X_Y color)
            parts = get_parts(goal)
            if parts and parts[0] == 'painted' and len(parts) == 3:
                tile = parts[1]
                color = parts[2]
                self.goal_tiles[tile] = color
            # Note: Other goal types like (robot-at ...) are possible but less common
            # in typical floortile problems focused on painting. We only consider 'painted' goals.

        # Extract available colors from static facts
        self.available_colors = set()
        for fact in self.static_facts:
             parts = get_parts(fact)
             if parts and parts[0] == 'available-color' and len(parts) == 2:
                 self.available_colors.add(parts[1])

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Find robot locations and colors
        robots_info = {}
        # Iterate through state facts to find robot info
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'robot-at' and len(parts) == 3:
                robot_name = parts[1]
                location = parts[2]
                if robot_name not in robots_info:
                    robots_info[robot_name] = {'location': location, 'color': None}
                else:
                     robots_info[robot_name]['location'] = location
            elif parts and parts[0] == 'robot-has' and len(parts) == 3:
                 robot_name = parts[1]
                 color = parts[2]
                 if robot_name not in robots_info:
                     robots_info[robot_name] = {'location': None, 'color': color}
                 else:
                      robots_info[robot_name]['color'] = color

        # If no robots found (shouldn't happen in valid instances), return large value
        if not robots_info:
             return 1000 # Or float('inf')

        total_cost = 0

        # Iterate through goal tiles that are not yet painted correctly
        for goal_tile, goal_color in self.goal_tiles.items():
            is_painted_correctly = False
            is_painted_wrong = False

            # Check current state of the goal tile
            # Check if it's painted with the correct color
            if f'(painted {goal_tile} {goal_color})' in state:
                 is_painted_correctly = True

            if is_painted_correctly:
                continue # This goal is already satisfied

            # If not painted correctly, check if it's painted with *any* other color
            # We need to check all available colors except the goal color
            for color in self.available_colors:
                if color != goal_color and f'(painted {goal_tile} {color})' in state:
                    is_painted_wrong = True
                    break # Tile is painted with a wrong color

            if is_painted_wrong:
                 # Goal tile is painted with the wrong color. Assuming unsolvable in this domain.
                 # Return a large value to prune this path.
                 return 1000 # Or float('inf')

            # If not painted correctly and not painted wrong, it must be clear
            # (based on domain structure and typical solvable instances).
            # This tile needs to be painted.

            min_robot_cost_for_tile = float('inf')

            # Calculate minimum cost for any robot to paint this tile
            for r_name, r_info in robots_info.items():
                r_loc = r_info.get('location') # Use .get for safety
                r_color = r_info.get('color')   # Use .get for safety

                # Ensure robot location and color info is complete (should be in valid states)
                if r_loc is None or r_color is None:
                    # This robot's info is incomplete, skip for this tile calculation
                    continue

                cost_this_robot = 0

                # Cost for changing color if needed (1 action)
                if r_color != goal_color:
                    cost_this_robot += 1

                # Cost for movement and painting (approximated by Manhattan distance)
                dist = manhattan_distance(r_loc, goal_tile)

                # If distance calculation failed (e.g., bad tile name), this robot can't reach
                if dist == float('inf'):
                    continue # Skip this robot for this tile calculation

                move_paint_cost = 0
                if r_loc == goal_tile:
                    # Robot is on the tile. Needs to move off (1) then paint (1).
                    move_paint_cost = 2
                elif is_adjacent(r_loc, goal_tile):
                    # Robot is adjacent. Needs 1 paint action.
                    move_paint_cost = 1
                else:
                    # Robot is not on or adjacent. Needs moves to get adjacent (dist-1) + paint (1).
                    # Total moves + paint = (dist - 1) + 1 = dist.
                    move_paint_cost = dist

                cost_this_robot += move_paint_cost

                min_robot_cost_for_tile = min(min_robot_cost_for_tile, cost_this_robot)

            # Add the minimum cost for this tile to the total
            if min_robot_cost_for_tile == float('inf'):
                 # This means no robot could potentially paint this tile (e.g., no robots,
                 # or parsing failed for all robots/tiles).
                 # In a solvable instance, this shouldn't happen, but return large value defensively.
                 return 1000 # Or float('inf')

            total_cost += min_robot_cost_for_tile

        return total_cost
