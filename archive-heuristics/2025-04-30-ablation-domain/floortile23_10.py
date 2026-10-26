from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if pattern is longer than fact parts
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Domain-specific helper function for tile coordinates
def get_tile_coords(tile_name):
    """Parses tile name 'tile_x_y' and returns (x, y) tuple."""
    match = re.match(r'tile_(\d+)_(\d+)', tile_name)
    if match:
        # Convert to integers. PDDL rows/cols are often 0-indexed or 1-indexed.
        # The Manhattan distance formula works regardless of indexing base,
        # as long as it's consistent. We assume 0-indexed for parsing.
        return (int(match.group(1)), int(match.group(2)))
    return None # Should not happen in valid floortile instances

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles."""
    coords1 = get_tile_coords(tile1_name)
    coords2 = get_tile_coords(tile2_name)
    if coords1 is None or coords2 is None:
        # This indicates an issue with tile naming or parsing
        # For heuristic purposes, treat as unreachable or very far
        return float('inf')
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing up the estimated costs for each tile that is not yet painted with its required goal color. The estimated cost for a single tile considers the paint action, the need to pick up the correct color, and the movement cost from the closest robot.

    # Assumptions
    - Tiles are arranged in a grid, and movement cost between adjacent tiles is 1. Manhattan distance is used as a lower bound for movement.
    - Painting a tile requires a robot to be at the tile with the correct color.
    - Picking a color costs 1 action and can be done by a robot if the color is available (which is a static fact). A robot can only hold one color at a time, and painting consumes the color. Thus, a robot might need to re-pick a color if it changes color or paints multiple tiles of the same color (this heuristic simplifies by assuming a pick is needed if the robot doesn't currently hold the required color).
    - Tiles that need painting are assumed to be 'clear' or can be painted over (the latter is less likely based on typical PDDL domains, but the heuristic doesn't strictly enforce 'clear' precondition, it just counts the 'painted' goal not being met). We assume tiles painted with the wrong color in the initial state are handled by the 'painted' goal not being met check.

    # Heuristic Initialization
    - Extract the goal conditions, specifically which tiles need to be painted with which colors.
    - Static facts like `available-color` are noted but don't require complex structures for this heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Unpainted Goal Tiles:
       Iterate through the goal conditions. For each goal `(painted T C)`, check if the current state contains `(painted T C)`. Collect all pairs `(T, C)` for which the goal is not met.

    2. Identify Robot States:
       Find the current location `LocR` and held color `ColorR` for each robot `R` present in the state facts (`robot-at` or `robot-has`).

    3. Calculate Individual Tile Costs:
       Initialize total heuristic cost to 0.
       For each unpainted goal tile `(T, C)` identified in step 1:
         a. Initialize minimum cost for this tile to infinity.
         b. For each robot `R` found in step 2:
            i. Get robot's current location `LocR` and color `ColorR`.
            ii. Calculate movement cost: Manhattan distance from `LocR` to `T`.
            iii. Calculate color cost: 1 if `ColorR` is not `C`, else 0. This assumes a robot needs one 'pick-color' action if it doesn't currently hold the correct color.
            iv. Calculate total cost for this robot for this tile: movement cost + color cost + 1 (for the paint action itself).
            v. Update minimum cost for tile `T` if this robot's cost is lower.
         c. Add the minimum cost found for tile `T` to the total heuristic cost. If no robots were found or could reach the tile, the minimum cost remains infinity, indicating an unreachable goal tile from this state (heuristic remains large).

    4. Return Total Heuristic Cost:
       The accumulated total cost is the heuristic value for the state.
       If there are no unpainted goal tiles, the state is a goal state, and the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals  # Goal conditions.
        # Extract goal painting requirements: {tile_name: color_name}
        self.goal_paintings = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_paintings[tile] = color

        # Static facts are available in task.static if needed,
        # but this heuristic doesn't require pre-processing them
        # into complex structures beyond what's implicitly used
        # by manhattan_distance (tile naming convention).
        # self.static_facts = task.static

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
             return 0

        # Identify unpainted goal tiles
        unpainted_goal_tiles = {} # {tile_name: required_color}
        for tile, color in self.goal_paintings.items():
            goal_fact = f"(painted {tile} {color})"
            if goal_fact not in state:
                # If the goal fact (painted T C) is not in the state,
                # this tile needs to be painted with color C.
                unpainted_goal_tiles[tile] = color

        # If all goal tiles are painted correctly, it's a goal state (should be caught by first check)
        if not unpainted_goal_tiles:
            return 0

        # Identify robot states
        robot_states = {} # {robot_name: {'location': tile_name, 'color': color_name or None}}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot, location = parts[1], parts[2]
                if robot not in robot_states:
                    robot_states[robot] = {'location': None, 'color': None}
                robot_states[robot]['location'] = location
            elif parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                if robot not in robot_states:
                    robot_states[robot] = {'location': None, 'color': None}
                robot_states[robot]['color'] = color

        total_heuristic_cost = 0

        # Calculate cost for each unpainted goal tile
        for tile_to_paint, required_color in unpainted_goal_tiles.items():
            min_tile_cost = float('inf')

            # Find the best robot for this tile
            for robot_name, robot_info in robot_states.items():
                robot_location = robot_info.get('location') # Use .get for safety
                robot_color = robot_info.get('color') # Use .get for safety

                if robot_location is None:
                    # Robot location unknown, cannot calculate distance. Skip this robot for this tile.
                    continue

                # Cost to move to the tile
                move_cost = manhattan_distance(robot_location, tile_to_paint)

                # Cost to get the required color
                color_cost = 0
                if robot_color != required_color:
                    color_cost = 1 # Cost of one pick-color action

                # Total cost for this robot to paint this tile
                # move + pick (if needed) + paint
                current_robot_tile_cost = move_cost + color_cost + 1

                min_tile_cost = min(min_tile_cost, current_robot_tile_cost)

            # Add the minimum cost found for tile `T` to the total heuristic
            # If min_tile_cost is still infinity, it means no robot could reach it
            # (e.g., no robots in state, or robots are at unparseable locations).
            # This state might be unsolvable or very far. Adding infinity keeps it
            # correctly ordered in a greedy search (will be explored last).
            if min_tile_cost == float('inf'):
                 # If even one unpainted tile is unreachable by any robot,
                 # the whole state might be considered infinitely far.
                 # Summing infinities is okay in Python.
                 total_heuristic_cost += float('inf')
            else:
                 total_heuristic_cost += min_tile_cost


        return total_heuristic_cost
