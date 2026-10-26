from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

# Helper functions (can be defined outside the class)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle cases like "(predicate)" or "(predicate arg)"
    if fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    return [] # Should not happen with valid PDDL facts

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
    """Parses a tile name like 'tile_X_Y' into coordinates (X, Y)."""
    match = re.match(r'tile_(\d+)_(\d+)', tile_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None # Should not happen with valid tile names

def manhattan_distance(tile1_name, tile2_name):
    """Calculates the Manhattan distance between two tiles."""
    coords1 = parse_tile_name(tile1_name)
    coords2 = parse_tile_name(tile2_name)
    if coords1 is None or coords2 is None:
        # This indicates an issue with tile naming or parsing
        # For heuristic purposes, treat as unreachable
        return float('inf')
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their target colors. It sums the cost for each unpainted goal tile,
    considering the paint action itself, the minimum movement cost for any robot
    to get adjacent to the tile, and the color change cost if that robot
    doesn't have the required color.

    # Assumptions
    - Goal tiles are initially clear or become clear before needing to be painted.
      (The domain only allows painting clear tiles).
    - Robots always have a color.
    - Tile names follow the format 'tile_X_Y' allowing coordinate extraction.
    - The grid connectivity defined by up/down/left/right predicates corresponds
      to a grid structure where Manhattan distance is a reasonable movement cost estimate.
    - The cost of moving between adjacent tiles is 1.
    - The cost of changing color is 1.
    - The cost of painting a tile is 1.
    - All tiles mentioned in static facts (up/down/left/right) or goals exist and
      have valid 'tile_X_Y' names.

    # Heuristic Initialization
    - Extracts the goal conditions, specifically the required color for each goal tile.
    - Builds an adjacency map for tiles based on up/down/left/right static facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all goal facts of the form `(painted tile_X_Y color)`. Store these as
       a mapping from tile name to required color.
    2. Identify all facts in the current state.
    3. Determine the set of unsatisfied goal facts: `(painted T C)` is unsatisfied
       if it is a goal fact but not present in the current state.
    4. If there are no unsatisfied goal facts, the heuristic is 0 (goal state).
    5. If there are unsatisfied goal facts, initialize the total heuristic cost `h = 0`.
    6. For each unsatisfied goal fact `(painted T C)`:
       a. Add 1 to `h` for the paint action itself.
       b. Find the current location and color of each robot in the state.
       c. Calculate the minimum cost for *any* robot to be ready to paint tile `T` with color `C`.
          Initialize `min_ready_cost_for_tile = infinity`.
          For each robot `R` at location `R_loc` with color `C_R`:
          i. Calculate the cost to change color: `color_change_cost = 1` if `C_R != C`, else `0`.
          ii. Calculate the minimum cost to move from `R_loc` to *any* tile `T'` adjacent to `T`.
              This `move_cost` is `min_{T' adjacent to T} Manhattan(R_loc, T')`.
              Iterate through all tiles `T'` adjacent to `T` (using the pre-computed adjacency map)
              and find the minimum Manhattan distance from `R_loc` to `T'`.
          iii. The total cost for robot `R` to be ready is `ready_cost = color_change_cost + move_cost`.
          iv. Update `min_ready_cost_for_tile = min(min_ready_cost_for_tile, ready_cost)`.
       d. If `min_ready_cost_for_tile` is still infinity (e.g., no robots found or tile has no neighbors),
          the problem might be unsolvable from this state. Return infinity.
       e. Add `min_ready_cost_for_tile` to `h`.
    7. The total heuristic value is the sum accumulated in step 6.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building adjacency map."""
        self.goals = task.goals  # Goal conditions as a frozenset of strings.
        self.static_facts = task.static # Static facts

        # Extract required painted states from goals
        self.goal_painted_tiles = {} # {tile_name: required_color}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile_name = parts[1]
                color = parts[2]
                self.goal_painted_tiles[tile_name] = color

        # Build adjacency map from static facts
        self.adjacency = {} # {tile_name: {neighbor_tile_name, ...}}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and len(parts) == 3:
                pred, t1, t2 = parts
                if pred in ["up", "down", "left", "right"]:
                    if t1 not in self.adjacency:
                        self.adjacency[t1] = set()
                    if t2 not in self.adjacency:
                        self.adjacency[t2] = set()
                    # Predicate (up t1 t2) means t1 is up from t2. So t1 and t2 are adjacent.
                    self.adjacency[t1].add(t2)
                    self.adjacency[t2].add(t1)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state as a frozenset of strings.

        # Check if the goal is already satisfied
        # A state is a goal state if all goal_painted_tiles are present in the state
        is_goal_state = True
        for tile, color in self.goal_painted_tiles.items():
            if f"(painted {tile} {color})" not in state:
                is_goal_state = False
                break

        if is_goal_state:
            return 0

        # Extract robot locations and colors from the current state
        robot_info = {} # {robot_name: {'location': tile_name, 'color': color_name}}
        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "robot-at" and len(parts) == 3:
                    robot, location = parts[1], parts[2]
                    if robot not in robot_info:
                        robot_info[robot] = {}
                    robot_info[robot]['location'] = location
                elif parts[0] == "robot-has" and len(parts) == 3:
                    robot, color = parts[1], parts[2]
                    if robot not in robot_info:
                        robot_info[robot] = {}
                    robot_info[robot]['color'] = color

        # If there are no robots but goals require painting, it's unsolvable
        if not robot_info and self.goal_painted_tiles:
             # Check if any goal tile is NOT already painted correctly
             needs_painting = False
             for tile, color in self.goal_painted_tiles.items():
                 if f"(painted {tile} {color})" not in state:
                     needs_painting = True
                     break
             if needs_painting:
                 return float('inf')


        total_cost = 0

        # Identify unsatisfied goal tiles
        unsatisfied_goals = [] # List of (tile_name, required_color)
        for tile, required_color in self.goal_painted_tiles.items():
            # Check if the tile is painted with the correct color in the current state
            is_painted_correctly = f"(painted {tile} {required_color})" in state

            if not is_painted_correctly:
                 # This tile needs to be painted with required_color
                 unsatisfied_goals.append((tile, required_color))

        # Calculate cost for each unsatisfied goal tile
        for tile_to_paint, required_color in unsatisfied_goals:
            # Cost for the paint action itself
            cost_for_tile = 1

            min_ready_cost_for_tile = float('inf')

            # Find the minimum cost for any robot to get ready to paint this tile
            for robot_name, info in robot_info.items():
                robot_location = info.get('location')
                robot_color = info.get('color')

                # Should not happen if robot_info is populated correctly from state
                if robot_location is None or robot_color is None:
                    continue

                # Cost to change color if needed
                color_change_cost = 1 if robot_color != required_color else 0

                # Cost to move adjacent to the tile
                min_dist_to_adjacent = float('inf')
                adjacent_tiles = self.adjacency.get(tile_to_paint, set())

                if not adjacent_tiles:
                    # This tile has no adjacent tiles defined in static facts.
                    # It cannot be painted by paint_up/down/left/right actions.
                    # This indicates an unsolvable problem if this tile needs painting.
                    # Return infinity.
                    return float('inf')

                for adjacent_tile in adjacent_tiles:
                    dist = manhattan_distance(robot_location, adjacent_tile)
                    min_dist_to_adjacent = min(min_dist_to_adjacent, dist)

                # The cost is the minimum number of moves to reach one of these adjacent tiles
                move_cost = min_dist_to_adjacent

                # Total cost for this robot to be ready for this tile
                ready_cost = color_change_cost + move_cost

                min_ready_cost_for_tile = min(min_ready_cost_for_tile, ready_cost)

            # Add the minimum ready cost (color change + move) for this tile
            # If min_ready_cost_for_tile is still inf, it means no robots could reach it
            # (e.g., if robot_info was empty, handled above, or if tile has no neighbors, handled above)
            # If it is reached, it implies robots exist and the tile has neighbors, but none are reachable.
            # This would indicate a disconnected grid or other unsolvable state.
            if min_ready_cost_for_tile == float('inf'):
                 return float('inf')


            total_cost += cost_for_tile + min_ready_cost_for_tile

        return total_cost
