from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper functions specific to the Floortile domain grid structure
def parse_tile_name(tile_name):
    """
    Parses tile name 'tile_r_c' into (row, col) tuple.
    Returns None if the name format is unexpected.
    """
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            return (int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        # Handle unexpected formats if necessary
        return None
    return None # Should not be reached if format is correct

def manhattan_distance(tile1_name, tile2_name):
    """
    Calculates Manhattan distance between two tiles based on their names 'tile_r_c'.
    Returns float('inf') if tile names cannot be parsed.
    """
    coords1 = parse_tile_name(tile1_name)
    coords2 = parse_tile_name(tile2_name)
    if coords1 is None or coords2 is None:
        # Cannot calculate distance for invalid tile names
        return float('inf')
    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct color. It sums the estimated cost for each unsatisfied goal tile,
    considering the minimum cost for any robot to achieve that specific goal tile.
    The cost for a single tile includes movement, cleaning (if necessary),
    getting the correct color, and painting.

    # Assumptions
    - The grid structure allows movement between adjacent tiles (up, down, left, right)
      with a cost of 1 per step. Manhattan distance is used as a lower bound for movement.
    - A tile painted with the wrong color must be "cleaned" before being painted correctly (cost 1).
      This assumes a 'clean' action exists or is implicitly part of the process.
    - Getting the correct color involves either picking it up (cost 1, if robot has no color)
      or painting a dummy tile with the current wrong color and then picking up the correct color (cost 2, if robot has wrong color).
      This assumes a clear tile is available somewhere for dummy painting and that colors are available for pickup.
    - Each paint action costs 1.
    - Multiple robots can work in parallel, but the heuristic sums costs independently for each goal tile,
      picking the best robot for that tile in isolation. This ignores potential conflicts or synergies
      between robots working on different tiles.
    - Tiles not explicitly listed as 'painted' or 'clear' in the state are treated as implicitly clear
      if they are not painted with the goal color. If they are painted with a wrong color, they need cleaning.

    # Heuristic Initialization
    - Extracts the target color for each goal tile from the task's goal conditions.
    - Identifies all robot and color objects defined in the task by looking for type predicates
      like `(robot robot1)` in the task's facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all goal tiles and their required colors from the task definition (`self.goal_tiles`).
    2. For the current state, determine the location of each robot (`robot_locations`),
       the color each robot is holding (`robot_colors`),
       and the current status of each tile (painted with which color, or clear) (`tile_status`).
    3. Initialize the total heuristic cost (`total_cost`) to 0 and a flag `all_goals_satisfied` to True.
    4. Iterate through each goal tile (`goal_tile`) and its required color (`goal_color`) in `self.goal_tiles`.
    5. For each goal tile:
       a. Check if the tile is already painted with the `goal_color` in the current state. If yes, continue to the next goal tile.
       b. If the goal is not satisfied, set `all_goals_satisfied` to False.
       c. Calculate the minimum estimated cost for *any* robot to paint this `goal_tile` with `goal_color`.
       d. Initialize the minimum cost for this specific tile (`min_cost_for_tile`) to infinity.
       e. Determine the cost to clear the tile (`cost_clear`): 1 if the tile is currently painted with a color different from `goal_color`, 0 otherwise (assuming it's clear or implicitly clear if not painted).
       f. The cost to paint is 1 (`cost_paint`).
       g. Iterate through each robot (`robot_name`) identified during initialization.
          i. Get the robot's current location (`robot_loc`) and the color it is holding (`robot_has_color`).
          ii. If the robot's location is unknown (e.g., not in `robot_locations`), this robot cannot reach the tile; skip it.
          iii. Calculate the Manhattan distance (`dist`) between the robot's location and the `goal_tile`. If distance cannot be calculated (e.g., invalid tile name), skip this robot for this tile.
          iv. Calculate the cost for this robot to get the required `goal_color` (`cost_get_color`):
              - 0 if `robot_has_color` is already `goal_color`.
              - 1 if `robot_has_color` is None (robot has no color, needs to pick up).
              - 2 if `robot_has_color` is different from `goal_color` (robot has wrong color, needs to paint dummy + pickup).
          v. The total estimated cost for this specific robot to paint this `goal_tile` is `dist + cost_clear + cost_get_color + cost_paint`.
          vi. Update `min_cost_for_tile = min(min_cost_for_tile, cost_for_this_robot)`.
       h. Add `min_cost_for_tile` to the `total_cost`.
    6. After iterating through all goal tiles:
       a. If `all_goals_satisfied` is True, return 0.
       b. If `total_cost` is still infinity (meaning there were unsatisfied goals, but no robot could reach any of them), return infinity.
       c. Otherwise, return the calculated `total_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        self.static = task.static
        # task.facts is assumed to contain all ground facts including type predicates like (robot robot1)

        # Extract goal tiles and their required colors
        self.goal_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if len(parts) == 3 and parts[0] == 'painted':
                # Goal is (painted tile_name color_name)
                self.goal_tiles[parts[1]] = parts[2]

        # Extract robot and color objects based on type predicates in task.facts
        self.robots = set()
        self.colors = set()
        # Assuming task.facts contains facts like (robot robot1), (color white), etc.
        for fact in task.facts:
             parts = get_parts(fact)
             if len(parts) == 2: # Check for type predicates like (robot robot1)
                 obj_type, obj_name = parts
                 if obj_type == 'robot':
                     self.robots.add(obj_name)
                 elif obj_type == 'color':
                     self.colors.add(obj_name)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state  # Current world state (frozenset of strings)

        # Parse current state to get robot locations, robot colors, and tile status
        robot_locations = {} # Maps robot name to tile name
        robot_colors = {} # Maps robot name to color name or None
        tile_status = {} # Maps tile name to ('painted', color) or 'clear'

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == 'robot-at' and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts[0] == 'robot-has' and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif parts[0] == 'painted' and len(parts) == 3:
                tile, color = parts[1], parts[2]
                tile_status[tile] = ('painted', color)
            elif parts[0] == 'clear' and len(parts) == 2:
                 tile = parts[1]
                 # Record clear status only if not already marked as painted
                 if tile not in tile_status or tile_status[tile][0] != 'painted':
                     tile_status[tile] = 'clear'

        total_cost = 0
        all_goals_satisfied = True

        # Iterate through each goal tile
        for goal_tile, goal_color in self.goal_tiles.items():
            # Check if this goal tile is already satisfied
            current_status = tile_status.get(goal_tile)
            if current_status == ('painted', goal_color):
                continue # This goal is met

            all_goals_satisfied = False # Found an unsatisfied goal

            # This goal tile needs painting. Find the minimum cost for any robot.
            min_cost_for_tile = float('inf')

            # Cost to clear the tile if it's painted with the wrong color
            cost_clear = 0
            if current_status is not None and current_status[0] == 'painted' and current_status[1] != goal_color:
                 cost_clear = 1 # Assumes a 'clean' action exists

            # Cost to paint the tile itself
            cost_paint = 1

            # Consider each robot
            if not self.robots:
                 # If there are unsatisfied goals but no robots, min_cost_for_tile remains inf
                 pass # min_cost_for_tile is already inf
            else:
                for robot_name in self.robots:
                    robot_loc = robot_locations.get(robot_name)
                    if robot_loc is None:
                        # Robot exists but its location is unknown in the state.
                        # This robot cannot work on this tile.
                        continue

                    robot_has_color = robot_colors.get(robot_name) # None if robot-has fact is missing

                    # Calculate movement cost
                    dist = manhattan_distance(robot_loc, goal_tile)
                    if dist == float('inf'):
                         # Tile name malformed or grid disconnected? Assume unreachable.
                         continue

                    # Calculate cost to get the required color for this robot
                    cost_get_color = 0
                    if robot_has_color != goal_color:
                        if robot_has_color is not None: # Has wrong color
                            cost_get_color = 2 # Assumes paint dummy + pickup
                        else: # Has no color
                            cost_get_color = 1 # Assumes pickup

                    # Total cost for this robot to paint this specific goal tile
                    cost_for_this_robot = dist + cost_clear + cost_get_color + cost_paint

                    # Update minimum cost for this goal tile
                    min_cost_for_tile = min(min_cost_for_tile, cost_for_this_robot)

            # Add the minimum cost found for this goal tile to the total heuristic
            total_cost += min_cost_for_tile # Adding inf is handled correctly by float

        # Final heuristic value
        if all_goals_satisfied:
            return 0
        elif total_cost == float('inf'):
            # This happens if there are unsatisfied goals, but for every unsatisfied goal,
            # the min_cost_for_tile was infinity (e.g., no robots, or unreachable tiles).
            # This indicates an unsolvable state.
            return float('inf')
        else:
            return total_cost
