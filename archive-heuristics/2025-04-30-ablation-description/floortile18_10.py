from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic
import re # To parse tile names
import math # For infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed ones defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(robot-at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we have the same number of parts as pattern arguments, or handle wildcards appropriately
    # A simple zip check works with fnmatch
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_tile_coords(tile_name):
    """
    Parses a tile name like 'tile_row_col' into a (row, col) tuple of integers.
    Assumes tile names follow the pattern 'tile_R_C' where R and C are integers.
    """
    match = re.match(r'tile_(\d+)_(\d+)', tile_name)
    if match:
        row = int(match.group(1))
        col = int(match.group(2))
        return (row, col)
    # Return None if format is unexpected
    return None

def manhattan_distance(coords1, coords2):
    """Calculates the Manhattan distance between two (row, col) coordinate tuples."""
    if coords1 is None or coords2 is None:
        # Should not happen if get_tile_coords is reliable for all tile objects
        return float('inf')
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    The heuristic estimates the minimum number of actions required to paint
    all goal tiles that are currently unpainted (clear). It sums the estimated
    cost for each unpainted goal tile, where the cost for a single tile is
    the minimum cost for any robot to reach an adjacent tile, change color
    if needed, and paint the tile.

    # Assumptions
    - Goal tiles are either initially clear or already painted the correct color.
      Wrongly painted tiles are not expected to be repainted to the goal color.
    - Tile names follow the format 'tile_row_col' allowing coordinate extraction.
    - The grid structure defined by 'up', 'down', 'left', 'right' predicates
      corresponds to a regular grid where Manhattan distance is a reasonable
      estimate for movement cost assuming clear paths.
    - The heuristic calculates the cost for each unpainted goal tile independently
      and assigns it to the "cheapest" robot to paint it, ignoring potential
      conflicts or shared resources (like robots needing the same path or color).
      This is a relaxation.
    - All tile objects mentioned in the problem instance follow the 'tile_R_C'
      naming convention and are parsable.
    - Robots always have a color.

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need to be painted
      and what color they need to be.
    - Extracts available colors from static facts.
    - Stores goal tiles and their required colors in a dictionary.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal conditions of the form `(painted ?tile ?color)`.
    2. For each such goal tile `T` requiring color `C`:
       a. Check the current state: If `(painted T C)` is already true, this tile is satisfied. Contribution to heuristic is 0.
       b. If `(clear T)` is true in the current state: This tile needs to be painted.
          i. Calculate the minimum estimated cost for *any* robot to paint this tile.
          ii. For each robot `R`:
              - Find its current position `R_pos` and color `R_color`.
              - Parse `R_pos` and `T` to get their grid coordinates `(rr, rc)` and `(tr, tc)`.
              - Calculate the Manhattan distance `dist = abs(rr - tr) + abs(rc - tc)`.
              - Estimate the number of move actions needed for robot R to reach a tile adjacent to T:
                - If `dist == 0` (robot is on tile T): 1 move (to move off T).
                - If `dist == 1` (robot is adjacent to T): 0 moves.
                - If `dist > 1`: `dist - 1` moves (minimum moves assuming clear path).
                Let this be `move_cost_to_adjacent`.
              - Estimate the number of color change actions needed for robot R:
                - If `R_color == C`: 0 actions.
                - If `R_color != C`: 1 action (`change_color`).
                Let this be `color_cost`.
              - The paint action itself costs 1.
              - Total estimated cost for robot R to paint tile T = `move_cost_to_adjacent + color_cost + 1`.
          iii. The minimum cost for tile T is the minimum of `Total estimated cost for robot R` over all robots R.
          iv. Add this minimum cost to the total heuristic value.
       c. If the tile is painted with a wrong color (`(painted T C')` where `C' != C`): This case is ignored by this heuristic, assuming such states either don't occur in solvable problems or don't require fixing for the goal.
    3. The total heuristic value is the sum of the minimum costs calculated for all `clear` goal tiles.
    4. If all goal tiles are painted correctly, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal tiles and their required colors.
        # Example goal: (painted tile_1_1 white)
        self.goal_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_tiles[tile] = color

        # Store available colors (useful for checking color change cost)
        self.available_colors = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "available-color", "*")
        }

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to paint all clear goal tiles.
        """
        state = node.state  # Current world state.

        # Check if the goal is already reached
        if self.goals <= state:
             return 0

        # Extract current robot positions and colors
        robot_pos = {}
        robot_color = {}
        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "robot-at" and len(parts) == 3:
                    robot, tile = parts[1], parts[2]
                    robot_pos[robot] = tile
                elif parts[0] == "robot-has" and len(parts) == 3:
                    robot, color = parts[1], parts[2]
                    robot_color[robot] = color

        # Identify clear tiles and correctly painted goal tiles
        clear_tiles = {get_parts(fact)[1] for fact in state if match(fact, "clear", "*")}
        painted_tiles = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "painted", "*", "*")}

        total_heuristic_cost = 0

        # Iterate through goal tiles that need to be painted
        for tile, required_color in self.goal_tiles.items():
            # Check if the tile is already painted correctly
            if (tile, required_color) in painted_tiles:
                continue # This goal is satisfied for this tile

            # Check if the tile is clear (and needs painting)
            if tile in clear_tiles:
                # This tile needs to be painted with required_color
                min_robot_cost_for_tile = float('inf')

                tile_coords = get_tile_coords(tile)
                # If tile_coords is None, the tile name format is unexpected.
                # This heuristic relies on parsable tile names.
                # In a real scenario, this might indicate an unsolvable problem
                # or a problem instance outside the heuristic's assumptions.
                # For this exercise, assuming parsable tile names for solvable states.
                if tile_coords is None:
                     # Cannot compute distance, skip this tile or handle error
                     # Skipping means this tile won't contribute, potentially underestimating.
                     # A better approach for unsolvable parts might be needed in a robust planner.
                     # But for a simple greedy heuristic, skipping might be acceptable if rare.
                     # Let's assume valid tile names for all relevant objects.
                     continue # Skip this tile if coordinates cannot be parsed

                # Find the minimum cost for any robot to paint this tile
                for robot, r_pos in robot_pos.items():
                    r_color = robot_color.get(robot) # Use .get() in case robot_has is missing (shouldn't happen per domain)

                    if r_color is None:
                         # Robot doesn't have a color? Skip this robot.
                         continue

                    robot_coords = get_tile_coords(r_pos)
                    if robot_coords is None:
                         # Cannot parse robot position coordinates. Skip this robot.
                         continue

                    # Calculate estimated move cost to reach a tile adjacent to the target tile
                    dist = manhattan_distance(robot_coords, tile_coords)

                    # Cost to get robot to a tile adjacent to the target tile T
                    if dist == 0: # Robot is on the target tile T
                        move_cost_to_adjacent = 1 # Must move off T first
                    else: # Robot is not on the target tile T
                        # Minimum moves assuming clear path to reach *any* tile adjacent to T
                        move_cost_to_adjacent = max(0, dist - 1)

                    # Calculate estimated color change cost
                    color_cost = 1 if r_color != required_color else 0

                    # Total estimated cost for this robot to paint this tile
                    # = moves_to_adjacent + color_change + paint_action
                    cost_for_this_robot = move_cost_to_adjacent + color_cost + 1

                    min_robot_cost_for_tile = min(min_robot_cost_for_tile, cost_for_this_robot)

                # Add the minimum cost found for this tile to the total heuristic
                if min_robot_cost_for_tile != float('inf'):
                     total_heuristic_cost += min_robot_cost_for_tile
                else:
                     # This case should ideally not be reached in a solvable state
                     # with parsable tile names and at least one robot.
                     # If it happens, it suggests no robot can paint this tile.
                     # Returning a large finite number or indicating unsolvable is an option.
                     # For a simple greedy heuristic, let's assume valid problems.
                     pass # Should not happen in expected scenarios

            # If the tile is painted the wrong color, we ignore it based on assumption.
            # If the tile is clear but NOT a goal tile, we ignore it.

        return total_heuristic_cost
