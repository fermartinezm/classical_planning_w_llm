from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming heuristic_base.py is available

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class FloortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct colors. It sums the minimum estimated cost for each unpainted
    goal tile independently, considering the closest robot and necessary color changes.
    It also detects dead-end states where a goal tile is painted with the wrong color.

    # Assumptions:
    - Tiles are named using a 'tile_row_col' convention, allowing coordinate extraction.
    - The grid structure is implied by tile names.
    - Movement cost is approximated by Manhattan distance to the target tile's location.
    - Painting a tile requires being adjacent to it and having the correct color.
    - The heuristic ignores potential blocking by painted non-goal tiles.
    - Assumes any robot can pick up any available color.
    - Assumes goal tiles are either clear or painted in any given state.

    # Heuristic Initialization
    - Extracts the goal painted states (which tile needs which color).
    - Parses tile names to create a coordinate map for Manhattan distance calculation.
    - Identifies all possible colors from the static facts.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Check if the goal state is already reached. If yes, return 0.
    2. Identify goal tiles that are not painted correctly.
    3. For each such goal tile T:
       a. If T is painted with a color C' different from the goal color C, the state is a dead end. Return infinity.
       b. If T is clear, it needs painting. Add (T, goal_color) to a list of tasks.
       c. If T is not clear and not painted wrong, it must be painted correctly (already handled by step 1).
    4. If any dead end was detected, return infinity.
    5. If there are no clear goal tiles needing painting (but goal not reached), this implies all goal tiles are either painted correctly or painted wrong (dead end, handled in step 4). So if we reach here and the task list is empty, the goal must be reached (handled in step 1).
    6. For each task (Tile T, required Color C) in the list of tasks:
       a. Initialize a minimum cost for painting this specific tile to infinity.
       b. Consider each robot R:
          i. Determine the robot's current location (R_loc) and color (R_color).
          ii. Calculate the cost to change the robot's color to C: 0 if R_color is already C, 1 otherwise.
          iii. Estimate the movement cost for the robot to reach Tile T. Use Manhattan distance (dist) between R_loc and T.
          iv. Estimate the total actions for this robot R to paint Tile T:
              - Estimated actions to get adjacent and paint: `max(1, dist)`. This counts at least 1 action (paint) if already adjacent (dist=1), 2 actions if on the tile (dist=0, needs 1 move off + 1 paint), and `dist` actions if further (`dist-1` moves + 1 paint).
              - Color change cost: 1 if needed.
              - Total cost for robot R to paint T = `max(1, dist) + color_change_cost`.
          v. Update the minimum cost for painting Tile T with the minimum found across all robots.
       c. Add the minimum cost for painting Tile T to the total heuristic value.
    7. Return the total heuristic value (sum of minimum costs for each unpainted goal tile).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal painted states and tile coordinates.
        """
        self.goals = task.goals

        # Store goal painted states: {tile_name: color_name}
        self.goal_painted = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_painted[tile] = color

        # Map tile names to coordinates (row, col) by parsing the name string
        self.tile_coords = {}
        # Collect all tile names from initial state, goals, and static facts
        all_tile_names = set()
        for fact in task.initial_state:
             parts = get_parts(fact)
             for part in parts:
                 if part.startswith('tile_'):
                     all_tile_names.add(part)
        for goal in task.goals:
             parts = get_parts(goal)
             for part in parts:
                 if part.startswith('tile_'):
                     all_tile_names.add(part)
        for fact in task.static:
             parts = get_parts(fact)
             for part in parts:
                 if part.startswith('tile_'):
                     all_tile_names.add(part)

        for tile_name in all_tile_names:
            try:
                # Assuming tile names are like 'tile_row_col'
                parts = tile_name.split('_')
                # Ensure there are enough parts and the row/col parts are digits
                if len(parts) == 3 and parts[0] == 'tile' and parts[1].isdigit() and parts[2].isdigit():
                    row = int(parts[1])
                    col = int(parts[2])
                    self.tile_coords[tile_name] = (row, col)
                # else:
                     # print(f"Warning: Tile name '{tile_name}' does not follow 'tile_row_col' convention. Skipping coordinate parsing.")
            except (ValueError, IndexError):
                # Should be caught by the isdigit() check, but defensive
                # print(f"Warning: Could not parse coordinates for tile name: {tile_name}. Skipping.")
                pass # Suppress warning

        # Get all possible colors from static facts
        self.available_colors = set()
        for fact in task.static:
             parts = get_parts(fact)
             if parts[0] == 'available-color':
                 self.available_colors.add(parts[1])


    def manhattan_distance(self, tile1_name, tile2_name):
        """Calculate the Manhattan distance between two tiles."""
        if tile1_name not in self.tile_coords or tile2_name not in self.tile_coords:
            # If coordinates are unknown for either tile, distance is infinite
            return float('inf')

        r1, c1 = self.tile_coords[tile1_name]
        r2, c2 = self.tile_coords[tile2_name]
        return abs(r1 - r2) + abs(c1 - c2)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Check if the goal state is already reached.
        if self.goals.issubset(state):
             return 0

        # 2. Identify goal tiles that are not painted correctly and check for dead ends.
        unpainted_goal_tasks = [] # List of (tile, required_color)
        is_dead_end = False

        for goal_fact in self.goals:
             parts = get_parts(goal_fact)
             if parts[0] == "painted":
                  tile, required_color = parts[1], parts[2]

                  if goal_fact not in state:
                      # This goal tile is not painted correctly.
                      # Check if it's painted with *any* color other than the required one.
                      painted_with_wrong_color = False
                      for color in self.available_colors:
                           if color != required_color and f"(painted {tile} {color})" in state:
                                painted_with_wrong_color = True
                                break

                      if painted_with_wrong_color:
                           is_dead_end = True
                           break # Found a dead end

                      elif f"(clear {tile})" in state:
                           # Not painted correctly, not painted wrong, and is clear -> needs painting
                           unpainted_goal_tasks.append((tile, required_color))
                      # Else: Not painted correctly, not painted wrong, and not clear.
                      # This state shouldn't happen if tiles are always clear or painted.
                      # Treat as dead end for safety.
                      else:
                           is_dead_end = True
                           break


        # 4. If any dead end was detected, return infinity.
        if is_dead_end:
             return float('inf') # Use infinity to prune dead ends effectively

        # If no clear goal tiles need painting, but goal not reached, this implies
        # the only unfulfilled goals are painted incorrectly (dead end, handled above).
        # So if we reach here and unpainted_goal_tasks is empty, the goal must be reached (handled in step 1).
        # This check is technically redundant because of step 1 and the dead-end check,
        # but the logic flow leads here.

        # Get robot states (location and color)
        robot_states = {} # {robot_name: {'loc': tile_name, 'color': color_name}}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot_name, tile_name = parts[1], parts[2]
                if robot_name not in robot_states: robot_states[robot_name] = {}
                robot_states[robot_name]['loc'] = tile_name
            elif parts[0] == 'robot-has':
                robot_name, color_name = parts[1], parts[2]
                if robot_name not in robot_states: robot_states[robot_name] = {}
                robot_states[robot_name]['color'] = color_name

        # 6. Calculate total cost by summing minimum costs per unpainted tile task
        total_cost = 0
        for tile, required_color in unpainted_goal_tasks:
            min_cost_for_this_tile = float('inf')

            for robot, r_state in robot_states.items():
                # Ensure robot state is complete (should have loc and color)
                if 'loc' not in r_state or 'color' not in r_state:
                     continue # Skip if robot state is incomplete

                r_loc = r_state['loc']
                r_color = r_state['color']

                # Cost to change color if needed
                color_change_cost = 0 if r_color == required_color else 1

                # Estimated movement cost (Manhattan distance)
                dist = self.manhattan_distance(r_loc, tile)

                # If distance is infinite (tile coords not found), this robot can't reach it
                if dist == float('inf'):
                    continue

                # Estimated actions to get adjacent and paint: max(1, dist)
                # This counts at least 1 action (paint) if already adjacent (dist=1),
                # 2 actions if on the tile (dist=0, needs 1 move off + 1 paint),
                # and dist actions if further (dist-1 moves + 1 paint).
                moves_and_paint_cost = max(1, dist)

                # Total estimated cost for this robot to paint this tile
                cost = moves_and_paint_cost + color_change_cost

                min_cost_for_this_tile = min(min_cost_for_this_tile, cost)

            # Add the minimum cost found for this tile across all robots
            if min_cost_for_this_tile != float('inf'):
                 total_cost += min_cost_for_this_tile
            else:
                 # This tile is a goal tile, is clear, needs painting, but no robot can reach it
                 # (e.g., tile_coords parsing failed for this tile or robot location).
                 # This implies an unreachable state.
                 return float('inf') # Treat as dead end

        # 7. Return the total estimated cost
        return total_cost
