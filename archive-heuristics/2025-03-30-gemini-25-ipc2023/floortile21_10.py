import math
import re
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Removes leading/trailing whitespace and parentheses.

    Args:
        fact (str): The PDDL fact string (e.g., "(robot-at robot1 tile_0_1)").

    Returns:
        list: A list containing the predicate name and its arguments.
              Returns an empty list if the fact format is invalid.
    """
    fact = fact.strip()
    if not fact.startswith("(") or not fact.endswith(")"):
        # print(f"Warning: Invalid fact format encountered: {fact}")
        return [] # Return empty list for invalid format
    return fact[1:-1].split()

# Helper function for matching facts
def match(fact_parts, *pattern):
    """
    Checks if the parts of a parsed fact match a given pattern.
    Allows '*' as a wildcard in the pattern.

    Args:
        fact_parts (list): The list of parts from get_parts().
        *pattern: A sequence of strings representing the pattern to match.

    Returns:
        bool: True if the parts match the pattern, False otherwise.
    """
    if not fact_parts: # Handle empty list from invalid format
        return False
    if len(fact_parts) != len(pattern):
        return False
    return all(fnmatch(part, pat) for part, pat in zip(fact_parts, pattern))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'floortile'.

    Summary:
    Estimates the cost to reach the goal state by summing the estimated costs for
    achieving each unsatisfied 'painted' goal predicate. The cost for each goal
    includes the paint action itself (cost 1), the minimum estimated movement cost
    for any robot to reach a tile adjacent to the goal tile (from where it can
    be painted), and the cost of changing color (cost 1) if the best robot for
    that goal doesn't already hold the required color. Manhattan distance is used
    to estimate movement costs.

    Assumptions:
    - Tile names follow the pattern 'tile_row_col' (e.g., 'tile_1_2'), enabling
      Manhattan distance calculation between tiles.
    - Robots paint tiles that are adjacent via 'up' or 'down' relations. A robot
      at tile 'x' can paint tile 'y' if '(up y x)' or '(down y x)' holds.
    - The cost of moving between adjacent tiles is 1 (approximated by Manhattan distance).
    - The cost of the 'change_color' action is 1.
    - The cost of 'paint_up' or 'paint_down' action is 1.
    - The heuristic calculates the minimum cost (movement + color change) for *each*
      unmet goal independently, considering all robots, and sums these minimum costs
      along with the base cost for the paint actions. This is a non-admissible
      simplification, as one robot action (e.g., moving, changing color) might
      enable multiple subsequent paint actions. It aims to provide a reasonably
      informative estimate for greedy search.

    Heuristic Initialization:
    - Parses the task's goal conditions (`task.goals`) to identify the target
      `(painted tile color)` states required.
    - Parses static facts (`task.static`) and initial state facts (`task.initial_state`) to:
        - Identify all robot, tile, and color objects.
        - Build an adjacency map (`self.paint_adj`) storing which tiles a robot
          must be at to paint a specific target tile (based on 'up'/'down' relations).
        - Extract tile coordinates from their names (e.g., 'tile_1_2' -> (1, 2))
          and store them in `self.tile_coords` for Manhattan distance calculation.
        - Store the set of all robots (`self.robots`).
    - These data structures are precomputed for efficient lookups during heuristic evaluation.

    Step-By-Step Thinking for Computing Heuristic:
    1. Parse Current State: Extract current robot locations (`robot_at`), the color
       each robot holds (`robot_has`), and the set of currently painted tiles
       (`painted`) from the input `node.state`.
    2. Identify Unsatisfied Goals: Compare the set of required goal `(painted tile color)`
       tuples (`self.goals`) with the currently `painted` set. If all goals are
       satisfied, the heuristic value is 0.
    3. Calculate Base Paint Cost: Initialize the heuristic value to the number of
       unsatisfied goals. Each requires at least one 'paint' action (cost 1).
    4. Calculate Movement and Color Change Costs:
       - Initialize `total_additional_cost = 0`.
       - For each unsatisfied goal `(goal_tile, goal_color)`:
         a. Find the set of possible tiles (`possible_robot_locations`) a robot
            could be on to paint `goal_tile`, using the precomputed `self.paint_adj`.
         b. If `goal_tile` cannot be painted (no entry in `paint_adj`), treat the
            state as potentially unsolvable (return infinity).
         c. Initialize `min_cost_for_this_goal = infinity`.
         d. For each robot `r` in `self.robots`:
            i. Get the robot's current tile `robot_tile` and color `robot_color`.
            ii. Calculate `color_change_cost = 1` if `robot_color != goal_color`, else 0.
            iii. Calculate the minimum Manhattan distance `min_move_dist` from
                `robot_tile` to any tile in `possible_robot_locations`. Handle cases
                where coordinates might be missing.
            iv. If `min_move_dist` is infinity (e.g., missing coordinates), this robot
                cannot reach the required position; its cost is effectively infinite.
            v. Otherwise, the estimated cost for robot `r` to prepare for painting
               this goal is `robot_cost = min_move_dist + color_change_cost`.
            vi. Update `min_cost_for_this_goal = min(min_cost_for_this_goal, robot_cost)`.
       e. If `min_cost_for_this_goal` remains infinity after checking all robots,
          the goal is considered unreachable from the current state (return infinity).
       f. Otherwise, add `min_cost_for_this_goal` to `total_additional_cost`.
    5. Total Heuristic Value: The final heuristic estimate is the sum of the base
       paint cost (step 3) and the `total_additional_cost` (step 4). Ensure the
       result is non-negative.
    """

    def __init__(self, task):
        self.goals = set()
        self.robots = set()
        self.tiles = set()
        self.colors = set()
        # self.adj = {} # General adjacency (not strictly needed for this heuristic)
        self.paint_adj = {} # tile_to_paint -> {tile_robot_must_be_at}
        self.tile_coords = {} # tile -> (row, col)

        # Compile regex for parsing tile coordinates efficiently
        self.tile_pattern = re.compile(r"tile_(\d+)_(\d+)")

        # 1. Parse goal conditions
        for goal_fact in task.goals:
            parts = get_parts(goal_fact)
            if match(parts, "painted", "*", "*"):
                tile, color = parts[1], parts[2]
                self.goals.add((tile, color))
                self.tiles.add(tile) # Keep track of tiles involved in goals

        # 2. Parse static and initial facts to identify objects and relations
        # Combine static and init for comprehensive initial knowledge
        all_initial_facts = task.static.union(task.initial_state)

        for fact in all_initial_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            # Extract objects
            if match(parts, "robot-at", "*", "*"):
                self.robots.add(parts[1])
                self.tiles.add(parts[2])
            elif match(parts, "robot-has", "*", "*"):
                self.robots.add(parts[1])
                self.colors.add(parts[2])
            elif match(parts, "painted", "*", "*"):
                self.tiles.add(parts[1])
                self.colors.add(parts[2])
            elif match(parts, "clear", "*"):
                self.tiles.add(parts[1])
            elif match(parts, "available-color", "*"):
                self.colors.add(parts[1])
            # Extract adjacency for painting
            elif match(parts, "up", "*", "*") or match(parts, "down", "*", "*"):
                # If (up y x) or (down y x), robot at x paints y
                tile_to_paint, robot_location = parts[1], parts[2]
                self.tiles.add(tile_to_paint)
                self.tiles.add(robot_location)
                self.paint_adj.setdefault(tile_to_paint, set()).add(robot_location)
            # Keep track of tiles from movement relations too
            elif match(parts, "left", "*", "*") or match(parts, "right", "*", "*"):
                 self.tiles.add(parts[1])
                 self.tiles.add(parts[2])


        # 3. Extract coordinates for all identified tiles
        missing_coords = False
        for tile in self.tiles:
            match_coords = self.tile_pattern.match(tile)
            if match_coords:
                row = int(match_coords.group(1))
                col = int(match_coords.group(2))
                self.tile_coords[tile] = (row, col)
            else:
                print(f"Warning: Tile name '{tile}' does not match expected pattern 'tile_row_col'. Manhattan distance cannot be computed for this tile.")
                missing_coords = True

        if not self.robots:
             print("Warning: No robots were identified from the task's initial or static facts.")
        if missing_coords:
             print("Warning: Some tile coordinates could not be determined. Heuristic might be inaccurate or return infinity.")


    def _manhattan_distance(self, tile1, tile2):
        """Calculates Manhattan distance between two tiles using precomputed coordinates."""
        if tile1 not in self.tile_coords or tile2 not in self.tile_coords:
            # Return infinity if coordinates are missing for either tile
            return float('inf')

        coord1 = self.tile_coords[tile1]
        coord2 = self.tile_coords[tile2]
        return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.

        Args:
            node: The search node containing the current state.

        Returns:
            int or float: The estimated cost to reach the goal. Returns float('inf')
                          if the goal seems unreachable from the current state.
        """
        state = node.state

        # 1. Parse current state
        robot_at = {}
        robot_has = {}
        painted = set()
        # clear_tiles = set() # Not used in this version, but could be for BFS pathfinding

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            if match(parts, "robot-at", "*", "*"):
                robot_at[parts[1]] = parts[2]
            elif match(parts, "robot-has", "*", "*"):
                robot_has[parts[1]] = parts[2]
            elif match(parts, "painted", "*", "*"):
                painted.add((parts[1], parts[2]))
            # elif match(parts, "clear", "*"):
            #     clear_tiles.add(parts[1])

        # 2. Identify unsatisfied goals
        unmet_goals = self.goals - painted

        if not unmet_goals:
            return 0 # Goal state reached

        # Check if robots exist
        if not self.robots:
            return float('inf') # Cannot achieve goals without robots

        heuristic_value = 0

        # 3. Cost Component 1: Base cost for painting actions
        heuristic_value += len(unmet_goals)

        # 4. Cost Component 2: Movement and Color Change
        total_additional_cost = 0
        for goal_tile, goal_color in unmet_goals:
            min_cost_for_this_goal = float('inf')

            # Find tiles from where goal_tile can be painted
            possible_robot_locations = self.paint_adj.get(goal_tile, set())

            if not possible_robot_locations:
                 # This goal tile cannot be painted based on static 'up'/'down' facts.
                 # This implies the goal is fundamentally unreachable.
                 # print(f"Warning: Goal tile {goal_tile} has no adjacent tiles defined via up/down for painting.")
                 return float('inf')

            # Check cost for each robot to achieve this goal
            for robot in self.robots:
                # Ensure robot state is known
                if robot not in robot_at:
                    # print(f"Warning: Robot {robot} position unknown in state.")
                    continue # Cannot calculate cost for this robot
                if robot not in robot_has:
                     # Assuming robots must always hold a color based on domain actions
                     # print(f"Warning: Robot {robot} color unknown in state.")
                     continue # Cannot calculate cost for this robot

                robot_tile = robot_at[robot]
                robot_color = robot_has[robot]

                # Calculate color change cost
                color_cost = 0 if robot_color == goal_color else 1

                # Calculate minimum movement cost to a painting position
                min_move_dist = float('inf')
                for adj_tile in possible_robot_locations:
                    dist = self._manhattan_distance(robot_tile, adj_tile)
                    min_move_dist = min(min_move_dist, dist)

                # If min_move_dist is inf, this robot cannot reach any required position
                if min_move_dist == float('inf'):
                    current_robot_cost = float('inf')
                else:
                    current_robot_cost = min_move_dist + color_cost

                # Update the minimum cost found so far for *this specific goal*
                min_cost_for_this_goal = min(min_cost_for_this_goal, current_robot_cost)

            # If no robot can achieve this goal (all costs were infinity)
            if min_cost_for_this_goal == float('inf'):
                # This specific goal is unreachable by any robot from the current state.
                return float('inf')

            # Add the minimum cost found for this goal to the total additional cost
            total_additional_cost += min_cost_for_this_goal

        # 5. Total Heuristic Value
        heuristic_value += total_additional_cost

        # Ensure heuristic is non-negative (should be, unless float issues occur)
        return max(0, heuristic_value)

