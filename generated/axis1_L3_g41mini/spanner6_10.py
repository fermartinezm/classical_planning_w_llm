from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a pattern with wildcards.

    Args:
        fact (str): A fact string like "(predicate arg1 arg2)".
        args (str): Pattern components, e.g. "available-color", "*", "tile4".

    Returns:
        bool: True if the fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (color) domain.

    Summary
    -------
    Estimates the number of required actions to paint all goal tiles (tile-2 objects) based on the current state.
    The heuristic is designed to guide a greedy best-first search efficiently (not necessarily admissible).

    Assumptions
    -----------
    - The goal facts are all painted tiles '(painted tileX)'.
    - Painting a tile requires the robot:
      * to be at some color (robot-at ?robotColor ?color),
      * to have availability of color on a tile below the target tile (available-color ?robotColor ?tile),
      * to perform "paint_up" on a tile-2 that is down,
      * and the color must be free (free-color ?color).
    - Change_color can be used to move the available-color predicate upward along tiles linked by "up".
    - move_down can be used to set robot position from available colors and tiles.
    - Each goal tile starts with down? predicate, so painting removes this and marks painted.
    - The static "up" relationship defines vertical chains of tiles.
    - To paint a tile, the robot must propagate availability of color from current tile upward until directly below the tile-2.
    - Robot colors and colors are distinct types, and the robot-at predicate links robot colors and colors.
    - The heuristic assumes the robot must move availability steps upward (change_color),
      ensure robot-at the correct color, and then paint.

    Heuristic Initialization
    ------------------------
    - Extract all goal tiles (tile-2) to be painted from the goal conditions.
    - Extract "up" relations from static facts, build a graph/dict to quickly find upward chains.
    - Identify tiles below each tile-2 (from initial state's available-color facts and static data).
    - No actual operators are expanded in the heuristic; only cost estimates.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify which goal tiles are not painted yet.
    2. For each unpainted goal tile:
       a. Find the tile below it (there must be an "available-color ?robotColor ?tileBelow" matching).
       b. Check if any robot color has availability of color on some tile.
       c. Compute minimal steps to move availability upward from the robot's current tile to the tile below the tile-2 to be painted.
          Each upward step corresponds to one change_color action.
       d. Ensure robot-at ties robot color and color, moving robot (move_down) between color positions if necessary,
          approximate steps required to get the robot at the correct color for painting.
       e. One paint_up action is needed per tile.
    3. Sum all costs for all unpainted goal tiles.
    4. Account for the fact that free-color must be true on the color used for painting; if not free, some extra steps needed (not modeled explicitly here).
    5. Return 0 if all goal tiles are painted.

    This heuristic efficiently estimates required moves up tile chains, robot moves between colors, and painting actions.
    """

    def __init__(self, task):
        """
        Initialize by extracting goal tiles to paint and building upward adjacency from static facts.
        """
        self.goals = task.goals            # frozenset of goal facts
        self.static = task.static          # frozenset of static facts

        # Extract goal tiles to paint (tile-2 objects)
        self.goal_tiles = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "painted" and len(parts) == 2:
                self.goal_tiles.add(parts[1])

        # Build upward adjacency dict: up_map[tile] = tile_above
        self.up_map = dict()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "up" and len(parts) == 3:
                self.up_map[parts[1]] = parts[2]

        # Build reverse map to find what tiles are below a given tile (needed for paint_up)
        self.down_map = dict()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "up" and len(parts) == 3:
                self.down_map[parts[2]] = parts[1]

    def __call__(self, node):
        """
        Estimate the cost to paint all goal tiles not yet painted.

        Parameters
        ----------
        node : object
            Search node containing the current state.

        Returns
        -------
        int
            Estimated cost (number of actions) to reach the goal.
        """
        state = node.state

        # Quick check: if goal is reached, heuristic is zero
        if self.goals <= state:
            return 0

        # Parse state facts into fast lookup structures:
        available_color = dict()  # maps (robot_color) -> tile where available-color (robot_color,tile)
        robot_at = dict()         # maps robot_color -> current color the robot is at
        free_color_set = set()    # set of free-color colors
        painted_tiles = set()     # tiles painted in current state
        robot_colors = set()      # all robot colors found in robot-at
        colors = set()            # all colors in robot-at

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "available-color" and len(parts) == 3:
                # Note: available-color can be (robot_color tile) or (tile1 tile2) by domain? Only count those with robot_color as first arg
                # domain typing shows available-color ?v1 - robot ?v2 - tile
                # Heuristic assumes first argument is robot color (color-2), second tile
                # We'll just store all available-color facts mapping the first argument to the second
                available_color[parts[1]] = parts[2]
            elif pred == "robot-at" and len(parts) == 3:
                robot_colors.add(parts[1])
                colors.add(parts[2])
                robot_at[parts[1]] = parts[2]
            elif pred == "free-color" and len(parts) == 2:
                free_color_set.add(parts[1])
            elif pred == "painted" and len(parts) == 2:
                painted_tiles.add(parts[1])

        # Determine which goal tiles still need to be painted
        tiles_to_paint = self.goal_tiles - painted_tiles
        if not tiles_to_paint:
            return 0  # goal reached

        # To paint a tile, robot color must have availability of color on tile below that tile-2, and the color must be free
        # Painting steps per tile roughly:
        #  - cost to move availability of robot_color from its current tile upward to tile below
        #  - cost to move robot-at to color related to painting color (approximate cost as 0 or 1)
        #  - cost 1 to paint_up

        # Helper function: compute steps needed to get from tile_from upward to tile_to on the tile chain using "up" relation
        def upward_distance(tile_from, tile_to):
            dist = 0
            current = tile_from
            while current != tile_to:
                if current not in self.up_map:
                    return None  # no upward path found
                current = self.up_map[current]
                dist += 1
            return dist

        total_cost = 0

        for tile2 in tiles_to_paint:
            # The tile2 to be painted is a tile-2 with a tile below it:
            # Find the tile below tile2 (down relation) or from up_map reversed:
            tile_below = self.down_map.get(tile2)
            if tile_below is None:
                # No tile below? Cannot paint this tile (should not happen in proper problems)
                # Penalize high cost (e.g. 10)
                total_cost += 10
                continue

            # Find minimal cost over all robot colors that have availability
            best_cost_tile = None
            best_cost = None
            for robot_color in robot_colors:
                # The robot color's current availability tile:
                if robot_color not in available_color:
                    # no availability for this robot color in state, disregard
                    continue
                current_tile = available_color[robot_color]

                # Cost to move availability from current tile up to tile_below
                steps_up = upward_distance(current_tile, tile_below)
                if steps_up is None:
                    # no path, discard this robot_color candidate
                    continue

                # Check if robot-at links this robot_color to a color for painting
                current_color = robot_at.get(robot_color)
                if current_color is None:
                    # robot not at any color, assume cost 1 for robot move_down action to correct color
                    robot_move_cost = 1
                else:
                    # if the color is free-color, assume no cost to move the robot (heuristic simplification)
                    # if not free-color, assume extra cost of 1 to free it (pessimistic)
                    robot_move_cost = 0 if current_color in free_color_set else 1

                # Paint action cost = 1 for paint_up
                paint_action_cost = 1

                # Change_color actions cost = steps_up (one per step)
                # Move_down actions for robot position approx robot_move_cost

                cost = steps_up + robot_move_cost + paint_action_cost

                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_cost_tile = robot_color

            if best_cost is None:
                # No robot_color found to paint this tile, penalize heavily
                total_cost += 20
            else:
                total_cost += best_cost

        return total_cost
