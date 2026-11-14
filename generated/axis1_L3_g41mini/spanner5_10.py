from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (color domain).

    Summary
    -------
    This heuristic estimates the number of actions required to paint the goal tiles 
    that are not yet painted. The painting process requires a robot with an available color 
    to move down the vertical tile chain by switching available colors along tiles (change_color actions), 
    move downwards (move_down actions), and then paint the tile above (paint_up action).

    Assumptions
    -----------
    - The goal tiles to paint are of type tile-2 and must have the predicate (painted).
    - To paint a goal tile, the robot must have a free color and be located at some color.
    - The robot moves through tiles (tile or tile-2) in a vertical chain using the 'up' relations.
    - The robot’s available color moves along tiles using change_color and move_down actions.
    - Painting a tile requires the tile to be in 'down' relation and free color.
    - Heuristic does not need to be admissible but tries to estimate actions by:
      * Counting unpainted goal tiles.
      * Estimating minimal steps to get robots and their available colors properly aligned.
    - The heuristic is zero exactly when all goal tiles are painted.

    Heuristic Initialization
    ------------------------
    - Extract all goal tiles (tile-2) that must be painted.
    - Parse static facts to build a directed graph representing 'up' relations between tiles.
    - Store quick lookup mappings from tile to its upper and lower neighbors (where applicable).
    - The static upward adjacency permits calculating vertical distances.
    - The painted tiles are recognized in states by the predicate (painted tile).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify all goal tiles that require painting.
    2. From the current state:
       - Identify which goal tiles are already painted.
       - Identify the robot colors and their current tiles (from available-color facts).
       - Identify which colors are free (free-color).
       - Identify robot-at facts linking robots to colors.
    3. For each unpainted goal tile, estimate the minimal cost:
       a) Finding the closest robot color that can paint it:
          - Painter must have a free color predicate.
          - Painter must be at some color (robot-at).
          - Available color must be moved vertically via change_color actions along 'up' edges.
          - Then move_down action places the robot on the tile below to prepare for paint_up action.
          - Finally, one paint_up action to paint the tile.
       b) Compute shortest vertical distance from robot's current tile to the tile below the goal tile,
          following upward edges reversed (since upward edges go from lower tile to upper tile).
       c) Number of change_color moves corresponds to vertical steps needed to align available color correctly.
       d) Number of move_down actions is generally 1 (to move from tile above to the tile-2 below).
       e) Add 1 paint_up action.
    4. Since there could be multiple robots/colors, take the minimal such cost for painting each unpainted tile.
    5. Sum the minimal estimated costs for all unpainted goal tiles.
    6. Return the total sum as heuristic value, or 0 if all goal tiles are painted.

    Notes:
    - The heuristic underestimates or approximates distances for efficiency but tries to keep a meaningful cost.
    - The number of change_color actions roughly corresponds to vertical tile moves on the tile chain.
    - Painting all goal tiles is assumed independent (sum of per-tile estimates).
    """

    def __init__(self, task):
        """
        Extract the relevant static information.

        - Identify all goal tiles to be painted.
        - Build a mapping of upward tile relationships for distance computation.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract goal tiles to paint (tile-2) from goals (painted ?tile) predicates only
        self.goal_tiles = set()
        for goal_fact in self.goals:
            pred, *args = get_parts(goal_fact)
            if pred == "painted" and args:
                self.goal_tiles.add(args[0])

        # Build upward tile adjacency from static facts (up tileX tileY)
        # up ?v4 - tile ?v5 - tile => edge from tileX -> tileY means tileY is above tileX
        # Store adjacency in both directions for navigation
        self.up_map = dict()     # tile -> tile directly above it
        self.down_map = dict()   # tile -> tile directly below it

        for fact in self.static:
            pred, *args = get_parts(fact)
            if pred == "up" and len(args) == 2:
                lower_tile, upper_tile = args
                self.up_map[lower_tile] = upper_tile
                self.down_map[upper_tile] = lower_tile

    def __call__(self, node):
        """Estimate the number of actions to paint all remaining goal tiles."""

        state = node.state

        # Quick membership functions to help identify facts of interest
        def has_fact(prefix, obj):
            # Checks for facts like (painted tileX)
            fact_str = f"({prefix} {obj})"
            return fact_str in state

        def get_facts_with_predicate(prefix):
            # Return all facts in state matching the predicate
            prefix_with_space = f"({prefix} "
            return [f for f in state if f.startswith(prefix_with_space)]

        # 1) Identify painted goal tiles in the current state
        painted_tiles = {
            fact[9:-1]  # from '(painted tileX)' extract tileX
            for fact in get_facts_with_predicate("painted")
        }

        # 2) Determine which goal tiles are still unpainted
        unpainted_goals = self.goal_tiles - painted_tiles
        if not unpainted_goals:
            return 0  # All painted: heuristic cost zero

        # 3) Gather robot colors and their current available color tiles (available-color ?color ?tile)
        # format: (available-color colorX tileY)
        available_color_facts = get_facts_with_predicate("available-color")
        # Map: robot_color -> current tile where available-color is
        available_color_positions = {}
        for fact in available_color_facts:
            parts = get_parts(fact)
            if len(parts) == 3:
                _, color_obj, tile_obj = parts
                available_color_positions[color_obj] = tile_obj

        # 4) Identify free colors (free-color ?color)
        free_color_facts = get_facts_with_predicate("free-color")
        free_colors = {fact[11:-1] for fact in free_color_facts}  # '(free-color colorX)' -> colorX

        # 5) Identify robot-at facts: (robot-at ?color-2 ?color)
        # Map robot_color2 -> color that it is at currently (color-2 and color types are robots/colors)
        robot_at_facts = get_facts_with_predicate("robot-at")
        robot_at_map = {}
        for fact in robot_at_facts:
            parts = get_parts(fact)
            if len(parts) == 3:
                _, robot_color2, color = parts
                robot_at_map[robot_color2] = color

        # Since the robot-at predicate binds a color-2 robot (robot_color2) to a color,
        # and paint_up requires robot color-2 and color, and free-color is for color,
        # we need to find which available colors correspond to free colors and those that robot-at links.

        # 6) Estimate the minimal cost for painting each unpainted tile:
        # For each unpainted tile (tile-2), find the tile (of type tile) below it (required by paint_up)
        # and estimate vertical moves needed from robot's available color tile to that tile,
        # plus one move_down action, plus one paint_up action.

        # A function to find the tile below a tile-2 (paint target)
        # As per PDDL, paint_up requires (available-color ?v1 ?v2) and (available-color ?v6 ?v2) (the same tile ?v2),
        # and (down ?v6) for tile-2, so tile-2 is down of a tile.
        def get_tile_below(tile2):
            # The down relation relates tile-2 to tile.
            # The static facts have (down tile-2 tile), but as down predicate has one parameter (?), we assume
            # from domain: (down ?v6 - tile-2), so down fact unary predicate. But domain used (down ?v6 - tile-2).
            # Since PDDL fragment shows (down ?v6 - tile-2), it is a unary predicate for tiles with down relation.
            #
            # From instance, we see 'down tile1', so tile1 (tile-2) has 'down tile1' means tile1 needs painting.
            # However, for paint_up precondition, it has available-color robot tile and available-color tile tile (same tile)
            # and robot-at robot color, free-color color, and down tile-2
            #
            # We can approximate tile below tile2 as the tile with an up edge to tile2.
            # Because if (up tileX tileY), tileY is above tileX, so tileX is below tileY.
            # So tileX is right below tileY (tile2).
            # Searching for any tile t where up(t, tile2) is in static.

            for tile_lower, tile_upper in self.up_map.items():
                if tile_upper == tile2:
                    return tile_lower
            return None

        # 7) To estimate vertical distance from one tile to another along 'up' edges:
        # The tiles form a vertical chain via 'up' edges: lower -> upper.
        # We want distance from tile start to tile target by moving up edges.
        def vertical_distance(start_tile, target_tile):
            # Return number of 'up' steps from start_tile to target_tile, or None if unreachable.
            dist = 0
            current = start_tile
            while current != target_tile:
                if current not in self.up_map:
                    return None
                current = self.up_map[current]
                dist += 1
                # Safety: if too large -> break to avoid infinite loops
                if dist > 1000:
                    return None
            return dist

        total_cost = 0

        # For each unpainted goal tile, find minimal cost over all robot colors that are free and have available-color positions.
        for tile2 in unpainted_goals:
            tile_below = get_tile_below(tile2)
            if tile_below is None:
                # If no access tile below, pessimistically assign a high cost
                total_cost += 10
                continue

            min_tile_cost = None
            # Iterate over all available colors and robots positions
            for robot_color2, color in robot_at_map.items():
                # Find available color tile the robot color2 controls
                # We must find an available-color fact where robot_color2 (color-2) has an available tile
                # From domain, available-color takes (?v1 - color-2) (?v2 - tile)
                # So robot_color2 -> tile mapping:
                available_tile = available_color_positions.get(robot_color2, None)
                if available_tile is None:
                    continue

                # Check if free-color applies to color the robot_at links to
                if color not in free_colors:
                    continue  # Must have free color

                # Compute vertical distance from available_tile to tile_below along up edges
                dist = vertical_distance(available_tile, tile_below)

                if dist is None:
                    # If unreachable, assign a large penalty for this candidate
                    candidate_cost = 10
                else:
                    # Estimated actions:
                    # dist change_color actions to move available-color upward tile by tile,
                    # 1 move_down action to move robot down to tile below goal tile,
                    # 1 paint_up action.
                    candidate_cost = dist + 1 + 1

                if min_tile_cost is None or candidate_cost < min_tile_cost:
                    min_tile_cost = candidate_cost

            if min_tile_cost is None:
                # No robot/color can reach or paint tile; assign a large cost
                total_cost += 10
            else:
                total_cost += min_tile_cost

        return total_cost
