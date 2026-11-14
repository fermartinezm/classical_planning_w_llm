from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by stripping parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with wildcards.

    :param fact: PDDL fact as a string, e.g. "(available-color color1 tile2)".
    :param args: Pattern parts with possible wildcards.
    :return: True if fact matches pattern, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the 'color' (secret_domain) domain.

    Summary:
    This heuristic estimates the number of actions required to paint all goal tiles.
    Painting a tile requires the robot to be at some color, 
    the tile to be "down" (not yet painted),
    and the robot to have a free color. 
    The heuristic approximates this by:
    1) Counting the number of tiles that must still be painted,
    2) Estimating the minimum number of "paint_up" actions (one per tile),
    3) Adding estimated costs for preparing robot position (moving "available-color" along the "up" chain) 
       to enable painting,
    4) Approximating the minimal necessary "change_color" actions by counting steps between current and required tile positions,
    5) Ensuring the estimation is efficient without full search or path computation.

    Assumptions:
    - Each goal tile must be painted once (one "paint_up" action per tile).
    - The robot can only paint a tile if:
      * The tile has "down" (meaning it is paintable),
      * The robot is associated to a color that has an "available-color" fact on some tile below,
      * The robot color has a "free-color" fact (not currently busy painting),
      * The "available-color" of the robot can be moved up tile-by-tile via "change_color" actions.
    - The "up" relation defines a chain of tiles through which "available-color" can be moved upward.
    - Painting a tile consumes "free-color" from the color and removes "down" from the tile.
    - Moving the robot along colors does not change "available-color" facts, only "change_color" actions move availability upward along tiles.
    - "move_down" action changes robot-at facts but in the problem it appears static between colors.
    - We ignore interactions between multiple robots/colors except to count how many are free and how many are busy.
    - The heuristic need not be admissible, only informative and efficiently computable.

    Heuristic Initialization:
    - Extract the "up" relations between tiles into a dictionary mapping each tile to the tile above it.
    - Extract the set of goal tiles to paint.
    - Extract static facts for "down" tiles and "up" relations, since they don't change.
    - During heuristic calls, analyze the current state to determine painted and unpainted tiles.
    - Extract current "available-color" facts to find positions of colors.
    - Extract which colors are free (have "free-color") and which are busy.
    - Extract robot-at facts linking robot colors to colors.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify all goal tiles that still are not painted (missing "painted" fact).
    2. For each tile requiring painting:
       - Check the tile(s) below it through "up" relations to estimate how many "change_color" actions are needed to bring "available-color" of the robot to the tile below the goal tile.
       - In the domain, "change_color" moves "available-color" from a tile to the tile above it.
       - The robot can only paint when the relevant "available-color" is at the tile "below" the tile to paint (because paint_up requires "available-color" of both the robot and the tile below the tile-2 to paint).
    3. Count the "paint_up" actions needed: one per unpainted goal tile.
    4. Count the minimal needed "change_color" steps for the robot to bring its available color to the correct tile to enable painting.
       - Since we only have upward moves from tile to tile via "change_color", estimate the number of upward steps needed from the robot's current "available-color" tile to the immediate lower tile of the goal tile.
       - Sum these minimal steps across robots/colors relevant to painting goal tiles.
       - If multiple tiles share the same "available-color" robot, try to minimize counting duplicate upward moves.
         Because our heuristic only needs to be efficient and informative, we will approximate by summing per tile independently.
    5. Also add a small cost to switch robot "free-color" status if needed.
    6. The heuristic is zero if all goal tiles are painted.
    7. The heuristic returns a finite integer estimation of minimal actions needed given current facts.

    This heuristic guides greedy best-first search by focusing on painting remaining tiles,
    moving "available-color" upward in the tile stack as needed, and considering the robot color availability.
    """

    def __init__(self, task):
        # Extract goal tiles to paint
        self.goal_tiles_to_paint = set()
        for goal in task.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                self.goal_tiles_to_paint.add(args[0])

        # Extract static 'up' relations: tile -> tile_above
        self.up_map = {}
        for fact in task.static:
            pred, *args = get_parts(fact)
            if pred == "up":
                lower_tile, upper_tile = args
                self.up_map[lower_tile] = upper_tile

        # For quick downward search, build inverse map: tile_above -> tile_below (many-to-one possible)
        # This is useful for finding "below" tiles of tile-2
        self.down_map = {}
        for lower, upper in self.up_map.items():
            self.down_map.setdefault(upper, []).append(lower)

    def _distance_upwards(self, start_tile, target_tile):
        """
        Compute a number of upward steps (using up_map) from start_tile to target_tile.
        Returns number of steps if target_tile is reachable upwards from start_tile,
        else None.

        This function counts how many "change_color" moves needed to bring available-color
        from start_tile to target_tile (which is above start_tile).

        Note: upward moves only go from lower to upper in 'up_map'.
        """
        steps = 0
        current = start_tile
        while current in self.up_map:
            if self.up_map[current] == target_tile:
                return steps + 1
            current = self.up_map[current]
            steps += 1
            if current == target_tile:
                return steps
        return None

    def __call__(self, node):
        state = node.state

        # Quick check: if goals already reached, heur = 0
        if self.goal_tiles_to_paint <= state:
            return 0

        # Identify painted tiles
        painted_tiles = {get_parts(fact)[1] for fact in state if fact.startswith("(painted ")}

        # Tiles still requiring painting
        tiles_to_paint = self.goal_tiles_to_paint - painted_tiles
        if not tiles_to_paint:
            return 0  # all painted

        # Extract free-colors set
        free_colors = {get_parts(fact)[1] for fact in state if fact.startswith("(free-color ")}

        # Extract robot-at facts: robot_color -> color
        robot_at = {}  # robot_color: color
        for fact in state:
            if fact.startswith("(robot-at "):
                parts = get_parts(fact)
                # (robot-at robot_color color)
                robot_color, color = parts[1], parts[2]
                robot_at[robot_color] = color

        # Extract available-color facts: robot_color or tile or tile2 -> tile
        # They are of form (available-color ?v1 - robot ?v2 - tile)
        # but note in il examples some are (available-color tile1 tile4) (tile1 and tile4 are both tiles)
        # We consider the first argument as robot_color or tile (since typing is unusual).
        # We'll index by the first argument: available_color_positions[robot_color_or_tile] = tile where color is available
        available_color_positions = {}
        for fact in state:
            if fact.startswith("(available-color "):
                parts = get_parts(fact)
                arg1, arg2 = parts[1], parts[2]
                available_color_positions[arg1] = arg2

        # Extract the tiles that have 'down' (must be painted).
        down_tiles = {get_parts(fact)[1] for fact in state if fact.startswith("(down ")}

        # Heuristic value accumulator
        heuristic = 0

        # Count paint_up actions: one per tile to paint
        heuristic += len(tiles_to_paint)

        # For each tile that needs painting, estimate needed change_color actions for the robot color to have available-color on the tile below

        # For 'paint_up' action, preconditions require available-color ?v1 ?v2 (robot_color on tile below?)
        # and available-color ?v6 ?v2 (tile2 on same tile), and robot-at ?v1 ?v3 and free-color ?v3 and down ?v6

        # This means the robot color must have available-color on a tile ?v2 which is below the tile-2 (goal tile) to paint,
        # and the tile-2 has down and is adjacent (up) to that tile.

        # We will try to estimate for each tile_to_paint the minimal upward moves (change_color actions)
        # needed for some robot color to move available-color from current tile to the tile below the tile_to_paint.

        # We try to find the closest available robot color for each tile.

        # For this, for each tile_to_paint:
        #    Find tiles below it (down_map)
        #    Among tiles below, pick tiles where robot-color has available-color fact
        #    Compute steps needed from current available-color position to tile below the tile_to_paint
        #    Add minimal steps for change_color + 1 paint_up action (paint_up counted already)

        # Get for each tile_to_paint its tile_below (from down_map)
        # According to example, tile-2 is painted, tiles of type tile are below, and "available-color" must be on tile below target tile

        # For simplification, we try to find for each tile_to_paint the tile just below it (from down_map)
        # down_map may have multiple lower tiles for one tile above, but paint_up precondition is on available-color on ?v2 which is tile below.

        # We take the first tile below (if exists). If no tile below, assume 0 moves needed (corner case).

        # Also consider minimal upward moves for robot colors to bring available-color to the tile below the tile_to_paint

        for tile_goal in tiles_to_paint:
            # Find tiles below tile_goal
            tiles_below = self.down_map.get(tile_goal, [])
            if not tiles_below:
                # No tile below => no change_color needed? 
                # Because paint_up requires available-color on some tile below.
                # If none, heuristic adds no upward moves.
                continue
            tile_below = tiles_below[0]  # pick first

            # For each robot_color with free-color that has available-color position, estimate upward steps to tile_below
            min_up_moves = None
            for robot_color, available_tile in available_color_positions.items():
                if robot_color not in free_colors:
                    continue  # the color not free cannot paint now
                # Compute how many change_color moves to move from available_tile upwards to tile_below
                if available_tile == tile_below:
                    # already in place, 0 moves
                    steps = 0
                else:
                    # find distance upwards
                    steps = self._distance_upwards(available_tile, tile_below)
                    if steps is None:
                        # Not reachable upwards, impossible path for this robot_color
                        continue
                if min_up_moves is None or steps < min_up_moves:
                    min_up_moves = steps

            # If no robot color can reach tile_below by upwards moves, assume a penalty of 2 to encourage search
            if min_up_moves is None:
                min_up_moves = 2

            heuristic += min_up_moves

        return heuristic
