from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't try to match more parts than the fact has
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are currently clear. It sums the cost of painting each such tile,
    the cost of acquiring necessary colors, and the cost of moving a robot
    adjacent to tiles that need painting.

    # Assumptions
    - Problem instances are solvable.
    - Goal tiles that are not yet painted correctly are currently in a 'clear' state.
    - Tiles painted with the wrong color are not goal tiles or indicate unsolvability
      (this heuristic assumes the former for simplicity in solvable instances).
    - Adjacency relations (up, down, left, right) define where a robot can stand
      to paint an adjacent tile.

    # Heuristic Initialization
    - Extract the goal conditions to identify which tiles need to be painted and with which colors.
    - Build a map `paintable_from` where `paintable_from[T]` is the set of tiles `X`
      such that a robot at `X` can paint tile `T` (based on `up`, `down`, `left`, `right` static facts).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Goal Tiles Needing Painting:
       - Iterate through the goal facts `(painted T C)`.
       - For each such goal fact, check if the fact `(painted T C)` is present in the current state.
       - If the goal fact is NOT present, and the tile `T` is currently `(clear T)` in the state,
         add `T` to a set `TilesToPaint` and add color `C` to a set `ColorsNeeded`.
         (We assume goal tiles are either correctly painted or clear).

    2. Calculate Cost for Colors:
       - Initialize heuristic value `h = 0`.
       - Identify which colors are currently held by robots by checking `(robot-has R C)` facts in the state.
       - For each color `C` in the set `ColorsNeeded`:
         - If no robot currently holds color `C`, add 1 to `h`. This represents the cost of a `change_color` action for that color. This cost is added only once per needed color.

    3. Calculate Cost for Painting and Proximity:
       - For each tile `T` in the set `TilesToPaint`:
         - Add 1 to `h`. This represents the cost of the `paint` action for tile `T`.
         - Check if any robot is currently located at a tile `X` from which `T` can be painted (i.e., `X` is in `paintable_from[T]`).
         - Identify current robot locations by checking `(robot-at R Loc)` facts in the state.
         - If no robot is located at any tile in `paintable_from[T]`, add 1 to `h`. This represents the minimum cost (at least one move) to get a robot adjacent to `T`.

    4. Return Total Heuristic:
       - The final value of `h` is the estimated number of actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the adjacency map for painting.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Build the map: target_tile -> set_of_tiles_robot_can_be_at_to_paint_it
        self.paintable_from = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                target_tile = parts[1]
                from_tile = parts[2]
                if target_tile not in self.paintable_from:
                    self.paintable_from[target_tile] = set()
                self.paintable_from[target_tile].add(from_tile)

        # Store goal tiles and their required colors for quick lookup
        self.goal_tiles_info = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_tiles_info[tile] = color


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts).

        tiles_to_paint = set()
        colors_needed = set()
        robot_colors = {}
        robot_locations = {}
        state_facts_set = set(state) # Convert frozenset to set for faster lookups

        # Extract robot info and identify tiles needing paint and colors needed
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif parts[0] == "robot-at" and len(parts) == 3:
                robot, location = parts[1], parts[2]
                robot_locations[robot] = location

        # Identify tiles that are goal tiles, not yet painted correctly, and are clear
        for tile, required_color in self.goal_tiles_info.items():
             # Check if the tile is already painted correctly
            if f"(painted {tile} {required_color})" not in state_facts_set:
                # Check if the tile is clear (assuming solvable instances don't have wrong paint)
                if f"(clear {tile})" in state_facts_set:
                    tiles_to_paint.add(tile)
                    colors_needed.add(required_color)
                # Note: If a goal tile is not clear and not painted correctly,
                # this heuristic doesn't count it, assuming it's either a robot on it
                # (which needs to move off, handled implicitly by search) or
                # an unsolvable state (wrongly painted).

        h = 0

        # Cost for acquiring needed colors
        colors_held_by_robots = set(robot_colors.values())
        for color in colors_needed:
            if color not in colors_held_by_robots:
                h += 1 # Cost for one change_color action

        # Cost for painting and proximity for each tile that needs painting
        for tile in tiles_to_paint:
            h += 1 # Cost for the paint action

            # Check if any robot is adjacent to the tile
            is_robot_adjacent = False
            # Get the set of tiles from which 'tile' can be painted
            possible_paint_locations = self.paintable_from.get(tile, set())

            # Check if any robot's current location is in the set of possible paint locations
            for robot, location in robot_locations.items():
                 if location in possible_paint_locations:
                     is_robot_adjacent = True
                     break # Found an adjacent robot, no need to check other robots for this tile

            if not is_robot_adjacent:
                h += 1 # Cost for moving a robot adjacent

        return h
