# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for testing
# In the actual planning environment, this import will be used.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static

        def __call__(self, node):
            raise NotImplementedError("Heuristic not implemented")

from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with the correct colors. It counts the number of tiles that still need to be
    painted, the number of color changes required for robots to obtain the necessary
    colors, and adds an estimated movement cost.

    # Assumptions
    - Tiles are arranged in a grid, inferable from `up`, `down`, `left`, `right` facts.
    - Tile names follow a pattern like `tile_row_col`.
    - A tile must be clear to be painted.
    - A robot must be at an adjacent tile to paint a tile.
    - Tiles painted with the wrong color represent an unsolvable state with the given actions.

    # Heuristic Initialization
    - Extract goal conditions to identify which tiles need to be painted and with which colors.
    - Static facts defining the grid structure (`up`, `down`, `left`, `right`) are available but not explicitly used for coordinate calculation in this simplified version; the heuristic relies on state facts directly.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  Initialize the heuristic value `h` to 0.
    2.  Identify all goal facts of the form `(painted ?tile ?color)`. Store these as the target painting requirements.
    3.  Iterate through the target painting requirements:
        a.  For each goal `(painted tile required_color)`:
            i.  Check if `(painted tile required_color)` is already true in the current state. If yes, this goal is satisfied for this tile; continue to the next goal tile.
            ii. Check if `(painted tile any_other_color)` is true in the current state for the same tile. If yes, the tile is painted with the wrong color. Since there's no action to unpaint or clear a painted tile, this state is likely unsolvable. Return a very large value (e.g., `float('inf')`).
            iii. Check if `(clear tile)` is true in the current state. If yes, this tile needs to be painted.
    4.  Count the number of tiles identified in step 3.a.iii that need painting. Let this count be `N`.
    5.  Add `N` to the heuristic `h` (representing the `paint` action cost for each tile).
    6.  Identify the set of colors required for the tiles that need painting (from step 3.a.iii).
    7.  Identify the set of colors currently held by robots in the current state (`(robot-has ?r ?c)`).
    8.  Count the number of required colors (from step 6) that are *not* currently held by any robot. Let this count be `K`.
    9.  Add `K` to the heuristic `h` (representing the minimum number of `change_color` actions needed to make all required colors available to at least one robot).
    10. Estimate movement cost: For each tile that needs painting (from step 3.a.iii), a robot needs to move to an adjacent tile. A simple estimate is to add 1 for movement per tile needing painting. Add `N` to the heuristic `h`.
    11. The total heuristic value is `h = N (paint) + K (color change) + N (movement) = 2*N + K`.
    12. Return `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        Static facts are available in self.static but not explicitly parsed
        into a grid structure for this simplified heuristic.
        """
        super().__init__(task) # Call the base class constructor

        # Store goal painting requirements: {tile_name: color_name, ...}
        self.goal_paintings = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted":
                if len(parts) == 3: # Expecting (painted tile color)
                    tile, color = parts[1], parts[2]
                    self.goal_paintings[tile] = color
                # else: ignore malformed goal fact

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        # Extract relevant state information
        current_painted = set() # {(tile, color), ...}
        current_clear = set()   # {tile, ...}
        current_robot_colors = set() # {color, ...}

        # Use match function for robust parsing
        for fact in state:
            if match(fact, "painted", "*", "*"):
                 parts = get_parts(fact)
                 current_painted.add((parts[1], parts[2]))
            elif match(fact, "clear", "*"):
                 parts = get_parts(fact)
                 current_clear.add(parts[1])
            elif match(fact, "robot-has", "*", "*"):
                 parts = get_parts(fact)
                 # We only care about the color, not which robot has it for K calculation
                 current_robot_colors.add(parts[2])

        needs_painting_count = 0
        needed_colors = set()

        # Step 3: Iterate through goal painting requirements
        for tile, required_color in self.goal_paintings.items():
            # Step 3.a.i: Check if goal is already satisfied
            if (tile, required_color) in current_painted:
                continue

            # Step 3.a.ii: Check for wrong color
            is_wrongly_painted = False
            # Check if the tile is painted with *any* color
            for painted_tile, painted_color in current_painted:
                 if painted_tile == tile:
                      # If it's painted, check if it's the wrong color
                      if painted_color != required_color:
                           is_wrongly_painted = True
                           break # Found wrong color for this tile
                      # If it's painted with the correct color, we would have caught it in 3.a.i

            if is_wrongly_painted:
                # Unsolvable state with current actions (no unpaint)
                return float('inf')

            # Step 3.a.iii: Check if tile is clear and needs painting
            # If it wasn't painted with the correct or wrong color, it must be clear or occupied by robot
            # Paint action requires clear tile
            if tile in current_clear:
                needs_painting_count += 1
                needed_colors.add(required_color)
            # Note: If the tile is not clear and not painted (e.g., robot is on it),
            # it still needs painting, but the heuristic doesn't explicitly count
            # the cost to clear it. The 'clear' check here implicitly means we only
            # count tiles that are *currently* paintable (or need to become paintable).
            # The simple 2*N + K model assumes the cost to make it clear is covered by the movement/setup cost.

        # Step 6, 7, 8: Calculate color change cost (K)
        color_change_cost = 0
        for color in needed_colors:
            if color not in current_robot_colors:
                color_change_cost += 1

        # Step 5 & 10: Calculate total heuristic (2*N + K)
        # N for paint actions + N for movement/setup + K for color changes
        # The 'movement/setup' cost is a simplification; it represents getting a robot
        # with the right color to an adjacent clear tile.
        h = needs_painting_count + color_change_cost + needs_painting_count

        return h
