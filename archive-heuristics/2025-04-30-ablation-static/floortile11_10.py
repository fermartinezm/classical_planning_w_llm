from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # PDDL facts are expected to be well-formed strings like "(predicate arg1 arg2)"
    # Basic check for structure
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # This case should ideally not happen with valid planner states
        # print(f"Warning: Malformed fact string: {fact}") # Optional warning for debugging
        return []
    return fact[1:-1].split()


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state
    by summing the number of tiles that need to be painted and the number of
    colors that are needed for painting but not currently held by any robot.

    # Assumptions
    - The goal requires certain tiles to be painted with specific colors.
    - Tiles that need painting according to the goal are currently 'clear'.
      (Incorrectly painted tiles are not explicitly handled; it's assumed
       solvable instances don't reach states where a goal tile is painted
       with the wrong color).
    - The cost of movement is ignored.
    - The cost of changing color is counted once per needed color if no robot
      currently holds that color. Robot coordination is ignored.

    # Heuristic Initialization
    - Stores the set of goal facts, specifically focusing on the `painted` goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal facts of the form `(painted tile_X_Y color)`.
    2. Iterate through these goal facts. For each goal `(painted t c)`:
       - Check if the fact `(painted t c)` is present in the current state. If yes, this goal is met for this tile, continue to the next goal.
       - If the goal fact `(painted t c)` is *not* in the state, check if the fact `(clear t)` is present in the state.
       - If `(clear t)` is present, this tile `t` needs to be painted with color `c`. Add `t` to a set of tiles needing paint and add `c` to a set of colors needed.
    3. The base heuristic value is the total count of tiles identified in step 2 that need painting (`|TilesToPaint|`). This is a lower bound on the number of paint actions.
    4. Identify the set of distinct colors currently held by any robot in the state by looking for facts of the form `(robot-has robot_name color_name)`.
    5. Determine the set of colors that are needed for painting (from step 2) but are *not* currently held by any robot (from step 4). Let this be `ColorsToAcquire`.
    6. Add the number of colors in `ColorsToAcquire` (`|ColorsToAcquire|`) to the base heuristic value. This represents a simplified cost for acquiring the necessary colors.
    7. The total heuristic value is the sum from steps 3 and 6. If no tiles need painting, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        # Store the goal facts. We are interested in the (painted ?x ?c) goals.
        self.goals = task.goals
        # Static facts are not used in this heuristic.
        # self.static_facts = task.static # Not needed for this heuristic

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of fact strings).

        tiles_to_paint = set() # Stores tile names that are clear and need painting
        colors_needed = set() # Stores colors needed for tiles in tiles_to_paint
        robot_colors = set() # Stores colors currently held by robots

        # Pre-process state for quick lookups
        # Create a set of state facts for O(1) average time lookup
        state_set = set(state)

        # 4. Identify colors currently held by robots
        for fact in state:
             parts = get_parts(fact)
             # Check if the fact is a valid robot-has predicate with 3 parts
             if parts and parts[0] == 'robot-has' and len(parts) == 3:
                 # parts is ['robot-has', robot_name, color_name]
                 robot_colors.add(parts[2])

        # 1. Identify goal facts of the form (painted tile_X_Y color).
        # 2. Iterate through these goal facts and filter for tiles needing paint.
        for goal_fact_str in self.goals:
            goal_parts = get_parts(goal_fact_str)
            # Ensure it's a valid painted goal fact with 3 parts
            if not goal_parts or goal_parts[0] != 'painted' or len(goal_parts) != 3:
                continue

            # Goal is (painted tile_name color_name)
            goal_predicate, tile_name, color_name = goal_parts

            # Check if the goal is already met in the state
            if goal_fact_str in state_set:
                 continue # Goal already met for this tile

            # Goal is not met. Check if the tile is clear.
            clear_fact_str = f'(clear {tile_name})'
            if clear_fact_str in state_set:
                 # Tile needs painting and is clear
                 tiles_to_paint.add(tile_name) # Store just the tile name
                 colors_needed.add(color_name)
            # Note: If the tile is painted with the wrong color, it's not clear,
            # and the goal is not met. This heuristic ignores such tiles,
            # assuming they don't occur in solvable states or don't contribute
            # positively to reducing the heuristic value.

        # 3. Base heuristic value: number of tiles to paint
        h = len(tiles_to_paint)

        # If no tiles need painting, we are in a goal state (or an ignored state).
        # The heuristic is 0.
        if h == 0:
            return 0

        # 5. Colors needed: collected in colors_needed set.
        # 6. Colors currently held by robots: collected in robot_colors set.

        # 7. Colors to acquire: needed colors not currently held by any robot.
        colors_to_acquire = colors_needed - robot_colors

        # 8. Add cost for acquiring colors.
        h += len(colors_to_acquire)

        # 9. Return total heuristic value.
        return h
