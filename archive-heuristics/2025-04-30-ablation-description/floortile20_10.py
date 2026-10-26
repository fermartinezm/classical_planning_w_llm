from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or non-string input gracefully
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
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
    This heuristic estimates the number of tiles that need to be painted
    with a specific color according to the goal, and are not currently
    painted with that correct color. It returns infinity if any tile
    required by the goal is painted with the wrong color, as tiles cannot
    be unpainted or repainted in this domain.

    # Assumptions
    - The goal is a conjunction of (painted tile color) facts.
    - Tiles, once painted, cannot be cleared or repainted with a different color
      (based on the provided domain actions, which only allow painting clear tiles).
    - If a tile required by the goal is painted with the wrong color, the goal is unreachable.
    - The state representation includes all relevant 'painted' facts for tiles mentioned in the goal.

    # Heuristic Initialization
    - Extract the goal conditions, specifically the required (painted tile color) facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic cost to 0.
    2. Iterate through each goal condition. We are only interested in goal conditions of the form `(painted ?tile ?color)`.
    3. For each goal condition `(painted tile_X color_Y)`:
       a. Check if the current state contains the exact fact `(painted tile_X color_Y)`.
          - If yes, this goal is satisfied for this tile. Do nothing (cost remains 0 for this goal).
       b. If the state does *not* contain `(painted tile_X color_Y)`, check if the state contains *any* fact `(painted tile_X color_Z)` where `color_Z` is different from `color_Y`.
          - Iterate through all facts in the state. If a fact matches `(painted tile_X *)` and the color part is not `color_Y`, then the tile is painted with the wrong color.
          - If such a fact is found, the goal is unreachable from this state. Return `float('inf')`.
       c. If the state does *not* contain `(painted tile_X color_Y)` and the tile `tile_X` is *not* painted with any other color (as checked in step 3b), this tile needs to be painted with `color_Y`. Increment the heuristic cost by 1. This counts the minimum number of paint actions required.
    4. After checking all goal conditions, the total accumulated cost is the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        # Extract goal conditions of the form (painted tile color)
        self.goal_painted_facts = set()
        for goal in task.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                # Store as tuple for hashing/comparison: (predicate, tile, color)
                self.goal_painted_facts.add(tuple(parts))

        # Static facts are not needed for this heuristic.
        # self.static_facts = task.static

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        heuristic_cost = 0

        # Iterate through the required painted facts from the goal
        for goal_fact_parts in self.goal_painted_facts:
            _, goal_tile, goal_color = goal_fact_parts
            goal_fact_str = f"(painted {goal_tile} {goal_color})"

            # Check if the goal fact is in the current state
            if goal_fact_str in state:
                # Goal satisfied for this tile, cost is 0 for this part
                continue # Move to the next goal fact

            # If the goal fact is NOT in the state, check if the tile is painted with the wrong color
            is_painted_wrong = False
            for state_fact in state:
                state_parts = get_parts(state_fact)
                # Check if the state fact is a 'painted' predicate for the same tile
                if state_parts and state_parts[0] == "painted" and len(state_parts) == 3:
                    state_tile, state_color = state_parts[1], state_parts[2]
                    if state_tile == goal_tile and state_color != goal_color:
                        is_painted_wrong = True
                        break # Found wrong color for this tile

            if is_painted_wrong:
                # Tile is painted with the wrong color - goal unreachable
                return float('inf')
            else:
                # Tile is not painted with the correct color, and not painted with the wrong color.
                # It needs to be painted with the goal color.
                # This requires at least one paint action.
                heuristic_cost += 1

        return heuristic_cost
