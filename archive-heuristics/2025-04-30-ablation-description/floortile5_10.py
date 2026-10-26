from fnmatch import fnmatch
# Assuming Heuristic base class is available from 'heuristics.heuristic_base'
# from heuristics.heuristic_base import Heuristic

# If running standalone for testing purposes, you might need a dummy Heuristic class definition like this:
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle malformed facts gracefully, though PDDL facts should be well-formed
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
         return []
    # Split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments for a match
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost by counting the number of tiles that are
    required to be painted with a specific color in the goal state but are not
    yet painted correctly in the current state. Each such tile contributes 1
    to the heuristic value, representing the minimum of one paint action needed
    for that tile.

    # Assumptions
    - The primary goal is to paint specific tiles with specific colors.
    - Tiles that need to be painted in the goal are initially clear and remain
      clear until they are painted.
    - Tiles painted with the wrong color cannot be repainted in this domain
      (as there is no 'clear' action for painted tiles). The heuristic assumes
      valid problem instances where tiles requiring painting in the goal are
      either clear or already painted correctly in the current state. If a tile
      is painted with the wrong color and needs to be painted differently in the
      goal, the problem might be unsolvable, but the heuristic will still return
      a finite value (counting it as one unpainted tile).
    - The heuristic is admissible (does not overestimate the cost) because each
      tile that is not yet painted correctly requires at least one paint action.
    - The cost of movement, color changes, and adjacency requirements are
      abstracted away for simplicity and admissibility.

    # Heuristic Initialization
    - The constructor receives the planning task.
    - It extracts the goal conditions, specifically identifying all facts of the
      form `(painted tile_X_Y color)` that must be true in the goal state.
    - Static facts are stored as per the base class example, although they are
      not used in this simple heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Access the set of goal facts provided in the task.
    2. Filter these goal facts to identify only those that represent a tile
       being painted with a specific color (i.e., facts matching the pattern
       `(painted * *)`). Store these as the set of 'painted goal facts'.
    3. For a given state (represented as a frozenset of facts), initialize a
       counter `unpainted_goal_tiles_count` to 0.
    4. Iterate through each fact in the pre-computed set of 'painted goal facts'.
    5. For each 'painted goal fact', check if this exact fact is present in the
       current state.
    6. If the 'painted goal fact' is NOT found in the current state, it means
       the corresponding tile is not yet painted correctly according to the goal.
       Increment `unpainted_goal_tiles_count` by 1.
    7. After checking all 'painted goal facts', the final value of
       `unpainted_goal_tiles_count` is the heuristic estimate for the current state.
       This value represents the number of paint actions minimally required to
       satisfy the painting goals from this state, ignoring other costs.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts from the task. Stored as per example.
        self.static = task.static

        # Filter goal facts to keep only the 'painted' conditions.
        # These are the specific tiles and colors required in the goal.
        self.goal_painted_facts = {
            goal for goal in self.goals if match(goal, "painted", "*", "*")
        }

    def __call__(self, node):
        """
        Estimate the minimum cost to reach the goal state by counting
        unpainted goal tiles.

        @param node: The search node containing the current state.
        @return: The estimated cost (heuristic value) as an integer.
        """
        state = node.state

        # Count the number of goal 'painted' facts that are not present
        # in the current state. Each missing fact represents a tile that
        # still needs to be painted correctly.
        unpainted_goal_tiles_count = 0
        for goal_fact in self.goal_painted_facts:
            if goal_fact not in state:
                unpainted_goal_tiles_count += 1

        return unpainted_goal_tiles_count
