# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic # Uncomment in actual environment

# Define a dummy Heuristic base class for standalone testing if needed
# This is only for ensuring the code structure is correct in a simple test environment.
# The actual environment will provide the base class.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
        def __call__(self, node):
            raise NotImplementedError
        def __repr__(self):
            return f"<{self.__class__.__name__}>"


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Handle potential non-string facts or malformed facts gracefully
         return []
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required by counting the
    number of blocks that are not resting on their correct immediate base
    (either the table or another block) according to the goal state, and
    multiplying this count by 2.

    # Assumptions
    - Each block that is not on its correct goal base requires at least two
      actions (one to pick it up/unstack it and one to put it down/stack it)
      to move it towards its correct position.
    - This heuristic does not explicitly account for the cost of clearing
      blocks that are on top of misplaced blocks or target locations, making
      it potentially non-admissible but aiming for better performance in
      greedy best-first search by prioritizing states where more blocks are
      on their correct base.
    - All blocks whose final position is specified in the goal state's
      `on` or `on-table` predicates are considered.

    # Heuristic Initialization
    - Parses the goal predicates to determine the required immediate base
      for each block whose position is specified in the goal. This mapping
      is stored in `self.goal_bases`.
    - Identifies all blocks that have a specified goal base.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value `h = 0`.
    2. Initialize a counter `misplaced_base_count = 0`.
    3. Iterate through all blocks that have a specified goal base (identified
       during initialization).
    4. For each such block `b`:
       - Retrieve its goal base from `self.goal_bases`.
       - Construct the string representation of the goal predicate for this block's base
         (either `(on b goal_base)` or `(on-table b)`).
       - Check if this goal predicate string exists in the current state facts.
       - If the goal predicate string is *not* found in the current state,
         increment `misplaced_base_count`. This means the block is not on
         its correct goal base (it could be on the wrong block, on the table
         when it shouldn't be, or held by the arm).
    5. The heuristic value is `2 * misplaced_base_count`.
    6. If the current state is the goal state, all blocks with goal bases
       will be on their correct bases, `misplaced_base_count` will be 0,
       and the heuristic will correctly return 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal base information.
        """
        self.goals = task.goals
        # Static facts are not needed for this heuristic.
        # static_facts = task.static

        # Store goal locations for each block whose position is specified in the goal.
        # goal_bases[block] = block_below_it or 'table'
        self.goal_bases = {}
        # Keep track of blocks that have a specified goal base
        self.blocks_with_goal_base = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: # Skip malformed facts
                continue
            predicate = parts[0]

            if predicate == "on":
                # Goal is (on ?x ?y)
                if len(parts) == 3:
                    block, base = parts[1], parts[2]
                    self.goal_bases[block] = base
                    self.blocks_with_goal_base.add(block)
            elif predicate == "on-table":
                # Goal is (on-table ?x)
                if len(parts) == 2:
                    block = parts[1]
                    self.goal_bases[block] = 'table'
                    self.blocks_with_goal_base.add(block)
            # Ignore other goal predicates like (clear ?x) or (arm-empty) for this heuristic

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached - this is a quick check, though the heuristic
        # should naturally return 0 for goal states.
        if self.goals <= state:
             return 0

        misplaced_base_count = 0

        # Iterate through all blocks that have a specified goal base
        for block in self.blocks_with_goal_base:
            goal_base = self.goal_bases[block] # Goal base is guaranteed to exist for these blocks

            # Construct the goal predicate string for this block's base
            if goal_base == 'table':
                goal_predicate_str = f"(on-table {block})"
            else:
                goal_predicate_str = f"(on {block} {goal_base})"

            # Check if this goal predicate is true in the current state
            if goal_predicate_str not in state:
                 misplaced_base_count += 1

        # The heuristic is 2 * number of blocks not on their correct goal base.
        # Each such block needs at least 2 actions (pickup/unstack + putdown/stack).
        return 2 * misplaced_base_count
