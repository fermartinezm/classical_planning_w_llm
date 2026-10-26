from heuristics.heuristic_base import Heuristic

# Helper function outside the class
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function outside the class
def get_objects_from_all_facts(all_possible_facts):
    """Extract all unique objects mentioned in the set of all possible ground facts."""
    objects = set()
    for fact in all_possible_facts:
        parts = get_parts(fact)
        # Objects are arguments to predicates. In blocksworld, 'table' is a special location.
        # Let's collect all arguments that are not the predicate name itself.
        for obj in parts[1:]:
            # Assuming 'table' is the only non-block object argument.
            # We want block objects.
            if obj != 'table':
                 objects.add(obj)
    return objects


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state
    by counting the number of blocks that are in the wrong position relative to
    their required base in the goal stack, plus the number of blocks that have
    the wrong block on top of them (or shouldn't have anything on top but do,
    or should have something on top but don't).

    # Assumptions
    - The goal state defines specific stacks of blocks on the table using `on` and `on-table` predicates.
    - All blocks involved in the problem are mentioned in the task's set of all possible ground facts.
    - The cost of moving a block (pickup/unstack + putdown/stack) or clearing a block (unstack + putdown) is simplified and counted based on positional/stack errors.
    - The heuristic is non-admissible but aims to guide a greedy best-first search effectively.

    # Heuristic Initialization
    - Extracts all unique block objects from the task's set of all possible ground facts (`task.facts`).
    - Builds a map from each block to its required base block or 'table'
      according to the goal state (`goal_base_map`). Blocks not appearing as the first argument
      in a goal `on` or `on-table` predicate do not have a defined goal base in this map.
    - Builds a map from each block (as the base) to the block that should be directly on top
      of it according to the goal state (`goal_top_map`). Blocks not appearing as the second argument
      in a goal `on` predicate do not have a defined goal top in this map (they should be clear).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current position of each block: Is it `holding`, `on-table`,
       or `on` another block? This information is derived from the state facts to build
       the block being held (`holding_block`), the current base of each block (`current_base_map`),
       and the block currently on top of each block (`current_top_map`).
    2. Initialize the total heuristic cost to 0.
    3. Iterate through all blocks identified during initialization (`self.all_blocks`):
       a. For the current block B, find its goal base (G_base) using `self.goal_base_map`. If B is not the first argument of any goal `on` or `on-table` predicate, G_base is None.
       b. Determine B's current state (`holding`, `on-table`, or `on`) and its current base (C_base) using the maps built from the state.
       c. Find the block that should be on top of B in the goal (G_top) using `self.goal_top_map`. If B is not the second argument of any goal `on` predicate, G_top is None (meaning B should be clear in the goal).
       d. Find the block that is currently on top of B (C_top) using `current_top_map`. If nothing is on B, C_top is None.
       e. Add 1 to the total cost if block B is currently `holding`. This counts a block that needs to be placed.
       f. If B is not `holding` AND it has a defined goal base (G_base is not None), add 1 to the total cost if its current base (C_base) is different from its goal base (G_base). This counts a block that is on the wrong base and needs to be moved.
       g. Add 1 to the total cost if the block currently on top of B (C_top) is different from the block that should be on top of B in the goal (G_top). This counts a block that has the wrong block on top, requiring an action to clear it. This covers cases where B should be clear but isn't (C_top is not None, G_top is None), or should have a specific block on it but has the wrong one or nothing (C_top != G_top).
    4. The total accumulated cost is the heuristic value for the state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal structure and objects."""
        self.goals = task.goals
        self.static = task.static # Static facts are empty in blocksworld

        # Extract all unique block objects from all possible facts in the domain/problem.
        self.all_blocks = get_objects_from_all_facts(task.facts)

        # Build the goal base map: block -> its goal base (another block or 'table')
        self.goal_base_map = {}
        # Build the goal top map: block_under -> block_on_top
        self.goal_top_map = {}

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == 'on':
                block, under_block = parts[1], parts[2]
                self.goal_base_map[block] = under_block
                self.goal_top_map[under_block] = block
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_base_map[block] = 'table'
            # Ignore 'clear' and 'arm-empty' goals for base/top maps, they are handled implicitly

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Build current state maps
        current_base_map = {}
        current_top_map = {}
        holding_block = None

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                block, under_block = parts[1], parts[2]
                current_base_map[block] = under_block
                current_top_map[under_block] = block
            elif predicate == 'on-table':
                block = parts[1]
                current_base_map[block] = 'table'
            elif predicate == 'holding':
                holding_block = parts[1]
            # Ignore 'clear' and 'arm-empty' state facts for base/top maps

        total_cost = 0

        for block in self.all_blocks:
            # Find goal base G_base for B (None if B not in goal_base_map).
            g_base = self.goal_base_map.get(block)

            # Find current state/base C_state/C_base for B.
            c_state = None
            c_base = None
            if holding_block == block:
                c_state = 'holding'
            elif block in current_base_map:
                 c_base = current_base_map[block]
                 c_state = 'on' if c_base != 'table' else 'on-table'
            # else: block is not in state facts? Assume valid states where every block is somewhere.

            # Find goal top G_top for B (None if B not in goal_top_map).
            g_top = self.goal_top_map.get(block)

            # Find current top C_top for B.
            c_top = current_top_map.get(block)

            # Cost for fixing B's position:
            if c_state == 'holding':
                total_cost += 1
            # Only penalize base mismatch if the block is part of a goal stack structure
            elif g_base is not None and c_base != g_base:
                total_cost += 1

            # Cost for fixing B's top:
            # Penalize if the current top is different from the goal top.
            # This covers:
            # - Should be clear (G_top is None) but isn't (C_top is not None).
            # - Should have G_top (G_top is not None) but has C_top != G_top (wrong block or nothing).
            if c_top != g_top:
                 total_cost += 1

        # The heuristic is 0 iff:
        # - No block is holding (total_cost += 1 for holding is 0).
        # - For every block B with a goal base G_base, its current base C_base == G_base.
        # - For every block B, its current top C_top == its goal top G_top.
        # If all these conditions are met, the state must match all goal `on`, `on-table`, and `clear` facts.
        # If `(arm-empty)` is also a goal, it's true because no block is holding.
        # So, the heuristic is 0 iff the state is a goal state.

        return total_cost
