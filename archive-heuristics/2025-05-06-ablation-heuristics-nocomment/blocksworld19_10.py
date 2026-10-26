from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to parse a PDDL fact string into parts."""
    # Remove surrounding parentheses and split by space
    # Example: '(on b1 b2)' -> ['on', 'b1', 'b2']
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        three components:
        1. A base cost for each goal fact (on ?x ?y) or (on-table ?x) that is not
           satisfied in the current state. Each unsatisfied goal fact contributes
           a cost of 2, representing the minimum actions (pickup/unstack +
           stack/putdown) required to place the block in its goal position.
        2. A clearing cost for blocks that are obstructing a goal base. If a
           block Y is required to be clear (either because another block needs
           to be stacked on it according to the goal, or Y itself needs to be
           on the table according to the goal), and Y is not clear in the
           current state, a penalty is added. This penalty is 2 times the total
           number of blocks currently stacked on top of Y. Each block in the
           obstructing stack needs to be moved (unstack + putdown/stack).
        3. A small penalty (1) if the arm is not empty, as the goal requires
           the arm to be empty.

    Assumptions:
        - The heuristic is designed for the standard Blocksworld domain as
          provided, with actions pickup, putdown, stack, and unstack, each
          having an implicit cost of 1.
        - The goal state consists of a conjunction of (on ?x ?y) and (on-table ?x)
          facts, and possibly (clear ?x) facts (which are typically derived).
        - The heuristic is non-admissible and intended for greedy best-first search.
        - The state representation is a frozenset of PDDL fact strings.
        - The task object provides goal facts in `task.goals` and static facts
          in `task.static` (static facts are empty in Blocksworld).

    Heuristic Initialization:
        In the constructor, the heuristic pre-processes the goal facts to
        identify:
        - `goal_on_facts`: A set of tuples representing the (on ?x ?y) goal facts.
        - `goal_ontable_facts`: A set of tuples representing the (on-table ?x) goal facts.
        - `goal_bases`: A set of blocks that serve as bases in the goal state
          (i.e., blocks Y in (on X Y) goal facts, or blocks C in (on-table C)
          goal facts). These are the blocks that might need to be clear.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the heuristic value `h` to 0.
        2. Parse the current state facts into convenient structures for quick lookups:
           - `current_on_facts`: Set of tuples ('on', X, Y) from the state.
           - `current_ontable_facts`: Set of tuples ('on-table', X) from the state.
           - `current_clear_facts`: Set of tuples ('clear', X) from the state.
           - `arm_empty`: Boolean indicating if '(arm-empty)' is in the state.
           - Build a map `current_stack_map`: block_below -> set of blocks_on_top, from `current_on_facts`.
        3. Calculate the base cost for unsatisfied goal facts:
           - Iterate through each goal fact ('on', A, B) in `self.goal_on_facts`.
           - If ('on', A, B) is not present in `current_on_facts`, add 2 to `h`.
           - Iterate through each goal fact ('on-table', C) in `self.goal_ontable_facts`.
           - If ('on-table', C) is not present in `current_ontable_facts`, add 2 to `h`.
        4. Calculate the clearing cost for blocked goal bases:
           - Iterate through each block `base` in `self.goal_bases`.
           - If `base` is not 'table' (the table is always clear):
             - Check if `base` is clear in the current state by looking for ('clear', base) in `current_clear_facts`.
             - If `base` is *not* clear:
               - Use the `current_stack_map` and a recursive helper function `count_stack_on_top` to find the total number of blocks currently stacked directly or indirectly on top of `base`.
               - Add `2 * (total number of blocks in the stack on top of base)` to `h`.
        5. Add a penalty if the arm is not empty:
           - If `arm_empty` is False, add 1 to `h`.
        6. Return the final heuristic value `h`.
    """
    def __init__(self, task):
        super().__init__(task)
        # Pre-process goal facts
        self.goal_on_facts = set()
        self.goal_ontable_facts = set()
        self.goal_bases = set() # Blocks that are bases in goal stacks or on table

        for goal_fact_str in self.goals:
            parts = get_parts(goal_fact_str)
            predicate = parts[0]
            if predicate == 'on':
                # Goal is (on A B)
                A, B = parts[1], parts[2]
                self.goal_on_facts.add(tuple(parts))
                self.goal_bases.add(B) # B is a base
            elif predicate == 'on-table':
                # Goal is (on-table C)
                C = parts[1]
                self.goal_ontable_facts.add(tuple(parts))
                self.goal_bases.add(C) # C is a base on the table
            # Ignore (clear X) goals, they are usually derived and handled by clearing cost

    def __call__(self, node):
        state = node.state

        # Parse current state facts into sets for efficient lookup
        current_on_facts = set()
        current_ontable_facts = set()
        current_clear_facts = set()
        arm_empty = False

        # Build a map: block_below -> set of blocks_on_top for quick stack traversal
        current_stack_map = {}

        for fact_str in state:
            parts = get_parts(fact_str)
            predicate = parts[0]
            if predicate == 'on':
                current_on_facts.add(tuple(parts))
                # Add to stack map: parts[2] is below parts[1]
                current_stack_map.setdefault(parts[2], set()).add(parts[1])
            elif predicate == 'on-table':
                current_ontable_facts.add(tuple(parts))
            elif predicate == 'clear':
                current_clear_facts.add(tuple(parts))
            elif predicate == 'arm-empty':
                arm_empty = True
            # Ignore 'holding' for this heuristic calculation

        h = 0

        # 1. Base cost for unsatisfied goal facts
        for goal_on_fact in self.goal_on_facts:
            if goal_on_fact not in current_on_facts:
                h += 2 # Cost to place block A on B (pickup/unstack + stack)

        for goal_ontable_fact in self.goal_ontable_facts:
             if goal_ontable_fact not in current_ontable_facts:
                 h += 2 # Cost to place block C on table (pickup/unstack + putdown)

        # 2. Clearing cost for blocked goal bases
        # Helper function to count blocks in the stack on top of 'current_base'
        def count_stack_on_top(current_base, stack_map):
            count = 0
            blocks_directly_on_top = stack_map.get(current_base, set())
            for block in blocks_directly_on_top:
                count += 1 # Count the block itself
                # Recursively count blocks on top of this block
                count += count_stack_on_top(block, stack_map)
            return count

        for base in self.goal_bases:
            # The table ('table') is always clear, no clearing cost needed
            if base != 'table':
                # Check if the base block is clear in the current state
                # A block Y is clear if (clear Y) is in the state AND nothing is on top of it.
                # The check ('clear', base) not in current_clear_facts is sufficient
                # because the PDDL state representation includes (clear Y) iff nothing is on Y.
                if ('clear', base) not in current_clear_facts:
                    # Base is not clear, find the stack on top of it
                    num_blocks_in_stack_on_base = count_stack_on_top(base, current_stack_map)
                    # Each block in the stack needs to be moved (unstack + putdown/stack)
                    h += 2 * num_blocks_in_stack_on_base

        # 3. Penalty if arm is not empty
        if not arm_empty:
             h += 1

        # The heuristic is 0 iff all goal (on) and (on-table) facts are satisfied,
        # all goal bases are clear, and the arm is empty. This is exactly the goal state.
        # So, h=0 iff goal state is reached.

        return h
