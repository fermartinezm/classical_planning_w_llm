from fnmatch import fnmatch
# Assuming heuristics.heuristic_base exists and provides a Heuristic base class
# If not, a minimal base class might be needed, but the problem description implies it exists.
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully, though PDDL facts are structured.
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    # Use shlex.split for more robust splitting if arguments could contain spaces or special chars,
    # but simple split is usually fine for standard PDDL atoms like in Blocksworld.
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by counting blocks that are in the wrong position relative to the block
    below them in the goal stack, blocks that are obstructing desired
    placements, and whether the arm is holding a block. It provides a
    non-admissible estimate suitable for greedy best-first search.

    # Assumptions
    - The goal state is primarily defined by the desired (on ?x ?y) and
      (on-table ?x) predicates, forming one or more stacks.
    - (clear ?x) goal predicates for top blocks are implicitly handled
      if the stack below is correct and no other blocks are on top.
    - Each block that is misplaced relative to its goal position below it,
      or each block that is on top of another block where it shouldn't be
      according to the goal, requires approximately 2 actions (pick up and
      put down/stack) to move it out of the way or into place.
    - If the arm is holding a block, it requires at least 1 action to free it
      (put down or stack).

    # Heuristic Initialization
    - The heuristic parses the goal facts provided in the task.
    - It builds the `goal_below` map, which stores for each block `B` that is
      part of a goal stack, the block `C` it should be directly on top of
      (`goal_below[B] = C`) or if it should be on the table (`goal_below[B] = 'table'`).
    - It also stores the set of goal `(on ?x ?y)` facts (`goal_on_facts`) for
      efficient checking of blocking blocks.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:

    1.  **Parse Current State:** Iterate through the facts in the current state.
        -   Build the `current_below` map, storing for each block `B`, the block
            `D` it is currently on (`current_below[B] = D`) or if it's on the
            table (`current_below[B] = 'table'`).
        -   Identify the block currently held by the arm, if any (`held_block`).
        -   Collect all current `(on ?x ?y)` facts into a set (`current_on_facts`).

    2.  **Identify Misplaced Blocks:** Iterate through the blocks `B` that are
        keys in the `goal_below` map (i.e., blocks whose position relative to
        the block below is specified in the goal).
        -   Determine the desired block below `B` from `goal_below[B]`.
        -   Determine the current block below `B`. This is `current_below[B]`
            if `B` is in `current_below`, or 'arm' if `B` is the `held_block`.
            (Blocks in `goal_below` keys are assumed to be either on something,
            on the table, or held in any valid state).
        -   If the current block below `B` does not match the desired goal block
            below `B`, then `B` is considered "misplaced".
        -   Count the total number of misplaced blocks. Add `count * 2` to the
            total heuristic cost (estimating 2 actions to pick up and correctly
            place each misplaced block).

    3.  **Identify Blocking Blocks:** Compare the set of `(on ?x ?y)` facts in
        the current state (`current_on_facts`) with the set of `(on ?x ?y)`
        facts required in the goal (`goal_on_facts`).
        -   Any fact `(on A B)` that is in `current_on_facts` but *not* in
            `goal_on_facts` represents a block `A` that is currently sitting
            on block `B` but is not supposed to be there in the goal
            configuration. These are "blocking" blocks.
        -   Count the number of such non-goal `(on ?x ?y)` facts. Add `count * 2`
            to the total heuristic cost (estimating 2 actions to pick up and
            move each blocking block out of the way).

    4.  **Check Held Block:** If the `held_block` is not None (i.e., the arm is
        holding a block), add 1 to the total heuristic cost (estimating 1 action
        to put down or stack the held block and free the arm).

    5.  **Total Cost:** The sum accumulated from steps 2, 3, and 4 is the
        heuristic estimate for the current state. This value is 0 if and only
        if the state contains exactly the goal `(on ?x ?y)` and `(on-table ?x)`
        facts and the arm is empty, which corresponds to a goal state in
        Blocksworld.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        self.goals = task.goals

        # Build the goal_below map: block -> block_below_it_or_table
        self.goal_below = {}
        # Also build goal_on_facts set for step 3 calculation
        self.goal_on_facts = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, below_block = parts[1], parts[2]
                self.goal_below[block] = below_block
                self.goal_on_facts.add(goal)
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_below[block] = 'table'
            # Ignore (clear ?x) goals for this heuristic as they are often
            # consequences of the stack structure and handled by blocking checks.

        # Static facts are empty in Blocksworld, so no processing needed.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Parse current state
        current_below = {}
        held_block = None
        current_on_facts = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, below_block = parts[1], parts[2]
                current_below[block] = below_block
                current_on_facts.add(fact)
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                current_below[block] = 'table'
            elif predicate == "holding" and len(parts) == 2:
                held_block = parts[1]
            # Ignore (clear ?x) and (arm-empty) for this heuristic calculation

        total_cost = 0

        # 2. Identify misplaced blocks
        misplaced_blocks = set()
        for block, target_below in self.goal_below.items():
            current_below_block = current_below.get(block) # Default is None if not in map

            # If block is held, its effective 'below' is 'arm'
            if block == held_block:
                 current_below_block = 'arm'
            # Note: If a block is in goal_below keys but not in current_below and not held,
            # it implies the state representation is inconsistent or the block is missing.
            # Assuming valid states, blocks are always on something, on table, or held.
            # The .get(block) will return None if the block isn't a key in current_below.
            # If block == held_block, we override. If not held and not in current_below,
            # it's not on anything or on table, which is weird. We'll rely on .get()
            # returning None and comparing None != target_below if it happens.

            if current_below_block != target_below:
                 misplaced_blocks.add(block)

        total_cost += len(misplaced_blocks) * 2 # Estimate 2 actions (pick, put/stack) per misplaced block

        # 3. Identify blocking blocks (non-goal 'on' facts)
        non_goal_on_facts = current_on_facts - self.goal_on_facts
        total_cost += len(non_goal_on_facts) * 2 # Estimate 2 actions (pick, putdown/stack) per blocking block

        # 4. Check if arm is holding a block
        if held_block is not None:
            total_cost += 1 # Estimate 1 action (putdown/stack) to free the arm

        # The heuristic should be 0 iff the state is a goal state.
        # Based on the analysis, this calculation should yield 0 only when
        # all goal (on) and (on-table) facts are true and the arm is empty.
        # This covers typical Blocksworld goals including implicit (clear)
        # for stack tops.

        return total_cost
