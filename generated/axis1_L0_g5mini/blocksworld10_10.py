from heuristics.heuristic_base import Heuristic


def get_parts(fact: str):
    """Return tokens of a PDDL fact string without surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the classic Blocksworld STRIPS domain.
    The heuristic estimates the number of actions required to reach a goal
    state from the current state by counting how many blocks are not already
    arranged in the goal stacks (bottom-to-top) and converting that number
    into an estimated number of primitive actions (unstack/pick + stack/putdown).
    The heuristic also includes light corrections for the robot arm being
    already holding a block and for goals that require the arm to be empty
    or a specific block to be held.

    Assumptions
    -----------
    - Every block that appears in either the current state or the goal is a
      candidate to consider.
    - Each misplaced block will typically require two actions to be moved
      to its goal position: one pick/unstack and one putdown/stack. This
      is the basis for the 2 * (misplaced blocks) estimate.
    - If the planner's arm is already holding a block that will need to be
      moved anyway, one pick/unstack action can be saved.
    - If the goal requires the arm to be empty but it is currently holding
      a block, one extra putdown action will be required (unless that putdown
      is already counted as part of moving a misplaced block).
    - If the goal requires the robot to be holding a particular block and
      it currently is not, an extra pick/unstack action is added.

    Heuristic Initialization
    ------------------------
    The constructor extracts the set of goal facts and builds convenient
    goal mappings:
    - goal_on: mapping block -> block it should be on (or "table").
    - goal_above: inverse mapping under_block -> block that should be directly on it
                 in the goal (used to build goal stacks).
    - goal_on_table: set of blocks that should be on the table in the goal.
    The static facts are unused for Blocksworld, but are accepted for API
    compatibility.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state already satisfies all goal facts, return 0.
    2. Collect all blocks that appear either in the current state or in the
       goal facts (this defines the universe of blocks to consider).
    3. Parse the current state to build:
       - current_on: mapping block -> support (another block or "table").
       - current_above: inverse mapping support -> block that is directly on it.
       - on_table set, holding block (if any), and whether the arm is empty.
       - current stacks: lists of blocks bottom-to-top for every block that is on-table.
    4. Build goal stacks bottom-to-top similarly using goal facts.
    5. For each goal stack (a stack whose base is on-table in the goal),
       compare its bottom-up sequence with the current stack for the same base.
       Count the length of the bottom-up prefix that already matches. The sum
       of these matched positions is matched_blocks.
    6. Every block that is not part of one of these matched prefixes is
       considered "misplaced". As an approximation each misplaced block costs
       2 actions (pick/unstack + put/stack). Hence base_estimate = 2*(total_blocks - matched_blocks).
    7. Adjust the base estimate:
       - If the robot is currently holding a block that is misplaced,
         subtract 1 (we already did the pick/unstack).
       - If the goal requires (arm-empty) but the arm is not empty, add 1
         (we will need to put something down).
       - If the goal requires (holding X) for some X and X is not currently
         held, add 1 (an extra pickup is needed).
    8. Return the adjusted estimate (clipped to be non-negative). This value
       is efficiently computable and tends to guide greedy best-first search
       towards building correct stacks bottom-up.

    Notes
    -----
    - The heuristic is not admissible in general (it can underestimate or
      overestimate), but it is designed to be informative and quick to compute.
    - It returns 0 exactly for goal states (checked first).
    """

    def __init__(self, task):
        # store goal facts as a set for quick membership tests
        self.goals = set(task.goals)
        static_facts = set(task.static) if task.static is not None else set()

        # Parse goal facts to build goal_on (block -> support) and inverse goal_above
        self.goal_on = {}  # block -> support ('table' or other block)
        self.goal_above = {}  # support -> block that should be directly above it in goal
        self.goal_on_table = set()
        self.goal_holding = None
        self.goal_arm_empty = False

        for fact in self.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "on":
                block, under = parts[1], parts[2]
                self.goal_on[block] = under
                self.goal_above[under] = block
            elif pred == "on-table":
                block = parts[1]
                self.goal_on[block] = "table"
                self.goal_on_table.add(block)
            elif pred == "holding":
                self.goal_holding = parts[1]
            elif pred == "arm-empty":
                self.goal_arm_empty = True

    def __call__(self, node):
        state = set(node.state)

        # If goal already reached, heuristic is zero
        if self.goals <= state:
            return 0

        # Parse current state
        current_on = {}  # block -> support ('table' or other block)
        current_above = {}  # support -> block directly on top
        on_table = set()
        holding_block = None
        arm_empty = False

        # Collect all block names appearing in either state or goal
        all_blocks = set()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "on":
                block, under = parts[1], parts[2]
                current_on[block] = under
                current_above[under] = block
                all_blocks.add(block)
                all_blocks.add(under)
            elif pred == "on-table":
                block = parts[1]
                current_on[block] = "table"
                on_table.add(block)
                all_blocks.add(block)
            elif pred == "holding":
                holding_block = parts[1]
                all_blocks.add(holding_block)
            elif pred == "clear":
                all_blocks.add(parts[1])
            elif pred == "arm-empty":
                arm_empty = True

        # Also include blocks that appear only in the goal but not in current state facts
        for g in self.goals:
            parts = get_parts(g)
            for token in parts[1:]:
                all_blocks.add(token)

        # Build current stacks bottom -> top for every on-table base
        current_stacks = {}  # base_block -> [base, ..., top]
        for base in on_table:
            stack = [base]
            top = current_above.get(base)
            while top is not None:
                stack.append(top)
                top = current_above.get(top)
            current_stacks[base] = stack

        # Build goal stacks bottom -> top for every goal base (goal on-table), using goal_above
        goal_stacks = {}  # base_block -> [base, ..., top]
        for base in self.goal_on_table:
            stack = [base]
            top = self.goal_above.get(base)
            while top is not None:
                stack.append(top)
                top = self.goal_above.get(top)
            goal_stacks[base] = stack

        # Count matched blocks as bottom-up matching prefixes of goal stacks and current stacks
        matched_blocks = 0
        matched_set = set()
        for base, goal_stack in goal_stacks.items():
            # A match is possible only if the same base is currently on-table
            current_stack = current_stacks.get(base)
            if not current_stack:
                continue
            # Compare bottom-up
            for cur_b, goal_b in zip(current_stack, goal_stack):
                if cur_b == goal_b:
                    matched_blocks += 1
                    matched_set.add(cur_b)
                else:
                    break

        total_blocks = len(all_blocks)

        # Base estimate: each misplaced block typically needs a pick/unstack + put/stack
        misplaced = total_blocks - matched_blocks
        estimate = 2 * misplaced

        # Adjust for a block already being held: if the held block is misplaced, one pick is already done
        if holding_block is not None:
            if holding_block in all_blocks and holding_block not in matched_set:
                estimate = max(0, estimate - 1)

        # If the goal explicitly requires the arm to be empty but we are holding something, we will likely need one putdown
        if self.goal_arm_empty and not arm_empty:
            # If we're holding a block that is also counted as a misplaced block, putting it down may coincide with moving it,
            # so avoid double counting by only adding 1 if we did not already subtract that pick.
            estimate += 1

        # If the goal explicitly wants to be holding a specific block that we do not hold now, account for one additional pick
        if self.goal_holding is not None:
            if holding_block != self.goal_holding:
                # If the target block is already counted as misplaced we will likely pick it as part of the 2-action cost,
                # but to ensure the goal of holding it we add one extra pick if it is currently on the table/stack
                # (this is conservative to guide search).
                estimate += 1

        # Ensure non-negative and finite integer
        if estimate < 0:
            estimate = 0
        return int(estimate)