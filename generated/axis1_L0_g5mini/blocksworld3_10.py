from heuristics.heuristic_base import Heuristic


def get_parts(fact: str):
    """Remove surrounding parentheses and split a PDDL fact into its parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the Blocksworld (STRIPS) domain.

    The heuristic estimates the number of actions required to reach the goal by
    (1) counting how many blocks are not already placed in goal-aligned stacks
        (i.e., blocks that are not sitting on the correct block or on the table
        according to the goal), and
    (2) translating that number of misplaced blocks into an action estimate
        (roughly two actions per misplaced block: pick/unstack + put/stack),
        with a small correction if the planner's arm is already holding a block
        (because a held block does not need an immediate pick action).

    Assumptions
    -----------
    - A reasonable, efficiently computable estimate is obtained by assuming
      each misplaced block takes about two actions to move to its correct
      position (unstack/pick + stack/putdown).
    - Blocks that already form contiguous goal stacks anchored at a goal
      "on-table" position are treated as correctly placed and incur no cost.
    - The planner's representation uses facts in the forms:
        (on A B), (on-table A), (clear A), (holding A), (arm-empty)
      as strings (e.g. "(on b1 b2)").
    - The heuristic need not be admissible; it aims at guiding greedy best-first
      search by producing informative, quickly computed estimates.

    Heuristic Initialization
    ------------------------
    The constructor extracts and stores goal facts (self.goals) and static
    facts (self.static). From the goals it builds:
    - goal_on: mapping block -> block it should be on (from "(on A B)").
    - goal_on_table: set of blocks that should be on the table (from
      "(on-table A)").
    These structures are used repeatedly during heuristic evaluation and are
    inexpensive to build.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given search node/state:
    1. If the state's facts already satisfy all goal facts, return 0.
    2. Parse the current state's facts and build:
       - current_on: mapping block -> block it is currently on (from "(on A B)").
       - current_on_table: set of blocks currently on the table (from "(on-table A)").
       - holding_block: the block currently held (if any), and arm_empty flag.
       - the set of all block names observed in the state or in the stored goals.
    3. Count how many blocks are already in the correct relative positions
       according to goal stacks. To do this efficiently:
       - Start from each goal bottom block (a block that is required to be on
         the table in the goal), and walk upward along the goal tower using
         the inverse of goal_on (goal_below -> goal_above).
       - For each step upward, check whether the corresponding "on" relation
         exists in the current state. Stop at the first mismatch. Every matched
         pair corresponds to a block already correctly placed.
    4. Compute misplaced_blocks = total_number_of_blocks - correctly_placed_blocks.
    5. Convert misplaced_blocks to an action estimate with:
       estimated_cost = 2 * misplaced_blocks
       (each misplaced block roughly requires an unstack/pick and a stack/putdown).
    6. Adjust the estimate if the arm is already holding a block:
       - If the planner is holding a block, and that block is counted among the
         misplaced blocks (it will typically need one fewer action since the
         "pick" step is already done), subtract 1 from the estimate.
    7. Also ensure the estimate respects unsatisfied goal facts at least:
       - Let unsatisfied_goals be the number of goal facts not yet true in the
         current state. Use estimated_cost = max(estimated_cost, unsatisfied_goals).
    8. Finally, ensure the returned heuristic value is finite and non-zero for
       non-goal states (i.e., return at least 1 for any non-goal state).
    """

    def __init__(self, task):
        """
        Extract and store goal information and static facts.

        Parameters
        ----------
        task : Task
            The planning task instance provided by the planner. We read
            task.goals (set of goal facts) and task.static (set of static facts).
        """
        # Store whole goal set and static facts for use during evaluation.
        self.goals = set(task.goals)
        self.static = set(task.static)

        # Build goal_on mapping: block -> block it should be on in the goal.
        self.goal_on = {}
        # Build set of blocks that should be on the table in the goal.
        self.goal_on_table = set()

        for fact in self.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "on":
                top, below = parts[1], parts[2]
                self.goal_on[top] = below
            elif pred == "on-table":
                self.goal_on_table.add(parts[1])

        # Build inverse mapping for goal towers: below -> above (unique in proper inputs).
        self.goal_below_to_above = {}
        for top, below in self.goal_on.items():
            self.goal_below_to_above[below] = top

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given search node.

        Returns
        -------
        int
            Estimated number of actions to reach a goal state from this node.
            Zero is returned iff the node's state satisfies all goals.
            For any non-goal state, a positive integer (>= 1) is returned.
        """
        state = node.state

        # 1) If goals already reached, return 0.
        if self.goals <= state:
            return 0

        # 2) Parse current state:
        current_on = {}  # block -> block it is on (if any)
        current_on_table = set()
        holding_block = None
        arm_empty = False
        blocks_in_state = set()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "on":
                top, below = parts[1], parts[2]
                current_on[top] = below
                blocks_in_state.add(top)
                blocks_in_state.add(below)
            elif pred == "on-table":
                b = parts[1]
                current_on_table.add(b)
                blocks_in_state.add(b)
            elif pred == "holding":
                holding_block = parts[1]
                blocks_in_state.add(holding_block)
            elif pred == "clear":
                blocks_in_state.add(parts[1])
            elif pred == "arm-empty":
                arm_empty = True

        # Include blocks that appear in the goal facts but not in the current state
        # (they still count toward total blocks we must place).
        for fact in self.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in ("on", "clear"):
                blocks_in_state.add(parts[1])
                if pred == "on":
                    blocks_in_state.add(parts[2])
            elif pred == "on-table":
                blocks_in_state.add(parts[1])
            elif pred == "holding":
                blocks_in_state.add(parts[1])

        total_blocks = len(blocks_in_state)

        # 3) Count correctly placed blocks: follow each goal-anchored stack (goal on-table).
        correct = 0
        # Use a set to avoid double counting blocks that might be found in multiple traversals.
        counted = set()

        # If the goal does not explicitly state a block is on-table for a tower bottom,
        # we still try to find anchored bottoms by checking goal bottoms derived from goals:
        # bottoms are those blocks that either are on-table in goal or do not appear as a 'top' in goal_on.
        derived_bottoms = set(self.goal_on_table)
        # Add blocks that exist in goals but are not mapped as "top" (i.e., they are bottoms).
        all_goal_blocks = set()
        for top, below in self.goal_on.items():
            all_goal_blocks.add(top)
            all_goal_blocks.add(below)
        # Blocks that appear in goal but are not keys of goal_on may be bottoms (if not on someone).
        for b in all_goal_blocks:
            if b not in self.goal_on:
                derived_bottoms.add(b)

        # Prefer official goal_on_table entries as primary anchors but include derived bottoms.
        bottoms_to_check = derived_bottoms

        for bottom in bottoms_to_check:
            # bottom must exist in the current state to be considered correctly placed
            if bottom in counted:
                continue
            # Check whether bottom is indeed on-table in the current state.
            if bottom not in current_on_table:
                # Not on table in current state -> cannot be correct for that anchored tower.
                continue
            # bottom is on-table in current state; check upward chain according to goal.
            cur_below = bottom
            # If bottom itself is a goal block we count it.
            if bottom in blocks_in_state and bottom not in counted:
                # Only count bottom as correct if in the goal bottom set and the goal expects it to be on table.
                # If goal does not explicitly require bottom on-table, we still counted it as derived bottom,
                # but bottom might not be correct if goal relation below is different. We accept the on-table match
                # as correctness for the anchored stacks since goal bottoms were derived accordingly.
                correct += 1
                counted.add(bottom)

            # Walk up the goal tower: from below -> above (if any)
            while True:
                above = self.goal_below_to_above.get(cur_below)
                if above is None:
                    break
                # For 'above' to be correct, it must currently be on cur_below.
                if current_on.get(above) == cur_below:
                    if above not in counted:
                        correct += 1
                        counted.add(above)
                    cur_below = above
                else:
                    break

        # 4) Misplaced blocks are those not counted as correct.
        misplaced_blocks = max(0, total_blocks - correct)

        # 5) Base estimate: assume ~2 actions per misplaced block (pick/unstack + place/stack).
        estimate = 2 * misplaced_blocks

        # 6) Adjust if the arm is already holding a block:
        if holding_block is not None:
            # If the held block is among the blocks that we counted as misplaced (or simply exists),
            # subtract one action because pick/unstack is already done.
            if holding_block in blocks_in_state:
                # Defensive: don't reduce below zero here; will be corrected later.
                estimate -= 1

        # 7) Ensure estimate is at least the number of unsatisfied goal facts
        unsatisfied_goals = sum(1 for g in self.goals if g not in state)
        estimate = max(estimate, unsatisfied_goals)

        # 8) Ensure returned value is positive for non-goal states.
        if estimate < 1:
            estimate = 1

        return int(estimate)