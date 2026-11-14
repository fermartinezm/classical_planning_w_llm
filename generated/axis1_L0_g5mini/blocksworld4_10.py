from functools import lru_cache
from heuristics.heuristic_base import Heuristic


def get_parts(fact: str):
    """Remove surrounding parentheses and split a fact into its parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the classical Blocksworld domain.

    The heuristic estimates how many block-moving actions are still needed to
    achieve the goal configuration. It works by counting how many blocks are
    not "anchored" in their goal positions (i.e., the block and the entire
    chain of supports below it match the goal). Each misplaced block is assumed
    to require roughly two actions to move into place (pickup/unstack + putdown/stack).
    The heuristic also gives a small credit when the arm is already holding a block
    (one pickup is saved).

    Assumptions
    -----------
    - Goals mostly consist of on and on-table predicates (possibly clear predicates).
      clear goals are naturally handled because some block above a required-clear block
      will be detected as misplaced and counted.
    - Block names are consistent across init/state/goal facts.
    - The world contains only blocks (no separate typed reasoning is required).
    - The heuristic is not guaranteed to be admissible; it aims to guide greedy best-
      first search effectively, not to be a lower bound.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic extracts and stores:
    - The set of goal facts (self.goals).
    - A mapping goal_on such that goal_on[b] = x means "in the goal b should be on x",
      where x is another block name or None for on-table.
    - The set of blocks that appear anywhere in the task (collected from the task's
      initial state and goal facts).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Early exit: if the state's facts already satisfy all goal facts, return 0.

    2. Parse the current state:
       - Build a mapping current_on where current_on[b] = x means currently b is on x,
         current_on[b] = None means b is on the table, and current_on[b] = 'HAND'
         means it is currently held (there is a (holding b) fact). Blocks should
         be present in current_on either via (on b x), (on-table b) or (holding b).

    3. Define "anchored" for a block b:
       - b is anchored if b currently rests on the same support as in the goal
         (table or a particular block), and if that supporting block (if any) is
         itself anchored. This captures correctly placed stacks from the bottom up.
       - Example: If the goal has (on b2 b1) and (on-table b1), then b1 is anchored
         when it is on-table, and b2 is anchored when it is on b1 and b1 is anchored.

    4. Count how many blocks are anchored. Blocks that are not anchored are
       considered misplaced and will need to be moved.

    5. Convert misplaced-block-count to a numeric heuristic:
       - Base cost = 2 * number_of_misplaced_blocks (approximate two actions: pick/unstack + put/stack).
       - If the agent is currently holding a block, subtract 1 from the cost because one
         pickup action is already done for free.
       - Ensure the heuristic is non-negative.

    6. Return the resulting integer heuristic value. The heuristic is finite for solvable
       states and equals 0 exactly in goal states (checked in step 1).
    """

    def __init__(self, task):
        """Extract goal structure and block set from the task."""
        # Store goal facts for quick goal check.
        self.goals = task.goals

        # Build goal_on mapping: block -> block it should be on, or None for on-table.
        self.goal_on = {}
        self.goal_clear = set()
        blocks = set()

        for fact in self.goals:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                block, under = parts[1], parts[2]
                self.goal_on[block] = under
                blocks.add(block)
                blocks.add(under)
            elif predicate == "on-table":
                block = parts[1]
                self.goal_on[block] = None
                blocks.add(block)
            elif predicate == "clear":
                block = parts[1]
                self.goal_clear.add(block)
                blocks.add(block)

        # Also collect any other blocks mentioned in the task's facts (initial state, static).
        # Task.initial_state may contain all block mentions; task.facts may not contain objects.
        # We are conservative: look into task.initial_state and task.goals.
        try:
            initial_state = task.initial_state
        except AttributeError:
            initial_state = frozenset()

        for fact in initial_state | self.goals | getattr(task, "static", frozenset()):
            parts = get_parts(fact)
            pred = parts[0]
            # predicates that mention blocks: on, on-table, holding, clear, etc.
            if pred in ("on",):
                blocks.update(parts[1:3])
            elif pred in ("on-table", "holding", "clear"):
                if len(parts) >= 2:
                    blocks.add(parts[1])
            # ignore other predicates (not used in blocksworld domain)

        self.blocks = blocks

    def __call__(self, node):
        """Compute the heuristic value for the given search node (state)."""
        state = node.state

        # If already a goal state, heuristic is zero.
        if self.goals <= state:
            return 0

        # Build current_on mapping.
        # Values: another block name (str), None for on-table, 'HAND' for holding.
        current_on = {}
        holding_block = None
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "on":
                block, under = parts[1], parts[2]
                current_on[block] = under
            elif pred == "on-table":
                block = parts[1]
                current_on[block] = None
            elif pred == "holding":
                block = parts[1]
                current_on[block] = "HAND"
                holding_block = block

        # Ensure every known block has an entry in current_on.
        # If a block is not present in the state facts (shouldn't happen in well-formed problems),
        # assume it is on the table as a conservative default.
        for b in self.blocks:
            if b not in current_on:
                current_on[b] = None

        # Provide default goal_on for blocks not specified in the goal:
        # If a block has no goal mention, treat its goal as "don't care" and consider it anchored.
        # This avoids penalizing blocks not constrained by the goal.
        @lru_cache(maxsize=None)
        def is_anchored(b):
            """
            A block b is anchored when:
            - There is a goal constraint for b (it must be either on some block or on-table),
              and
            - the current support of b matches the goal support, and
            - if the support is another block, that supporting block is also anchored.
            If b has no goal constraint, treat it as anchored (don't force movement).
            """
            if b not in self.goal_on:
                return True  # no constraint in the goal about b -> ignore it
            goal_support = self.goal_on[b]  # may be None for table, or another block
            curr_support = current_on.get(b)
            # If currently in hand, not anchored unless goal wants it in hand (rare).
            if curr_support == "HAND":
                return False
            # Check direct support equality.
            if goal_support is None:
                # goal wants b on table
                if curr_support is not None:
                    return False
                else:
                    return True
            else:
                # goal wants b on another block
                if curr_support != goal_support:
                    return False
                # recursively ensure the supporting block is anchored
                return is_anchored(goal_support)

        # Count anchored blocks among those that have goal constraints; treat unconstrained blocks as anchored.
        anchored_count = sum(1 for b in self.blocks if is_anchored(b))

        misplaced = max(0, len(self.blocks) - anchored_count)

        # Base assumption: each misplaced block will need approx. 2 actions (pickup/unstack + putdown/stack).
        h = misplaced * 2

        # If the arm already holds a block, one pickup action is already done.
        if holding_block is not None:
            # If the held block is unconstrained in the goal (anchored by default) there is no real benefit.
            # Otherwise give one-action credit.
            if holding_block in self.blocks and not is_anchored(holding_block):
                h = max(0, h - 1)

        # Return integer heuristic value.
        return int(h)