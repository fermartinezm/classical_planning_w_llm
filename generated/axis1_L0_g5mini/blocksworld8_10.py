from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the space-separated parts of a PDDL fact string without surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the classic STRIPS blocksworld.
    The heuristic estimates the number of actions required to achieve the goal
    placement (on / on-table) of all blocks that are mentioned in the goal.
    It focuses on the number of blocks that are not yet placed in the correct
    goal-relative stack configuration. Each misplaced block is given an
    approximate cost (two actions: pick and place), and the estimate is
    adjusted for the case where the arm is already holding a block (reducing
    the remaining work by one action).

    Assumptions
    -----------
    - We only consider blocks that appear in goal facts (on / on-table). Blocks
      not mentioned in the goal are treated as "don't care".
    - A single physical pick + place for a block is estimated as 2 actions
      (unstack/pickup + stack/putdown). This is not necessarily admissible;
      it is a fast, informative estimate for greedy search.
    - If the planner's arm is currently holding a block, that already-paid
      pickup reduces the estimate by one action (the remaining place).
    - The heuristic assumes no parallel actions and that each block move is
      independent in its baseline cost (but stacking dependencies reduce the
      number of blocks counted as misplaced if a whole chain already matches).

    Heuristic Initialization
    ------------------------
    During construction we:
    - Extract the set of goal facts (task.goals).
    - Build a goal_parent mapping: for each block mentioned in a goal, map the
      block -> what it should be on in the goal (another block name or 'table').
    - Record the set of all objects that appear in the goal (these are the
      only blocks we must place correctly).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a search node with a state (set/frozenset of PDDL fact strings):

    1. Parse the state's facts to build a current_parent mapping for blocks:
       - If '(on X Y)' is present, current_parent[X] = Y.
       - If '(on-table X)' is present, current_parent[X] = 'table'.
       - If '(holding X)' is present, current_parent[X] = 'hand'.
       Blocks that do not appear are treated as having no known parent in the
       current state (and thus certainly not in the goal position unless the
       goal also does not constrain them).

    2. For each block that appears in the goal (goal_blocks):
       - Check whether that block's downward chain of parents in the current
         state exactly matches the required downward chain in the goal (i.e.,
         the block is on the correct supporting block, that supporting block
         is on the correct supporting block, and so on down to the table).
       - A block is considered "correct" only when the entire chain from that
         block down to the table matches the goal chain. This captures the
         idea that an entire subsequence of a goal stack is already correctly
         assembled and need not be moved again.

    3. Count correct_blocks = number of goal_blocks whose chain equals the
       goal chain. The number of misplaced goal blocks is (len(goal_blocks) -
       correct_blocks).

    4. Estimate the number of actions as:
         estimate = 2 * (misplaced_blocks)
       because (roughly) each misplaced block will require a pick + place.
       If the arm is currently holding a block (holding_count == 1),
       decrease the estimate by 1 (the pickup has already been done).
       Clamp the estimate at 0.

    5. Return the integer estimate. If the state satisfies all goal facts,
       the estimate will be 0.

    This heuristic is cheap to compute (linear in number of facts and goal
    blocks) and tends to guide the search toward assembling correct stacks
    from the bottom up, which is effective for greedy best-first search.
    """

    def __init__(self, task):
        # Save goal facts and static facts (even though static is unused here).
        self.goals = task.goals
        self.static = task.static

        # Build goal_parent mapping and the set of blocks that matter (those
        # that appear in goal facts).
        self.goal_parent = {}  # block -> 'table' or block name
        self.goal_blocks = set()
        for g in self.goals:
            parts = get_parts(g)
            pred = parts[0]
            if pred == "on":
                block, under = parts[1], parts[2]
                self.goal_parent[block] = under
                self.goal_blocks.add(block)
                # Also add the supporting block to the set so chains are compared
                # correctly when support is itself constrained by the goal.
                self.goal_blocks.add(under)
            elif pred == "on-table":
                block = parts[1]
                self.goal_parent[block] = "table"
                self.goal_blocks.add(block)

    def __call__(self, node):
        state = node.state

        # If the goal is already reached, return 0.
        if self.goals <= state:
            return 0

        # Build current_parent mapping from the state's facts.
        current_parent = {}  # block -> 'table', 'hand' or another block
        holding_count = 0
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "on":
                block, under = parts[1], parts[2]
                current_parent[block] = under
            elif pred == "on-table":
                block = parts[1]
                current_parent[block] = "table"
            elif pred == "holding":
                block = parts[1]
                current_parent[block] = "hand"
                holding_count += 1

        # Helper to check whether the current downward chain for block `b`
        # matches the goal chain for b. Only defined for blocks that are in
        # self.goal_parent (blocks constrained by the goal).
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def chain_matches_goal(b):
            # If the goal doesn't constrain b, it's not part of the counted goal blocks.
            if b not in self.goal_parent:
                return False
            # Current parent must exist and match the goal parent.
            cur_parent = current_parent.get(b, None)
            goal_parent = self.goal_parent[b]
            if cur_parent != goal_parent:
                return False
            # If the parent is a block, recursively ensure its chain matches.
            if goal_parent == "table":
                return True
            # parent is a block name; ensure chain for parent also matches goal.
            return chain_matches_goal(goal_parent)

        # Count how many of the goal_blocks are already correct (their full
        # downward chain matches the goal).
        # Only count blocks that are explicitly mentioned in the goal_parent mapping
        # (i.e., those that have a required placement).
        correct_blocks = 0
        for b in self.goal_parent.keys():
            if chain_matches_goal(b):
                # Also ensure the block is not held (a held block cannot be in place).
                if current_parent.get(b, None) != "hand":
                    correct_blocks += 1

        total_goal_blocks = len(self.goal_parent)
        misplaced = max(0, total_goal_blocks - correct_blocks)

        # Each misplaced block roughly costs 2 actions (pick + place).
        estimate = 2 * misplaced

        # If the arm is already holding a block, one pickup is already done.
        # Reduce the remaining estimate by 1 (never go below 0).
        estimate = max(0, estimate - holding_count)

        return estimate