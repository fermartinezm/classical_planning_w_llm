from heuristics.heuristic_base import Heuristic
from task import Task

class blocksworldHeuristic(Heuristic):
    """
    Summary:
        Domain-dependent heuristic for Blocksworld. Estimates the number of
        actions required to reach the goal state by counting misplaced blocks
        within goal stacks and penalizing unsatisfied clear and arm-empty goals.

    Assumptions:
        - All objects in the problem are blocks.
        - All blocks relevant to the goal have their final position specified
          in the goal state (either on another block or on the table).
        - The goal state consists of facts like (on X Y), (on-table X),
          (clear X), and potentially (arm-empty).

    Heuristic Initialization:
        - Parses the goal facts to build a map `goal_below_map` where
          `goal_below_map[block]` is the block that `block` should be
          immediately on top of in the goal state, or the string 'table'
          if `block` should be on the table.
        - Identifies the set of blocks that are part of the goal stacks
          (`goal_blocks`).
        - Stores the set of goal facts for quick lookup.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize a counter `misplaced_count` to 0.
        2. Initialize a memoization dictionary `memo` for the recursive
           `_is_block_correctly_stacked` function.
        3. Find the block currently being held, if any, by iterating through
           the state facts.
        4. If a block is being held (`held_block` is not None) and it is one
           of the blocks relevant to the goal stacks (`goal_blocks`), increment
           `misplaced_count`. A held block is not in its final position within
           a stack.
        5. For each block `b` in the set of `goal_blocks`:
           - If `b` is not the block currently being held:
             - Recursively check if `b` is correctly stacked in the current
               state according to the goal structure using `_is_block_correctly_stacked`.
             - If `_is_block_correctly_stacked` returns False, increment
               `misplaced_count`.
        6. For each goal fact `(clear X)`:
           - If `(clear X)` is not true in the current state, increment
             `misplaced_count`. This penalizes blocks that are blocking
             a location that needs to be clear according to the goal.
        7. If `(arm-empty)` is a goal fact and `(arm-empty)` is not true
           in the current state, increment `misplaced_count`. This penalizes
           the arm being occupied when it needs to be free.
        8. Return the final `misplaced_count`.

    Recursive Helper Function `_is_block_correctly_stacked(block, state, goal_below_map, memo)`:
        - Checks if `block` is in its correct goal position relative to the
          block below it AND if the entire stack below it is also correctly
          stacked.
        - Uses memoization to avoid redundant calculations for the same block
          within a single heuristic call.
        - Base case: If `block` should be on the table in the goal, it's
          correctly stacked (at its base) if it is on the table in the state.
        - Recursive step: If `block` should be on block `Y` in the goal, it's
          correctly stacked if it is on `Y` in the state AND `Y` is correctly
          stacked. It assumes `Y` is also a block relevant to the goal stacks.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        # Store goal facts as a set for fast lookup
        self.goal_facts = set(task.goals)

        # Parse goal facts to build goal_below_map and collect goal_blocks
        self.goal_below_map = {}
        self.goal_blocks = set()
        for fact_str in self.goal_facts:
            parsed = self._parse_fact(fact_str)
            if parsed and parsed[0] == 'on':
                # Fact is like '(on b1 b2)'
                _, block, below_block = parsed
                self.goal_below_map[block] = below_block
                self.goal_blocks.add(block)
                self.goal_blocks.add(below_block)
            elif parsed and parsed[0] == 'on-table':
                # Fact is like '(on-table b2)'
                _, block = parsed
                self.goal_below_map[block] = 'table'
                self.goal_blocks.add(block)
            # Ignore clear and arm-empty goals for building the stack structure map

        # Ensure all blocks mentioned as values (if not 'table') are in goal_blocks
        # This handles cases where a block is only mentioned as being below another
        # but not explicitly in an 'on-table' or 'on' fact as the top block.
        # (e.g., goal is (on A B), B is not mentioned elsewhere. B should be on table by default
        # in many blocksworld variants, but our map only includes explicit goals.
        # We assume standard blocksworld where all blocks have a goal position).
        # The initial population from 'on' and 'on-table' facts should cover this
        # for standard problems.

    def _parse_fact(self, fact_str):
        """Helper to parse a fact string into predicate and arguments."""
        # Remove surrounding parentheses and split by space
        parts = fact_str.strip('()').split()
        if not parts:
            return None
        return parts

    def _is_block_correctly_stacked(self, block, state, goal_below_map, memo):
        """
        Recursive helper to check if a block is correctly stacked in the state.
        Uses memoization.
        Assumes 'block' is a key in goal_below_map.
        """
        if block in memo:
            return memo[block]

        target_below = goal_below_map[block]

        if target_below == 'table':
            # Base case: block should be on the table
            result = '(on-table {})'.format(block) in state
        else:
            # Recursive step: block should be on target_below
            on_fact = '(on {} {})'.format(block, target_below)
            # Check if block is on the correct block AND the block below is correctly stacked
            # We assume target_below is also in goal_below_map if it's not 'table'
            # because it must be part of a goal stack itself.
            result = (on_fact in state) and \
                     self._is_block_correctly_stacked(target_below, state, goal_below_map, memo)

        memo[block] = result
        return result

    def __call__(self, node):
        """
        Computes the blocksworld heuristic for the given state.
        """
        state = node.state # state is a frozenset of fact strings

        # Check if goal is reached - heuristic is 0 only at goal
        if self.task.goal_reached(state):
             return 0

        misplaced_count = 0
        memo = {} # Memoization for recursive calls within this state evaluation

        # Find the block being held
        held_block = None
        for fact_str in state:
            parsed = self._parse_fact(fact_str)
            if parsed and parsed[0] == 'holding':
                held_block = parsed[1]
                break # Only one block can be held at a time

        # 1. Count blocks that are not correctly stacked (among goal blocks)
        # Iterate over blocks that are part of the goal stacks
        for block in self.goal_blocks:
            if block == held_block:
                # A held block is considered misplaced relative to its goal stack position
                # unless the goal is specifically to hold it (unlikely in standard BW)
                # and it's the correct block. We simplify and count held goal blocks as misplaced.
                misplaced_count += 1
            elif block in self.goal_below_map: # Ensure block has a specified goal position
                # Check if the block is in its correct goal stack position
                if not self._is_block_correctly_stacked(block, state, self.goal_below_map, memo):
                    misplaced_count += 1
            # else: block is in goal_blocks but not goal_below_map. This implies
            # it's a block that something else should be on top of, but it itself
            # doesn't have a specified position below it in the goal map.
            # This case shouldn't contribute to 'incorrectly stacked' count for itself.

        # 2. Add penalty for unsatisfied clear goals
        for goal_fact_str in self.goal_facts:
            parsed = self._parse_fact(goal_fact_str)
            if parsed and parsed[0] == 'clear':
                # Check if the clear goal is not satisfied in the state
                if goal_fact_str not in state:
                    misplaced_count += 1

        # 3. Add penalty for unsatisfied arm-empty goal
        if '(arm-empty)' in self.goal_facts and '(arm-empty)' not in state:
             misplaced_count += 1

        return misplaced_count

