# Import necessary classes
from heuristics.heuristic_base import Heuristic
from task import Task # Although Task is not directly used in __call__, it's used in __init__ and part of the expected environment.

class blocksworldHeuristic(Heuristic):
    """
    Summary:
    A domain-dependent heuristic for the Blocksworld domain.
    It estimates the number of actions required to reach the goal state by
    counting several types of discrepancies between the current state and the
    goal state:
    1. Blocks that are part of the goal configuration but are currently on the wrong base (another block or the table).
    2. Blocks that are currently obstructing a position needed for a goal stack or a goal clearance.
    3. Blocks that are required to be clear in the goal state but are not clear in the current state.
    4. The arm is holding a block when the goal requires it to be empty.

    Assumptions:
    - Assumes standard Blocksworld predicates: (on ?x ?y), (on-table ?x), (clear ?x), (holding ?x), (arm-empty).
    - Assumes object names are simple strings (e.g., 'b1', 'b2').
    - Assumes the goal state is defined by a set of facts that must be true (superset semantics).

    Heuristic Initialization:
    In the constructor, the heuristic precomputes information about the goal
    configuration to efficiently evaluate states during the search. This includes:
    - Identifying all blocks present in the problem instance by parsing all possible ground facts.
    - Building maps (`goal_below`, `goal_above`) representing the desired stacking
      relationships specified in the goal facts (e.g., `goal_below[B] = A` if `(on B A)`
      is a goal, `goal_below[B] = 'table'` if `(on-table B)` is a goal).
    - Identifying the set of blocks that must be clear in the goal state (`goal_clear_blocks`).
    - Checking if the goal requires the arm to be empty (`arm_empty_goal`).

    Step-By-Step Thinking for Computing Heuristic:
    For a given state, the heuristic value is computed as the sum of four components:

    1.  Misplaced Goal Blocks: Iterate through all blocks. If a block `B` is part of
        the goal configuration (i.e., `goal_below[B]` is defined) and its current base
        (`current_below[B]`) is different from its goal base (`goal_below[B]`), increment the heuristic count.
        This captures blocks that are in the wrong place relative to what they should be directly on top of.

    2.  Obstructing Blocks: Iterate through all blocks `B`. If `B` is currently on top
        of another block `A` (`current_below[B] == A` where `A` is a block):
        Check if `A` needs to be clear for a goal (i.e., `A` is in `goal_clear_blocks`)
        OR if `A` is supposed to have a *different* block on top according to the goal
        (i.e., `goal_above[A]` is defined and `goal_above[A] != B`).
        If either condition is true, block `B` is obstructing a goal-relevant position,
        so increment the heuristic count.

    3.  Unsatisfied Clear Goals: Iterate through all blocks `B`. If `(clear B)` is a
        goal fact but `(clear B)` is not true in the current state, increment the
        heuristic count.

    4.  Unsatisfied Arm-Empty Goal: If `(arm-empty)` is a goal fact but `(arm-empty)`
        is not true in the current state, increment the heuristic count.

    The total heuristic value is the sum of counts from these four parts.
    A special check is performed first: if the current state is exactly the goal state
    (according to the planner's superset definition), the heuristic is 0. This ensures
    the heuristic is 0 only at the goal.
    """

    def __init__(self, task: Task):
        """
        Initializes the Blocksworld heuristic.

        Args:
            task: The planning task.
        """
        super().__init__()
        self.goals = task.goals
        self.all_blocks = self._extract_all_blocks(task)
        self.goal_below = self._build_below_map(task.goals)
        self.goal_above = self._build_above_map(task.goals)
        self.goal_clear_blocks = self._extract_clear_blocks(task.goals)
        self.arm_empty_goal = '(arm-empty)' in task.goals

    def _parse_fact(self, fact_str: str) -> list[str]:
        """Helper to parse a fact string into predicate and arguments."""
        # Removes leading/trailing parens and splits by space
        return fact_str[1:-1].split()

    def _build_below_map(self, fact_set: frozenset[str]) -> dict[str, str]:
        """
        Builds a map from block to the object directly below it ('table' or another block).
        Includes 'holding' state representation.
        """
        below_map = {}
        for fact_str in fact_set:
            parts = self._parse_fact(fact_str)
            if parts[0] == 'on':
                block = parts[1]
                below = parts[2]
                below_map[block] = below
            elif parts[0] == 'on-table':
                block = parts[1]
                below_map[block] = 'table'
            elif parts[0] == 'holding':
                 block = parts[1]
                 below_map[block] = 'holding' # Represents the state where block is held
        return below_map

    def _build_above_map(self, fact_set: frozenset[str]) -> dict[str, str]:
        """
        Builds a map from object (block or 'table') to the block directly above it.
        Note: In Blocksworld, only one block can be directly on another block.
        """
        above_map = {}
        for fact_str in fact_set:
            parts = self._parse_fact(fact_str)
            if parts[0] == 'on':
                above = parts[1]
                below = parts[2]
                above_map[below] = above
            # on-table facts don't define what's *above* something else
        return above_map

    def _extract_blocks(self, fact_set: frozenset[str]) -> set[str]:
        """Extracts all unique block names from a set of facts."""
        blocks = set()
        for fact_str in fact_set:
            parts = self._parse_fact(fact_str)
            # Predicates whose arguments are blocks
            if parts[0] in ['on', 'on-table', 'clear', 'holding']:
                 # Arguments are blocks (except 'arm-empty' which has no args)
                 for part in parts[1:]:
                     blocks.add(part)
        return blocks

    def _extract_all_blocks(self, task: Task) -> frozenset[str]:
        """Extracts all unique block names from all possible ground facts in the task."""
        all_blocks = set()
        # Assuming task.facts contains all possible ground fact strings
        for fact_str in task.facts:
            parts = self._parse_fact(fact_str)
            # Predicates whose arguments are blocks
            if parts[0] in ['on', 'on-table', 'clear', 'holding']:
                 # Arguments are blocks (except 'arm-empty' which has no args)
                 for part in parts[1:]:
                     all_blocks.add(part)
        return frozenset(all_blocks) # Use frozenset for immutability

    def _extract_clear_blocks(self, fact_set: frozenset[str]) -> set[str]:
        """Extracts all unique block names that are stated as clear in a set of facts."""
        clear_blocks = set()
        for fact_str in fact_set:
            parts = self._parse_fact(fact_str)
            if parts[0] == 'clear':
                clear_blocks.add(parts[1])
        return clear_blocks

    def __call__(self, node) -> int:
        """
        Computes the domain-dependent heuristic value for the given state.

        Args:
            node: The search node containing the state.

        Returns:
            The heuristic value (an integer).
        """
        state = node.state

        # Heuristic is 0 only for goal states
        if self.goals <= state:
             return 0

        h_value = 0

        # Build current state maps and sets
        current_below = self._build_below_map(state)
        current_above = self._build_above_map(state)
        current_clear_blocks = self._extract_clear_blocks(state)
        arm_empty_state = '(arm-empty)' in state

        # Part 1: Misplaced Goal Blocks
        # Count blocks that are part of the goal config but are on the wrong base
        for block in self.all_blocks:
            target_base = self.goal_below.get(block)
            if target_base is not None: # Block is part of the goal configuration
                current_base = current_below.get(block)
                if current_base != target_base:
                    h_value += 1

        # Part 2: Obstructing Blocks
        # Count blocks that are on top of a block that needs to be clear or needs a different block on top
        for obstructing_block in self.all_blocks:
            base_block = current_below.get(obstructing_block)
            # Check if the block is currently on top of another block (not table or holding)
            if base_block is not None and base_block != 'table' and base_block != 'holding':
                # Check if the base block needs to be clear for a goal
                needs_base_clear = base_block in self.goal_clear_blocks
                # Check if the base block is supposed to have a different block on top
                needs_different_above = (
                    self.goal_above.get(base_block) is not None and
                    self.goal_above.get(base_block) != obstructing_block
                )

                if needs_base_clear or needs_different_above:
                    h_value += 1

        # Part 3: Unsatisfied Clear Goals
        # Count blocks that should be clear but aren't
        for block in self.all_blocks:
            if block in self.goal_clear_blocks and block not in current_clear_blocks:
                 h_value += 1

        # Part 4: Unsatisfied Arm-Empty Goal
        # Add cost if arm should be empty but isn't
        if self.arm_empty_goal and not arm_empty_state:
             h_value += 1

        return h_value
