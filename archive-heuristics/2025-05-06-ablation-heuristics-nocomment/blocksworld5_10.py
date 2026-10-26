from typing import Set, Dict, Optional, Any, FrozenSet

# Assuming a base class for heuristics exists as per example usage
# from heuristics.heuristic_base import Heuristic

# Define a dummy base class if the actual one is not available
# This allows the code to be syntactically correct without the external dependency
class Heuristic:
    def __init__(self, task: Any):
        pass

    def __call__(self, node: Any) -> int:
        raise NotImplementedError

def get_parts(fact: str) -> list[str]:
    """Helper function to parse a PDDL fact string into its components."""
    # Remove parentheses and split by spaces
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Estimates the number of actions required to reach the goal state by
    counting misplaced blocks and blocks that are blocking desired placements.
    Designed for greedy best-first search, so admissibility is not required.
    The heuristic is 0 if and only if the state is a goal state.
    """

    def __init__(self, task: Any):
        """
        Initializes the heuristic with goal information.

        Heuristic Initialization:
        Parses the goal predicates from the task to build data structures
        representing the desired final configuration of blocks.
        - `goal_on`: A dictionary mapping each block to the block it should
          be directly on top of in the goal state, or 'table' if it should
          be on the table.
        - `goal_child`: A dictionary mapping each block to the block that
          should be directly on top of it in the goal state. If a block
          should be clear (nothing on top) in the goal, its entry will be None.
        - `goal_blocks`: A set containing all blocks explicitly mentioned
          in the goal `on` or `on-table` predicates.
        - `goal_arm_empty`: A boolean indicating whether the arm should be
          empty in the goal state. Assumed True unless a `(holding ?x)`
          predicate is explicitly in the goal.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        super().__init__(task)
        self.goals: FrozenSet[str] = task.goals

        # Parse goal predicates to build goal structure mappings
        self.goal_on: Dict[str, str] = {} # Maps block -> block it should be on, or 'table'
        self.goal_child: Dict[str, Optional[str]] = {} # Maps block -> block that should be on it
        self.goal_blocks: Set[str] = set() # Set of blocks mentioned in goal (on or on-table)
        self.goal_arm_empty: bool = True # Assume arm-empty is goal unless holding is specified

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                block, base = parts[1], parts[2]
                self.goal_on[block] = base
                self.goal_child[base] = block
                self.goal_blocks.add(block)
                self.goal_blocks.add(base)
            elif predicate == "on-table":
                block = parts[1]
                self.goal_on[block] = 'table'
                self.goal_blocks.add(block)
            elif predicate == "holding":
                 # If goal specifies holding, arm-empty is not the goal
                 self.goal_arm_empty = False
                 # Note: This heuristic primarily focuses on structural goals
                 # and the arm-empty state. Goals requiring holding are not
                 # specifically optimized for beyond acknowledging the arm-empty
                 # goal is absent.

        # For blocks that are bases in the goal structure but have no block
        # specified to be on them, they should be clear in the goal.
        # Ensure they have an entry in goal_child mapping to None.
        all_goal_bases = set(self.goal_on.values()) - {'table'}
        for block in all_goal_bases:
             if block not in self.goal_child:
                 self.goal_child[block] = None # Should be clear

        # Blocks that are in goal_on keys but not values are the top blocks in goal stacks.
        # They should also have no block on them in the goal.
        # This is implicitly handled by the blocking count logic later, but we can add them
        # to goal_child explicitly for completeness if needed, though it doesn't change the logic.
        # all_goal_children = set(self.goal_child.values()) - {None}
        # for block in self.goal_blocks:
        #     if block in self.goal_on and block not in all_goal_children:
        #          if block not in self.goal_child:
        #              self.goal_child[block] = None


    def __call__(self, node: Any) -> int:
        """
        Computes the heuristic value for the given state.

        Step-By-Step Thinking for Computing Heuristic:
        1. Parse the current state to understand the current configuration
           of blocks (which block is on which, which are on the table,
           which are clear, and which is being held). This information is
           stored in `current_on`, `current_on_table`, `current_clear`,
           `current_holding`, and derived maps `current_base` and `current_child`.
           Also identifies all blocks present in the current state.
        2. Calculate the "misplaced blocks" count: Iterate through all blocks
           that are part of the goal structure (`self.goal_blocks`). For each
           such block, check if its current base (the block it's on, or the
           table) is the same as its desired goal base (`self.goal_on`). If
           they are different, increment the heuristic count. This counts
           blocks that are in the wrong location relative to what's directly
           below them.
        3. Calculate the "blocking blocks" count: Iterate through all blocks
           that are currently acting as bases (i.e., have another block on
           top of them, or are on the table). For each such base block, check
           if the block currently on top of it (`current_child`) is the same
           as the block that should be on top of it according to the goal
           structure (`goal_child`). If there is a block on top in the current
           state, and it's *not* the correct block (which includes the case
           where the base block should be clear in the goal), increment the
           heuristic count. This counts blocks that are obstructing the correct
           configuration of the blocks below them.
        4. Calculate the "arm penalty": If the goal requires the arm to be empty
           (`self.goal_arm_empty` is True) but the arm is currently holding a
           block (`current_holding` is not None), increment the heuristic count
           by 1, as a `putdown` action is needed.
        5. The total heuristic value is the sum of the misplaced blocks count,
           the blocking blocks count, and the arm penalty.

        Args:
            node: The search node containing the state (a frozenset of fact strings).

        Returns:
            The estimated number of actions to reach the goal (an integer >= 0).
        """
        state: FrozenSet[str] = node.state

        # 1. Parse current state predicates
        current_on: Dict[str, str] = {} # Maps block -> block it is on
        current_on_table: Set[str] = set() # Set of blocks on the table
        current_clear: Set[str] = set() # Set of clear blocks
        current_holding: Optional[str] = None # Block being held, or None
        all_state_blocks: Set[str] = set() # All blocks mentioned in state facts

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                block, base = parts[1], parts[2]
                current_on[block] = base
                all_state_blocks.add(block)
                all_state_blocks.add(base)
            elif predicate == "on-table":
                block = parts[1]
                current_on_table.add(block)
                all_state_blocks.add(block)
            elif predicate == "clear":
                block = parts[1]
                current_clear.add(block)
                all_state_blocks.add(block)
            elif predicate == "holding":
                block = parts[1]
                current_holding = block
                all_state_blocks.add(block)
            # arm-empty is implicitly handled by checking current_holding

        # Build current structure mappings
        current_base: Dict[str, str] = {} # Maps block -> block it is on, or 'table'
        current_child: Dict[str, Optional[str]] = {} # Maps block -> block that is on it (or None)

        # Populate current_base and current_child from current_on and current_on_table
        for block in all_state_blocks:
            if block in current_on:
                current_base[block] = current_on[block]
                current_child[current_on[block]] = block # Inverse mapping
            elif block in current_on_table:
                current_base[block] = 'table'

        # Ensure all blocks that are bases in current_on or on_table have a child entry (None if no child)
        # Iterate through all blocks that appear as bases in the state
        current_bases_set = set(current_on.values()) | current_on_table
        for base in current_bases_set:
             if base not in current_child:
                 current_child[base] = None # This base block is clear in the current state


        h = 0

        # 2. Calculate misplaced blocks count
        # Only consider blocks that are part of the goal structure
        for block in self.goal_blocks:
            goal_base = self.goal_on.get(block)
            current_base_of_block = current_base.get(block) # Will be None if block not in state

            # If the block is in the goal structure and its current base is not its goal base
            if goal_base is not None and current_base_of_block != goal_base:
                 h += 1

        # 3. Calculate blocking blocks count
        # Iterate through all blocks that are bases in the current state
        # (i.e., have something on them or are on the table)
        for base_block in current_bases_set:
             current_child_of_base = current_child.get(base_block) # Block currently on base_block, or None
             goal_child_of_base = self.goal_child.get(base_block) # Block that should be on base_block, or None

             # If there is a block on base_block in the current state,
             # and it's not the block that should be there according to the goal
             if current_child_of_base is not None and current_child_of_base != goal_child_of_base:
                 h += 1

        # 4. Calculate arm penalty
        if self.goal_arm_empty and current_holding is not None:
            h += 1

        # The heuristic is 0 if and only if the state is a goal state.
        # If the state is a goal state, all goal predicates are true.
        # This implies:
        # - For every block in goal_blocks, its current_base matches its goal_base (misplaced count is 0).
        # - For every block that is a base in the goal, its current_child matches its goal_child (blocking count is 0 for these).
        # - For blocks that are bases in the state but not in the goal structure, they must be clear if the goal implies they should be clear (e.g., if they are not part of any goal stack and arm-empty is goal). The blocking count handles this.
        # - If goal_arm_empty is true, current_holding must be None (arm penalty is 0).
        # Conversely, if h=0, all counts are 0, implying the state matches the goal structure and arm state.

        return h
