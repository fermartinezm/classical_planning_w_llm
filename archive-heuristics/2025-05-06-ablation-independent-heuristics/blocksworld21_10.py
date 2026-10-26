import logging

from heuristics.heuristic_base import Heuristic
from task import Operator, Task

_LOGGER = logging.getLogger(__name__)


class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the distance to the goal state by counting
        the number of unsatisfied goal conditions related to the position
        of blocks (whether they are on the correct block or the table) and
        whether blocks required to be clear are indeed clear. It sums these
        two counts.

    Assumptions:
        - The domain is Blocksworld with standard predicates: (on ?x ?y),
          (on-table ?x), (clear ?x), (holding ?x), (arm-empty).
        - Goal states are defined using conjunctions of (on ?x ?y),
          (on-table ?x), and (clear ?x) facts.
        - The state is represented as a frozenset of fact strings.
        - Fact strings are in the format '(predicate arg1 arg2 ...)', e.g.,
          '(on b1 b2)', '(on-table b3)', '(clear b4)'.

    Heuristic Initialization:
        In the constructor, the heuristic pre-processes the goal facts
        from the task definition. It identifies:
        - `self.goal_pos`: A dictionary mapping each block that is required
          to be on another block or the table in the goal state to its
          required base (either the name of the block below it or the string
          'table').
        - `self.goal_clear`: A set containing the names of all blocks that
          are required to be clear in the goal state.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state `s`:
        1. Initialize the heuristic value `h` to 0.
        2. Iterate through each block `block` and its required base `base`
           in `self.goal_pos`:
           a. If `base` is the string 'table':
              Check if the fact `'(on-table {})'.format(block)` is present
              in the state `s`. If it is NOT present, increment `h` by 1.
           b. If `base` is a block name:
              Check if the fact `'(on {} {})'.format(block, base)` is present
              in the state `s`. If it is NOT present, increment `h` by 1.
        3. Iterate through each block `block` in `self.goal_clear`:
           Check if the fact `'(clear {})'.format(block)` is present in the
           state `s`. If it is NOT present, increment `h` by 1.
        4. The final value of `h` is the heuristic estimate for the state `s`.

        This heuristic counts how many blocks are not in their correct goal
        position relative to the block/table immediately below them in the
        goal configuration, plus how many blocks that should be clear are not.
        It is 0 if and only if all these specific goal conditions are met,
        which corresponds to the goal state. It is efficiently computable
        as it only requires iterating through the pre-processed goal facts
        and checking for the presence of corresponding facts in the state set.
    """

    def __init__(self, task: Task):
        """
        Initializes the blocksworld heuristic by processing the goal facts.

        Args:
            task: The planning task object.
        """
        super().__init__()
        self.goals = task.goals  # Store goal facts for easy access

        # Pre-process goal facts
        self.goal_pos = {}  # Maps block -> block_below or 'table'
        self.goal_clear = set() # Set of blocks that must be clear

        for goal_fact_str in self.goals:
            # Parse the fact string: remove outer parens and split
            parts = goal_fact_str[1:-1].split()
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'on':
                if len(args) == 2:
                    block, base = args
                    self.goal_pos[block] = base
                else:
                    _LOGGER.warning(f"Unexpected 'on' fact format in goal: {goal_fact_str}")
            elif predicate == 'on-table':
                if len(args) == 1:
                    block = args[0]
                    self.goal_pos[block] = 'table'
                else:
                     _LOGGER.warning(f"Unexpected 'on-table' fact format in goal: {goal_fact_str}")
            elif predicate == 'clear':
                if len(args) == 1:
                    block = args[0]
                    self.goal_clear.add(block)
                else:
                    _LOGGER.warning(f"Unexpected 'clear' fact format in goal: {goal_fact_str}")
            # Ignore other goal predicates if any (like arm-empty, holding)
            # as they are usually transient and not primary goals in BW.
            # The current heuristic focuses on block positions and clearness.


    def __call__(self, node) -> int:
        """
        Computes the heuristic value for the given state.

        Args:
            node: The search node containing the state.

        Returns:
            The estimated number of actions to reach the goal state.
        """
        state = node.state
        h_value = 0

        # Part 1: Count blocks not on their correct goal base (block or table)
        for block, base in self.goal_pos.items():
            if base == 'table':
                required_fact = f'(on-table {block})'
                if required_fact not in state:
                    h_value += 1
            else: # base is another block
                required_fact = f'(on {block} {base})'
                if required_fact not in state:
                    h_value += 1

        # Part 2: Count blocks that should be clear but are not
        for block in self.goal_clear:
            required_fact = f'(clear {block})'
            if required_fact not in state:
                 # Check if it's not clear because something is on it.
                 # We don't need to explicitly check for (on ? block) facts
                 # because if (clear block) is not in the state, it means
                 # something is on it (or it's being held, but clear applies
                 # to the block itself, not the arm). The absence of (clear block)
                 # is sufficient evidence that the condition is not met.
                 h_value += 1

        return h_value

