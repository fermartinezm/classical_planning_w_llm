from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Removes parentheses and splits a PDDL fact string into parts."""
    # Example: '(on b1 b2)' -> ['on', 'b1', 'b2']
    # Example: '(arm-empty)' -> ['arm-empty']
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for Blocksworld.

    Summary:
    This heuristic estimates the cost to reach the goal state by identifying
    blocks that are not in their correct position relative to the block below
    them in the goal configuration, and blocks that are obstructing other blocks
    by being on top of them when they shouldn't be according to the goal.
    Each such block is estimated to require at least two actions (pickup/unstack
    and putdown/stack) to move it out of the way or into its correct position.
    An additional cost of 1 is added if the arm is currently holding a block,
    as this block must be put down before other operations can occur.

    Assumptions:
    - The input task and state are valid for the Blocksworld domain as defined.
    - Goal predicates primarily consist of 'on', 'on-table', and 'clear'.
    - The goal state is achievable.
    - Goal predicates are consistent (e.g., if (on A B) is a goal, (clear B) is not).
    - The representation of facts in the state and goal is a frozenset of strings
      like '(predicate arg1 arg2)'.

    Heuristic Initialization:
    During initialization (`__init__`), the heuristic processes the goal predicates
    provided in the task object. It constructs a dictionary `goal_config` mapping
    each block that appears in a goal `(on X Y)` or `(on-table X)` predicate to
    its desired base (either another block Y or the string 'table'). It also stores
    the set of all goal `(on X Y)` predicates in `goal_on_predicates` for efficient
    lookup later.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Parse the current state: Iterate through the facts in the current state.
        Identify `(on X Y)` facts to build `current_config` (mapping X to Y) and
        collect all current `(on X Y)` facts. Identify `(on-table X)`
        facts to add to `current_config` (mapping X to 'table'). Identify the
        block being held from the `(holding X)` fact, if any.
    2.  Check for goal state: If the current state is the goal state, the heuristic
        value is 0.
    3.  Calculate cost for blocks on the wrong base: Iterate through the blocks
        that have a specified goal position (i.e., are keys in `goal_config`).
        If a block is currently held, it's on the wrong base. If it's not held,
        check its base in `current_config`. If it's in `current_config` and its
        current base is different from its goal base, it's on the wrong base.
        Add 2 to the heuristic for each such block (representing the estimated
        cost of picking it up/unstacking and then stacking/putting it down later).
    4.  Calculate cost for obstructing blocks: Iterate through all `(on T B)` facts
        in the current state. If the fact `(on T B)` is *not* present in the set
        of goal `(on X Y)` predicates (`goal_on_predicates`), it means block T is
        currently on top of block B, but this specific arrangement is not desired
        in the goal. Block T is considered an obstruction. Add 2 to the heuristic
        for each such obstructing block T (representing the estimated cost of
        moving it out of the way).
    5.  Calculate cost for held block: If the arm is currently holding a block
        (`holding` is not None), add 1 to the heuristic. This represents the cost
        of putting the block down to free the arm for other operations.
    6.  Return the total calculated heuristic value.
    """

    def __init__(self, task):
        self.goals = task.goals

        self.goal_config = {}
        self.goal_on_predicates = set()

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            pred = parts[0]
            if pred == 'on':
                block_on = parts[1]
                block_below = parts[2]
                self.goal_config[block_on] = block_below
                self.goal_on_predicates.add(goal_fact)
            elif pred == 'on-table':
                block_on_table = parts[1]
                self.goal_config[block_on_table] = 'table'
            # clear and arm-empty goals are implicitly handled by the heuristic logic

    def __call__(self, node):
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Parse current state
        current_config = {}
        holding = None

        # Collect current ON and ON-TABLE facts
        current_on_facts = set()
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == 'on':
                block_on = parts[1]
                block_below = parts[2]
                current_config[block_on] = block_below
                current_on_facts.add(fact)
            elif pred == 'on-table':
                block_on_table = parts[1]
                current_config[block_on_table] = 'table'
            elif pred == 'holding':
                holding = parts[1]

        h = 0

        # 1. Count blocks on the wrong goal base
        wrong_base_blocks = set()
        for block in self.goal_config: # Iterate through blocks that have a goal position
            if holding == block:
                 # If the block is held, it's definitely not on its goal base
                 wrong_base_blocks.add(block)
            elif block in current_config:
                if current_config[block] != self.goal_config[block]:
                     wrong_base_blocks.add(block)
            # If block is in goal_config but not in state config and not held,
            # it implies the block is missing from the state, which shouldn't happen
            # in valid blocksworld states derived from an initial state containing all objects.

        h += 2 * len(wrong_base_blocks)

        # 2. Count obstructing blocks on top
        obstructions = set()
        for on_fact_str in current_on_facts:
            # If this (on T B) relation is NOT a goal ON predicate, T is an obstruction
            if on_fact_str not in self.goal_on_predicates:
                parts = get_parts(on_fact_str)
                block_on = parts[1]
                obstructions.add(block_on)

        h += 2 * len(obstructions)

        # 3. Add cost if arm is holding a block
        if holding is not None:
            h += 1

        return h
