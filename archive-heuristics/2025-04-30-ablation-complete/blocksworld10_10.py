from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by counting blocks that are not in their correct goal position relative to their
    support, and blocks that are obstructing wrongly positioned blocks.

    # Assumptions
    - The cost of moving a block to its correct support position is approximately 2 actions
      (e.g., unstack/pickup + stack/putdown).
    - The cost of moving an obstructing block out of the way is also approximately 2 actions
      (e.g., unstack + putdown).
    - The heuristic does not account for the arm state directly, assuming the necessary
      arm state can be achieved with minimal cost within the block movements.

    # Heuristic Initialization
    - Parses the goal state to determine the desired support for each block involved
      in the goal (`goal_support` map).
    - Identifies all blocks that are part of the goal structure (`goal_blocks` set).
    - Identifies all blocks present in the initial state (`all_blocks` set) to consider
      potential obstructions by non-goal blocks.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state to determine the current support for each block
       (`current_support` map) and identify all `(on A B)` facts (`current_on_facts` set).
       The support can be another block, the table ('table'), or the arm ('holding').
    2. Identify the set of `WronglySupported` blocks: These are blocks that are part
       of the goal state structure (i.e., in `goal_blocks`) but whose current support
       (on another block, on the table, or being held) does not match their
       desired support in the goal state (`current_support[B] != goal_support[B]`).
       Blocks mentioned only in `(clear ?x)` goals are not considered wrongly supported
       based on their support, as their goal support is implicitly 'nothing above'.
    3. Identify the set of `ObstructingBlocks`: These are blocks `A` that are currently
       on top of a block `B` (`(on A B)` is in `current_on_facts`) where the block `B`
       is in the `WronglySupported` set. These blocks `A` must be moved out of the way
       before `B` can be repositioned.
    4. The heuristic value is calculated as `2 * |WronglySupported| + 2 * |ObstructingBlocks|`.
       Each wrongly supported block needs to be moved (approx. 2 actions). Each obstructing
       block also needs to be moved (approx. 2 actions) to clear the block below it.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and all blocks.
        """
        self.goals = task.goals

        # Build goal_support map and identify goal_blocks
        self.goal_support = {}
        self.goal_blocks = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'on':
                block, support = parts[1], parts[2]
                self.goal_support[block] = support
                self.goal_blocks.add(block)
                self.goal_blocks.add(support) # Support block is also part of the goal structure
            elif parts[0] == 'on-table':
                block = parts[1]
                self.goal_support[block] = 'table'
                self.goal_blocks.add(block)
            # Ignore 'clear' and 'arm-empty' goals for building goal_support map

        # Identify all blocks present in the initial state.
        # This set is used to iterate through all possible blocks that might be
        # wrongly supported or obstructing.
        self.all_blocks = set()
        for fact in task.initial_state:
            parts = get_parts(fact)
            # Consider predicates that involve blocks
            if parts[0] in ['on', 'on-table', 'clear', 'holding']:
                 # Add all parameters as blocks (assuming parameters are blocks)
                 for obj in parts[1:]:
                     self.all_blocks.add(obj)
            # Add blocks from goal that might not be explicitly in initial state facts
            # (e.g., only mentioned in a 'clear' goal).
            self.all_blocks.update(self.goal_blocks)


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # Build current_support map and current_on_facts set
        current_support = {}
        current_on_facts = set() # Store (A, B) tuples for (on A B) facts
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                block, support = parts[1], parts[2]
                current_support[block] = support
                current_on_facts.add((block, support))
            elif parts[0] == 'on-table':
                block = parts[1]
                current_support[block] = 'table'
            elif parts[0] == 'holding':
                block = parts[1]
                current_support[block] = 'holding'
            # Ignore 'clear' and 'arm-empty' facts for support mapping

        # Compute WronglySupported blocks
        WronglySupported = set()
        # Iterate through all blocks that are part of the goal structure
        for block in self.goal_blocks:
            # Get current support. Default to None if block somehow not in state (shouldn't happen in BW).
            current_sup = current_support.get(block)
            # Get goal support. Default to None if block is in goal_blocks but not goal_support (e.g. only in clear goal).
            goal_sup = self.goal_support.get(block)

            # A block is wrongly supported if it's explicitly assigned a support
            # in the goal (via 'on' or 'on-table') AND its current support
            # does not match that goal support.
            if block in self.goal_support and current_sup != goal_sup:
                 WronglySupported.add(block)

        # Compute ObstructingBlocks
        ObstructingBlocks = set()
        # Iterate through all blocks that are currently on top of another block
        for A, B in current_on_facts:
            # If the block below (B) is wrongly supported, then A is obstructing it
            if B in WronglySupported:
                ObstructingBlocks.add(A)

        # Heuristic value: 2 actions for each wrongly supported block + 2 actions for each obstructing block
        heuristic_value = 2 * len(WronglySupported) + 2 * len(ObstructingBlocks)

        # The heuristic should be 0 in the goal state.
        # If the state is a goal state, all goal predicates are true.
        # For any block B in goal_blocks with a goal_support, current_support[B] will equal goal_support[B].
        # Thus, WronglySupported will be empty.
        # If WronglySupported is empty, no block B exists in WronglySupported, so ObstructingBlocks will also be empty.
        # The heuristic value will be 2 * 0 + 2 * 0 = 0.

        return heuristic_value
