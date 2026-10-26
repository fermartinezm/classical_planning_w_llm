# Assume Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(on b1 b2)" -> ["on", "b1", "b2"]
    return fact[1:-1].split()

# Define the heuristic class inheriting from Heuristic
class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by counting blocks that are in the wrong position relative to their goal
    support, blocks that are obstructing others from reaching their goal
    position, and whether the arm needs to be empty according to the goal.
    It is designed for greedy best-first search and does not need to be admissible.

    # Assumptions:
    - The goal state is defined by a set of (on ?x ?y) and (on-table ?x) facts,
      and possibly (clear ?x) and (arm-empty).
    - Blocks relevant to the goal are mentioned in the goal facts.
    - A block is always either on the table, on another block, or being held.
    - The state representation is a frozenset of PDDL fact strings.

    # Heuristic Initialization
    - Parses the goal facts from the task to build:
        - A map `goal_structure` where keys are blocks and values are their
          desired support (the block or 'table' it should be directly on top of
          in the goal).
        - A set `goal_facts_set` containing all goal facts for quick lookup.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic cost `cost` to 0.
    2. Analyze the current state to determine:
       - The current support for each block (what it's directly on top of,
         or 'table', or 'holding'). Store this in `current_structure`.
       - If the robot arm is holding any block.
       - The set of all `(on Y B)` facts currently true.
    3. Count "misplaced" blocks: Iterate through the `goal_structure`. For each
       block and its `desired_below` support in the goal, check its `current_below`
       support in the state. If `current_below` is different from `desired_below`
       (or if the block's position isn't recorded in the state, which indicates
       an issue or it's not on table/block/held), increment `cost`.
    4. Count "obstructing" blocks: Iterate through the `(on Y B)` facts in the
       current state. If an `(on Y B)` fact is NOT present in the `goal_facts_set`,
       it means block Y is obstructing block B relative to the goal configuration.
       Increment `cost` for each such obstructing fact.
    5. Check the arm state goal: If `(arm-empty)` is a goal fact AND the robot
       arm is currently holding a block, increment `cost`.
    6. The total heuristic value for the state is the final value of `cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        # Store goals as a set for fast lookup
        self.goal_facts_set = set(task.goals)

        # Build the goal structure: map block -> block_below_in_goal or 'table'
        self.goal_structure = {}
        for goal_fact in task.goals:
            parts = get_parts(goal_fact)
            if parts[0] == 'on':
                # (on block below)
                block, below = parts[1], parts[2]
                self.goal_structure[block] = below
            elif parts[0] == 'on-table':
                # (on-table block)
                block = parts[1]
                self.goal_structure[block] = 'table'
            # Ignore (clear) and (arm-empty) goals for goal_structure as they
            # are handled separately or implicitly by position/obstruction.

    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        state = node.state

        # 2. Analyze the current state
        current_structure = {} # block -> block_below_in_state or 'table' or 'holding'
        arm_holding = None
        current_on_facts = set() # Store (on Y B) facts in current state

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'on':
                # (on block below)
                block, below = parts[1], parts[2]
                current_structure[block] = below
                current_on_facts.add(fact)
            elif parts[0] == 'on-table':
                # (on-table block)
                block = parts[1]
                current_structure[block] = 'table'
            elif parts[0] == 'holding':
                # (holding block)
                block = parts[1]
                current_structure[block] = 'holding'
                arm_holding = block # Assuming only one block can be held
            # We don't need to process (clear) or (arm-empty) facts here for the heuristic calculation

        # 1. Initialize cost
        cost = 0

        # 3. Count misplaced blocks relative to goal support
        # Iterate through blocks that are part of the goal structure
        for block, desired_below in self.goal_structure.items():
            current_below = current_structure.get(block)

            # If a block in the goal structure isn't found in the current state's
            # position facts ('on', 'on-table', 'holding'), it's definitely misplaced.
            # This shouldn't happen in valid states from init/actions, but handle defensively.
            if current_below is None or current_below != desired_below:
                cost += 1

        # 4. Count obstructing blocks
        # Iterate through (on Y B) facts currently true in the state
        for fact in current_on_facts:
             # fact is (on Y B)
             if fact not in self.goal_facts_set:
                 cost += 1

        # 5. Handle (arm-empty) goal
        # Check if (arm-empty) is a goal and the arm is not empty
        if '(arm-empty)' in self.goal_facts_set and arm_holding is not None:
             cost += 1

        # 6. Return cost
        return cost
