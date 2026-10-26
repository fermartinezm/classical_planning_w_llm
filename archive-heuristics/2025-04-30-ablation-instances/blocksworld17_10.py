from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(on b8 b9)" -> ["on", "b8", "b9"]
    # Example: "(on-table b9)" -> ["on-table", "b9"]
    # Example: "(arm-empty)" -> ["arm-empty"]
    if not fact or fact[0] != '(' or fact[-1] != ')':
         # Handle potential malformed facts, though unlikely with standard parsers
         return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed by counting how many
    blocks are not currently resting on the object (another block or the table)
    that is directly below them in the goal configuration. Each block not in
    its correct immediate goal position contributes 1 to the heuristic value.

    # Assumptions
    - The goal is defined by a set of (on ?x ?y) and (on-table ?x) facts
      that form one or more stacks.
    - Blocks not mentioned as being 'on' something or 'on-table' in the goal
      are not considered for the heuristic calculation.
    - The heuristic does not explicitly account for the cost of clearing blocks
      that are obstructing a desired placement or pickup. It focuses solely
      on the misplaced blocks themselves.

    # Heuristic Initialization
    - Parses the goal facts to build a mapping (`goal_below`) indicating,
      for each block that is the upper object in an (on ?x ?y) goal fact
      or the object in an (on-table ?x) goal fact, which object should be
      directly below it in the goal state.
    - Identifies the set of blocks (`goal_blocks`) that are the upper object
      in these goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize a counter `misplaced_count` to 0.
    2. Iterate through each block `block` that is the upper object in a goal
       (on ?x ?y) or (on-table ?x) fact (i.e., each block in `self.goal_blocks`).
    3. Determine the required object `target_below` that should be directly
       below `block` in the goal state, using the precomputed `self.goal_below` map.
    4. Construct the string representation of the desired goal fact:
       - If `target_below` is 'table', the desired fact is `(on-table block)`.
       - If `target_below` is a block name, the desired fact is `(on block target_below)`.
    5. Check if this `desired_fact_str` is present in the current state.
    6. If the `desired_fact_str` is NOT present in the current state, increment
       `misplaced_count`.
    7. After checking all relevant blocks in `self.goal_blocks`, the final value
       of `misplaced_count` is the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the goal structure mapping.
        """
        self.goals = task.goals  # Goal conditions.
        # Static facts are not needed for this heuristic in Blocksworld.
        # static_facts = task.static # Blocksworld has no static facts

        # Map a block to the object directly below it in the goal state.
        # 'table' is represented by the string 'table'.
        self.goal_below = {}
        # Set of blocks that are the *upper* object in an (on ?x ?y) goal fact
        # or the object in an (on-table ?x) goal fact. These are the blocks
        # whose position relative to the object below them is explicitly defined
        # in the goal.
        self.goal_blocks = set()

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if not parts: # Skip malformed facts if any
                continue
            predicate = parts[0]

            if predicate == 'on':
                if len(parts) == 3:
                    block, below = parts[1], parts[2]
                    self.goal_below[block] = below
                    self.goal_blocks.add(block)
                # else: # Malformed (on) fact, ignore
            elif predicate == 'on-table':
                 if len(parts) == 2:
                    block = parts[1]
                    self.goal_below[block] = 'table'
                    self.goal_blocks.add(block)
                 # else: # Malformed (on-table) fact, ignore
            # Ignore other goal predicates like 'clear' or 'arm-empty'
            # as they are typically consequences of the stack structure.


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state  # Current world state (frozenset of fact strings).

        misplaced_count = 0

        # Iterate through blocks whose target position relative to the object
        # below them is defined in the goal.
        for block in self.goal_blocks:
            # Find what should be directly below this block in the goal.
            # We only added blocks to goal_blocks if they are keys in goal_below.
            # So, block must be in self.goal_below.
            target_below = self.goal_below[block]

            # Construct the string representation of the desired goal fact.
            if target_below == 'table':
                desired_fact_str = f'(on-table {block})'
            else: # target_below is another block
                desired_fact_str = f'(on {block} {target_below})'

            # Check if the desired fact is present in the current state.
            if desired_fact_str not in state:
                misplaced_count += 1

        return misplaced_count
