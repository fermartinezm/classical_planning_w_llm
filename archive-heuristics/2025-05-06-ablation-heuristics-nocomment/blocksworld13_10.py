from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by spaces
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for Blocksworld.

    Summary:
        This heuristic estimates the cost to reach the goal by counting the number
        of "unhappy" blocks. A block is considered unhappy if it is not in its
        correct goal position relative to its base, or if it is in the correct
        position but has an incorrect block stacked on top of it, or if it is
        currently being held by the arm. The heuristic returns twice the number
        of unhappy blocks, providing a non-admissible estimate that aims to
        prioritize states where more blocks are in their desired configuration.

    Assumptions:
        - The goal state defines the desired positions for a subset of blocks
          using (on X Y) and (on-table Z) predicates. Blocks not mentioned
          in these goal predicates are assumed to not have a specific required
          base position, but they can still be obstructing.
        - The heuristic assumes a cost of roughly 2 actions (e.g., unstack/pickup
          followed by stack/putdown) are needed to correct the position of an
          unhappy block, potentially after clearing blocks on top.
        - The heuristic value is 0 if and only if the state is the goal state.

    Heuristic Initialization:
        The constructor pre-processes the goal state to build a mapping
        (self.goal_below) from each block to its required base (either another
        block or 'table') according to the goal facts (on X Y) and (on-table Z).
        It also stores the set of goal (on X Y) facts (self.goal_on_facts)
        to quickly check if a block Y should be on block X in the goal.
        It identifies all blocks present in the initial state and goal for
        iteration during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is the goal state. If yes, return 0.
        2. Parse the current state to determine the current base for each block.
           A block X's base is Y if (on X Y) is true, or 'table' if (on-table X)
           is true. Also, identify if a block is currently being held. Store
           all current (on X Y) facts.
        3. Initialize a counter for "unhappy" blocks to 0.
        4. Iterate through each block identified during initialization:
           a. Check if the arm is currently holding the block. If yes, the block
              is unhappy. Increment the counter and move to the next block.
           b. If the block is not held, determine its goal base from the
              pre-calculated self.goal_below map and its current base from the
              parsed state facts.
           c. Check if the block is unhappy based on its position and blocks on top:
              i. If the block has a required goal base (is in self.goal_below)
                 AND its current base is different from its goal base, the block
                 is unhappy. Increment the counter and move to the next block.
              ii. If the block is not held AND (it has the correct goal base OR
                  it has no required goal base), check if it is obstructed. A block
                  is obstructed if there is any block Y currently on top of it
                  ((on Y X) is true) AND that (on Y X) fact is NOT a goal fact.
                  If the block is obstructed, it is unhappy. Increment the counter.
        5. The heuristic value is 2 times the total count of unhappy blocks.
    """
    def __init__(self, task):
        self.goals = task.goals
        # static_facts = task.static # Blocksworld domain provided has no static facts

        self.goal_below = {}
        self.goal_on_facts = set()
        self.all_blocks = set()

        # Parse goal facts to build goal_below and goal_on_facts
        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                block, base = parts[1], parts[2]
                self.goal_below[block] = base
                self.goal_on_facts.add(goal)
                self.all_blocks.add(block)
                self.all_blocks.add(base)
            elif predicate == "on-table":
                block = parts[1]
                self.goal_below[block] = 'table'
                self.all_blocks.add(block)
            # (clear X) and (arm-empty) goal facts don't define block positions for goal_below

        # Add blocks from initial state.
        # This ensures we consider all blocks present in the problem instance,
        # even if they are not explicitly mentioned in the goal configuration facts.
        for fact in task.initial_state:
             parts = get_parts(fact)
             # Add all arguments of relevant predicates as blocks
             if parts[0] in ["on", "on-table", "holding", "clear"]:
                 for obj in parts[1:]:
                     self.all_blocks.add(obj)


    def __call__(self, node):
        state = node.state

        # Check if goal is reached (heuristic is 0)
        if self.goals <= state:
            return 0

        current_below = {}
        current_on_facts = set()
        holding_block = None

        # Parse current state facts
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                block, base = parts[1], parts[2]
                current_below[block] = base
                current_on_facts.add(fact)
            elif predicate == "on-table":
                block = parts[1]
                current_below[block] = 'table'
            elif predicate == "holding":
                holding_block = parts[1]
            # Ignore (clear X) and (arm-empty) for current_below/on_facts

        unhappy_count = 0

        # Iterate through all known blocks
        for block in self.all_blocks:
            # Condition 1: Block is being held
            if holding_block == block:
                unhappy_count += 1
                continue # Move to the next block

            goal_base = self.goal_below.get(block)
            current_base = current_below.get(block) # Can be None if not on/on-table

            # Condition 2: Block is not held, and it has a goal base, but current base is wrong
            if goal_base is not None and current_base != goal_base:
                unhappy_count += 1
                continue # Move to the next block

            # Condition 3: Block is not held, and either has correct goal base or no goal base, but is obstructed
            # Check if any block Y is currently on 'block'
            is_obstructed = False
            for fact in current_on_facts:
                parts = get_parts(fact)
                if parts[0] == "on" and parts[2] == block: # Y is on block
                    # Check if this (on Y block) fact is NOT a goal fact
                    if fact not in self.goal_on_facts:
                        is_obstructed = True
                        break # Found an obstructor

            if is_obstructed:
                unhappy_count += 1
                # continue # Move to the next block (implicitly done by loop)


        # The heuristic is 2 * the number of unhappy blocks.
        return 2 * unhappy_count
