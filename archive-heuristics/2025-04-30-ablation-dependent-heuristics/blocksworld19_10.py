# Import necessary modules if any (only set is needed, which is built-in)
# frozenset is used in state representation, no need to import

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a tuple.
    e.g., '(on b1 b2)' -> ('on', 'b1', 'b2')
          '(on-table b1)' -> ('on-table', 'b1')
          '(clear b1)' -> ('clear', 'b1')
          '(arm-empty)' -> ('arm-empty',)
    """
    # Remove leading/trailing parens and splits by space
    parts = fact_string[1:-1].split()
    return tuple(parts)

class blocksworldHeuristic:
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the number of actions required to reach the goal
        state by summing the number of unsatisfied goal facts related to block
        positions and adding penalties for blocks that are held or are stacked
        incorrectly. It is designed for greedy best-first search and is not
        guaranteed to be admissible. The heuristic value is 0 if and only if
        the state is a goal state.

    Assumptions:
        - The planning task is a standard STRIPS task in the Blocksworld domain.
        - Goal states are defined by a set of (on ?x ?y), (on-table ?x), and
          potentially (clear ?x) and (arm-empty) facts. The heuristic primarily
          focuses on (on ?x ?y) and (on-table ?x) goal facts.
        - The heuristic is used with a greedy search algorithm (like greedy
          best-first search) where admissibility is not required, but a good
          estimate of remaining steps is beneficial for minimizing expanded nodes.
        - The cost of each action is implicitly assumed to be uniform (e.g., 1).

    Heuristic Initialization:
        The heuristic's constructor (`__init__`) precomputes necessary information
        from the task's goal state. It identifies all blocks involved in the
        problem by examining the initial and goal states. It stores the set of
        goal facts that specify block positions (`(on ...)` and `(on-table ...)`).
        It also builds a mapping (`self.goal_below`) from each block to its
        required support in the goal state (i.e., which block it should be on,
        or if it should be on the table). Static facts are noted but not used
        as they are typically not relevant for this heuristic's calculation in
        Blocksworld.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state, the heuristic value is computed as follows:

        1.  Initialize the heuristic value `h` to 0.
        2.  **Unsatisfied Goal Positions:** Iterate through the precomputed set of
            goal facts specifying block positions (`(on ...)` and `(on-table ...)`).
            For each such goal fact, if it is *not* present in the current state,
            increment `h` by 1. This counts how many required block placements
            are missing.
        3.  **Held Blocks Penalty:** Iterate through all blocks identified during
            initialization. Check if a block `X` is currently held (i.e., the fact
            `'(holding X)'` is in the state). If `X` is held, and it has a defined
            goal position (meaning it should end up on another block or the table,
            as determined during initialization), increment `h` by 1. A block
            that is held but needs to be placed somewhere contributes to the
            remaining work.
        4.  **Wrongly Stacked Blocks Penalty:** Iterate through all facts currently
            true in the state. If a state fact is of the form `(on X Y)`, check
            if this exact fact `'(on X Y)'` is also a goal fact. If `(on X Y)` is
            true in the state but is *not* a goal fact, it means block X is
            incorrectly placed on block Y, obstructing the correct configuration.
            Increment `h` by 2. This penalty estimates the cost of moving the
            wrongly placed block X out of the way (e.g., unstacking it and putting
            it on the table).
        5.  The final value of `h` is the heuristic estimate for the given state.
            The heuristic is 0 if and only if all goal position facts are satisfied,
            no blocks that need placing are held, and no blocks are wrongly stacked.
            This corresponds precisely to the goal state for standard blocksworld
            problems.
    """
    def __init__(self, task):
        # Precompute goal info
        self.goal_below = {}
        self.all_blocks = set()
        self.goal_on_facts = set()
        self.goal_ontable_facts = set()

        # Extract all blocks from initial state and goal state
        # Iterate through all facts, extract arguments that are not 'table'
        for fact_string in task.initial_state | task.goals:
            parsed = parse_fact(fact_string)
            for part in parsed[1:]:
                if part != 'table': # 'table' is not a block object
                     self.all_blocks.add(part)

        # Process goal facts to build goal_below mapping and sets of goal facts
        for fact_string in task.goals:
            parsed = parse_fact(fact_string)
            if parsed[0] == 'on':
                b1, b2 = parsed[1], parsed[2]
                self.goal_below[b1] = b2
                self.goal_on_facts.add(fact_string)
            elif parsed[0] == 'on-table':
                b = parsed[1]
                self.goal_below[b] = 'table'
                self.goal_ontable_facts.add(fact_string)
            # Ignore (clear) and (arm-empty) goal facts for these mappings

        # Static facts are available in task.static but not used in this heuristic
        # as they are typically not relevant for blocksworld state evaluation.
        # For completeness, one could process task.static here if needed.
        # Example: for fact_string in task.static: ...

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: A frozenset of strings representing the facts true in the state.
        @return: The estimated number of actions to reach the goal.
        """
        h = 0

        # Part 1: Unsatisfied goal positions
        # Check each goal (on) fact
        for goal_fact_string in self.goal_on_facts:
            if goal_fact_string not in state:
                h += 1

        # Check each goal (on-table) fact
        for goal_fact_string in self.goal_ontable_facts:
             if goal_fact_string not in state:
                 h += 1

        # Part 2 & 3: Penalties for held blocks and wrongly stacked blocks
        state_on_facts = set()
        state_held_blocks = set()

        # Pre-process state facts for easier lookup
        for fact_string in state:
             parsed = parse_fact(fact_string)
             if parsed[0] == 'on':
                 state_on_facts.add(fact_string)
             elif parsed[0] == 'holding':
                 state_held_blocks.add(parsed[1])

        # Penalty for held blocks
        for block in state_held_blocks:
             # If a block is held, and it has *any* goal position (on or on-table)
             if self.goal_below.get(block) is not None:
                  h += 1 # Cost to put it down or stack it

        # Penalty for wrongly stacked blocks
        for fact_string in state_on_facts:
             # Check if this (on X Y) fact is NOT a goal fact
             if fact_string not in self.goal_on_facts:
                 h += 2 # Cost to unstack X from Y and put X down

        return h
