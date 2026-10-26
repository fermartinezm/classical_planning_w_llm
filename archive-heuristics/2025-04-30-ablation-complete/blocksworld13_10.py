from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by counting several types of discrepancies between the current state and the goal state:
    missing goal relationships (on, on-table), existing incorrect relationships,
    whether the arm is holding a block, and whether blocks that should be clear
    in the goal are currently obstructed.

    # Assumptions
    - The goal state consists of specific 'on' and 'on-table' relationships,
      and potentially 'clear' predicates for the top blocks of goal stacks.
    - The 'arm-empty' goal is implicitly handled by penalizing the 'holding' state.
    - The heuristic is non-admissible and designed for greedy best-first search.

    # Heuristic Initialization
    - The heuristic extracts and stores the required 'on', 'on-table', and 'clear'
      predicates from the task's goal conditions. Static facts are not relevant
      for this domain.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as the sum of the following penalties:

    1.  **Missing Goal 'on' Facts:** Add 1 for each '(on ?x ?y)' predicate that is
        required in the goal but is not true in the current state.
    2.  **Missing Goal 'on-table' Facts:** Add 1 for each '(on-table ?x)' predicate
        that is required in the goal but is not true in the current state.
    3.  **Wrong Current 'on' Facts:** Add 1 for each '(on ?x ?y)' predicate that is
        true in the current state but is *not* required in the goal state. These
        represent blocks that are stacked incorrectly and need to be unstacked.
    4.  **Wrong Current 'on-table' Facts:** Add 1 for each '(on-table ?x)' predicate
        that is true in the current state but is *not* required in the goal state.
        These represent blocks on the table that should be stacked elsewhere.
    5.  **Arm Holding:** Add 1 if the robot arm is currently holding a block. The
        arm must be empty or holding the correct block to perform most useful actions.
    6.  **Obstructed Goal 'clear' Facts:** Add 1 for each '(clear ?x)' predicate
        that is required in the goal state but is *not* true in the current state.
        This penalizes blocks that are supposed to be accessible (at the top of
        their goal stack) but are currently obstructed by other blocks.

    The total heuristic value is the sum of these penalties. A value of 0 is
    achieved only when all goal conditions are met and no incorrect conditions
    (wrong stacks, wrong blocks on table, arm holding) exist, which corresponds
    to the goal state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        Blocksworld has no static facts relevant to this heuristic.
        """
        self.goals = task.goals

        self.goal_on = set()
        self.goal_on_table = set()
        self.goal_clear = set()

        # Parse goal facts
        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                # Store as tuple (block_on_top, block_below)
                self.goal_on.add((parts[1], parts[2]))
            elif predicate == "on-table":
                # Store the block name
                self.goal_on_table.add(parts[1])
            elif predicate == "clear":
                 # Store the block name
                self.goal_clear.add(parts[1])
            # 'arm-empty' goal is handled by penalizing 'holding' state

        # Static facts are not used in this heuristic for Blocksworld
        # static_facts = task.static

    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        state = node.state

        current_on = set()
        current_on_table = set()
        current_clear = set()
        current_holding = None # Blocksworld arm can hold at most one block

        # Parse current state facts
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                current_on.add((parts[1], parts[2]))
            elif predicate == "on-table":
                current_on_table.add(parts[1])
            elif predicate == "clear":
                current_clear.add(parts[1])
            elif predicate == "holding":
                current_holding = parts[1] # Store the block being held
            # 'arm-empty' is not stored, its absence implies holding or vice versa

        h = 0

        # 1. Penalty for missing goal 'on' facts
        for goal_b, goal_y in self.goal_on:
            if (goal_b, goal_y) not in current_on:
                h += 1

        # 2. Penalty for missing goal 'on-table' facts
        for goal_b in self.goal_on_table:
            if goal_b not in current_on_table:
                h += 1

        # 3. Penalty for wrong current 'on' facts
        for current_b, current_y in current_on:
            if (current_b, current_y) not in self.goal_on:
                h += 1

        # 4. Penalty for wrong current 'on-table' facts
        for current_b in current_on_table:
            if current_b not in self.goal_on_table:
                h += 1

        # 5. Penalty for arm holding a block
        if current_holding is not None:
            h += 1

        # 6. Penalty for obstructed goal 'clear' facts
        for goal_b in self.goal_clear:
            if goal_b not in current_clear:
                h += 1

        return h
