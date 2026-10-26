# Import necessary base class
from heuristics.heuristic_base import Heuristic
# Import Task and Operator classes (although not directly used in heuristic calculation,
# they are part of the environment and Task is used in __init__)
from task import Operator, Task

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by combining
    several factors:
    1. The number of desired positional facts ((on ?x ?y) or (on-table ?x))
       that are not currently true.
    2. The number of current positional facts ((on ?x ?y) or (on-table ?x))
       that are not desired in the goal state (i.e., blocks in the wrong
       position that need to be moved).
    3. A penalty if the goal requires the arm to be empty but it is not.

    This heuristic is not admissible but aims to guide a greedy best-first
    search efficiently by prioritizing states where more goal positions are
    satisfied and fewer blocks are in incorrect, blocking positions.

    Assumptions:
    - The heuristic assumes a standard Blocksworld domain with predicates
      (on ?x ?y), (on-table ?x), (clear ?x), (holding ?x), (arm-empty).
    - It assumes the goal state is reachable and consistently defined
      (e.g., a block is not required to be both clear and have something on it).
    - Facts in the state and goal are represented as strings (e.g., '(on b1 b2)').

    Heuristic Initialization:
    The constructor preprocesses the goal facts from the task definition.
    It separates the goal facts into sets of (on ?x ?y) predicates,
    (on-table ?x) predicates, and checks if (arm-empty) is a goal.
    These precomputed sets allow for efficient lookup during heuristic evaluation.
    Static facts are not explicitly used by this heuristic as positional facts
    like (on) and (on-table) are dynamic.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Identify all (on ?x ?y) and (on-table ?x) predicates that are true in the current state.
    2. Calculate the number of goal (on ?x ?y) or (on-table ?x) predicates that are
       *not* present in the current state. Add this count to the heuristic value.
       This represents goal positions that still need to be achieved.
    3. Calculate the number of current (on ?x ?y) or (on-table ?x) predicates that are
       *not* present in the goal state. Add this count to the heuristic value.
       This represents blocks that are currently in positions they shouldn't be
       in the goal, and likely need to be moved out of the way.
    4. Check if (arm-empty) is a goal predicate. If it is, and (arm-empty) is
       *not* true in the current state (meaning the arm is holding a block),
       add 1 to the heuristic value. This accounts for the action needed to free the arm.
    5. The total sum is the heuristic value for the state.
    The heuristic value is 0 if and only if all goal (on) and (on-table) predicates
    are met, no blocks are in incorrect (on) or (on-table) positions, and the
    arm-empty condition matches the goal. Given consistent goals, this implies
    the state is the goal state. (Clear) goals are implicitly handled by the
    penalty for blocks sitting on top of positions that should be clear or
    have a different block on them according to the goal.
    """

    def __init__(self, task):
        """
        Initializes the Blocksworld heuristic.

        Args:
            task: An instance of the Task class containing problem definition.
        """
        super().__init__()
        self.goals = task.goals

        # Preprocess goal facts for efficient lookup
        self.goal_on_set = {f for f in self.goals if f.startswith('(on ')}
        self.goal_on_table_set = {f for f in self.goals if f.startswith('(on-table ')}
        self.goal_arm_empty = '(arm-empty)' in self.goals

        # Note: We ignore (clear ?x) goals in the direct calculation,
        # as the penalty for blocks sitting on top of them (if that
        # position is not a goal) implicitly covers the cost to clear them.
        # Static facts are not relevant for this dynamic heuristic.

    def __call__(self, node):
        """
        Computes the heuristic value for a given state.

        Args:
            node: The search node containing the state.

        Returns:
            The estimated cost (heuristic value) to reach the goal.
        """
        state = node.state

        # Identify relevant facts in the current state
        state_on_set = {f for f in state if f.startswith('(on ')}
        state_on_table_set = {f for f in state if f.startswith('(on-table ')}
        state_arm_empty = '(arm-empty)' in state

        # Calculate heuristic components
        h = 0

        # Penalty for unsatisfied goal positions ((on) or (on-table))
        # These are goal facts that are not in the current state
        h += len(self.goal_on_set - state_on_set)
        h += len(self.goal_on_table_set - state_on_table_set)

        # Penalty for wrongly placed blocks ((on) or (on-table))
        # These are state facts that are not in the goal
        h += len(state_on_set - self.goal_on_set)
        h += len(state_on_table_set - self.goal_on_table_set)

        # Penalty for arm not being empty if required by goal
        if self.goal_arm_empty and not state_arm_empty:
             h += 1

        return h
