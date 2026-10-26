from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(on b1 b2)" -> ["on", "b1", "b2"]
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by counting the number of facts that are "wrong" in the current state
    compared to the goal state. Specifically, it counts:
    1. Goal facts that are not true in the current state (missing desired conditions).
    2. `(on X Y)` facts that are true in the current state but are not goal facts
       (blocks that are on top of other blocks incorrectly).
    3. If `(arm-empty)` is a goal fact, and the arm is not empty in the current state.

    This heuristic is non-admissible and designed to guide a greedy best-first search.

    # Assumptions
    - The domain is Blocksworld with standard predicates (`on`, `on-table`, `clear`, `holding`, `arm-empty`).
    - The goal is specified using `on`, `on-table`, `clear`, and potentially `arm-empty` facts.
    - Actions have unit cost.

    # Heuristic Initialization
    The heuristic is initialized by parsing the goal conditions from the task.
    It separates goal facts into categories: `on`, `on-table`, `clear`, and `arm-empty`.
    Static facts are not used by this heuristic.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:
    1. Initialize the heuristic cost to 0.
    2. Parse the current state to identify all true facts, categorized by predicate (`on`, `on-table`, `clear`, `holding`, `arm-empty`).
    3. Count the number of goal `(on X Y)` facts that are not present in the current state. Add this count to the total cost.
    4. Count the number of goal `(on-table X)` facts that are not present in the current state. Add this count to the total cost.
    5. Count the number of goal `(clear X)` facts that are not present in the current state. Add this count to the total cost.
    6. Count the number of `(on T B)` facts that are present in the current state but are *not* among the goal `(on X Y)` facts. These represent blocks `T` that are currently on top of blocks `B` in a configuration not desired by the goal, and thus `T` needs to be moved. Add this count to the total cost.
    7. If `(arm-empty)` is a goal fact and the arm is currently holding a block (i.e., `(arm-empty)` is not in the state), add 1 to the total cost.
    8. The total cost is the heuristic value for the state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.

        @param task: The planning task object containing goal conditions.
        """
        self.goals = task.goals

        # Store goal facts by predicate for quick lookup
        self.goal_on = set()
        self.goal_on_table = set()
        self.goal_clear = set()
        self.goal_arm_empty = False

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            predicate = parts[0]
            if predicate == "on":
                self.goal_on.add(goal_fact)
            elif predicate == "on-table":
                self.goal_on_table.add(goal_fact)
            elif predicate == "clear":
                self.goal_clear.add(goal_fact)
            elif predicate == "arm-empty":
                self.goal_arm_empty = True
            # Ignore other potential goal predicates if any

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        @param node: The search node containing the current state.
        @return: The estimated cost (heuristic value) to reach the goal.
        """
        state = node.state
        cost = 0

        # Parse current state facts
        state_on = set()
        state_on_table = set()
        state_clear = set()
        state_holding = set()
        state_arm_empty = False

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                state_on.add(fact)
            elif predicate == "on-table":
                state_on_table.add(fact)
            elif predicate == "clear":
                state_clear.add(fact)
            elif predicate == "holding":
                state_holding.add(fact)
            elif predicate == "arm-empty":
                state_arm_empty = True

        # 1. Count missing goal 'on' facts
        for goal_fact in self.goal_on:
            if goal_fact not in state_on:
                cost += 1

        # 2. Count missing goal 'on-table' facts
        for goal_fact in self.goal_on_table:
            if goal_fact not in state_on_table:
                cost += 1

        # 3. Count missing goal 'clear' facts
        for goal_fact in self.goal_clear:
            if goal_fact not in state_clear:
                cost += 1

        # 4. Count 'on' facts in state that are not goal 'on' facts
        # These represent blocks that are on top of others incorrectly
        for state_fact in state_on:
            if state_fact not in self.goal_on:
                cost += 1

        # 5. If arm-empty is a goal and the arm is not empty
        if self.goal_arm_empty and not state_arm_empty:
             cost += 1
             # Optionally, could add cost for the block being held if it's misplaced,
             # but the current counting of misplaced 'on' facts might cover this implicitly
             # or we keep it simple. Let's keep it simple for now.

        return cost

