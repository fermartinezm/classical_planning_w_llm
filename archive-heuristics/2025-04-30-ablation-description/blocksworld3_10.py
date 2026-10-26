from fnmatch import fnmatch

# Assuming a base class Heuristic is provided in the environment,
# similar to the example:
# from heuristics.heuristic_base import Heuristic

# If Heuristic base class is not provided, you might need a mock definition like:
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the cost to reach the goal by counting the number
    of goal predicates of the form (on ?x ?y) or (on-table ?x) that are not
    satisfied in the current state. It ignores (clear ?x) and (arm-empty)
    goals for the heuristic calculation.

    # Assumptions
    - The goal state is defined by a set of (on ?x ?y) and (on-table ?x)
      predicates, potentially along with (clear ?x) and (arm-empty).
    - The heuristic focuses primarily on the positional goals ((on ...) and
      (on-table ...)), as achieving these is the core task and often
      implicitly satisfies or simplifies achieving clear and arm-empty goals.
    - Each action that changes the position of a block (pickup, putdown, stack, unstack)
      can satisfy at most one positional goal predicate ((on ...) or (on-table ...)).
      This makes the heuristic admissible (a lower bound on the number of
      positioning actions).

    # Heuristic Initialization
    - The heuristic extracts all (on ?x ?y) and (on-table ?x) predicates
      from the task's goal state. These are the specific positional
      relationships that must be achieved. These predicates are stored
      in `self.positional_goals`.
    - Blocksworld domain typically has no static facts relevant to this
      heuristic, so `task.static` is not used.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value `h` to 0.
    2. Retrieve the set of positional goal predicates (`self.positional_goals`)
       extracted during the initialization phase.
    3. Get the current state from the input `node`. The state is a frozenset
       of PDDL fact strings.
    4. Iterate through each goal predicate `goal_fact` in the set of
       `self.positional_goals`.
    5. For each `goal_fact`:
       - Check if `goal_fact` is present in the current `state`.
       - If `goal_fact` is *not* present in the current `state`, it means
         this specific positional requirement of the goal is not yet met.
         Increment the heuristic counter `h` by 1.
    6. The final value of `h` represents the total count of unsatisfied
       positional goal predicates, which serves as the heuristic estimate
       of the remaining effort.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the relevant goal predicates.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        # Call the base class constructor if inheriting from Heuristic
        # super().__init__(task)

        # Extract only the (on ?x ?y) and (on-table ?x) goal predicates
        # These are the positional goals that this heuristic counts.
        self.positional_goals = frozenset({
            goal for goal in task.goals
            if match(goal, "on", "*", "*") or match(goal, "on-table", "*")
        })

        # Blocksworld has no static facts relevant to this heuristic,
        # so task.static is not used here.

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        state = node.state  # The current state is a frozenset of facts.

        # The heuristic is the number of positional goal predicates
        # that are not true in the current state.
        h = 0
        for goal_fact in self.positional_goals:
            if goal_fact not in state:
                h += 1

        return h
