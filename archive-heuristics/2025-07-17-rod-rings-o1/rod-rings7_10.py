from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Helper function to strip parentheses and split
    a fact string into its components.

    Example:
      fact = "(on r1 r2)" -> returns ["on", "r1", "r2"]
    """
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern (wildcards allowed).

    fact: A string representing the fact, e.g., "(on r1 r2)"
    args: A tuple of pattern components, e.g., ("on", "*", "*")

    Returns True if each component of 'fact' matches the corresponding
    pattern in 'args', False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class rod_ringsHeuristic(Heuristic):
    """
    Summary
    -------
    This domain-dependent heuristic for the rod-rings domain estimates the number
    of actions (put_top or put_bottom) needed to achieve the goal arrangement of rings
    on rods. It does this by counting how many goal facts are currently unsatisfied,
    and it applies simple domain logic to group "on", "top", and "bot" goal relations.

    Assumptions
    -----------
    - There is only one ring that can be held at a time.
    - The domain uses two actions (put_top and put_bottom) to rearrange rings.
    - Each ring's final arrangement can be described by up to three predicates:
        (on ringA ringB), (top ring sX), (bot ring sY).
      The goal state is satisfied if all required predicates for each ring hold.

    Heuristic Initialization
    ------------------------
    - Extract all goal facts from the task (those in task.goals).
    - Partition them by predicate type into three dictionaries:
      goal_on, goal_top, goal_bot.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state already satisfies all goal facts (i.e., task.goals is a
       subset of the current state's facts), return 0.
    2. Otherwise, parse the current state's facts into dictionaries current_on,
       current_top, current_bot for quick look-up of each ring's "on", "top",
       and "bot" relationships.
    3. For each ring that must be "on" another ring in the goal, check whether
       the current state matches this relation. If not, increment the heuristic cost.
    4. Do the same for each ring that must be "top" on a particular stick, and each
       ring that must be "bot" on a particular stick.
    5. Sum these increments to yield the final heuristic value. Since each discrepancy
       in arrangement may indicate at least one required rod-rings action, the heuristic
       grows with the number of mismatched facts.

    This approach, while relatively simple, can guide a greedy best-first search
    effectively by recognizing how many pieces of the final arrangement are missing.
    """

    def __init__(self, task):
        """
        Constructor for the rod-ringsHeuristic. Stores the goal facts
        and organizes them by their call to "on", "top", or "bot".
        """
        super().__init__(task)
        self.goals = task.goals

        # Dictionaries to record goal relationships for quick comparison.
        self.goal_on = {}
        self.goal_top = {}
        self.goal_bot = {}

        # Partition the goal facts into separate dicts for 'on', 'top', 'bot'.
        for fact in self.goals:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                _, ring_top, ring_bottom = parts
                self.goal_on[ring_top] = ring_bottom
            elif predicate == "top" and len(parts) == 3:
                _, ring, stick = parts
                self.goal_top[ring] = stick
            elif predicate == "bot" and len(parts) == 3:
                _, ring, stick = parts
                self.goal_bot[ring] = stick

    def __call__(self, node):
        """
        Compute the heuristic estimate for the current node's state.

        Returns an integer cost: the higher the cost, the more
        unsatisfied goal facts remain.
        """
        state = node.state

        # If the current state already satisfies all goals, return 0.
        if self.goals <= state:
            return 0

        # Parse current state facts into dictionaries for easy look-up.
        current_on = {}
        current_top = {}
        current_bot = {}

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "on" and len(parts) == 3:
                _, ring_top, ring_bottom = parts
                current_on[ring_top] = ring_bottom
            elif predicate == "top" and len(parts) == 3:
                _, ring, stick = parts
                current_top[ring] = stick
            elif predicate == "bot" and len(parts) == 3:
                _, ring, stick = parts
                current_bot[ring] = stick

        # Count mismatches for each relevant ring arrangement.
        cost = 0

        # Check "on" relations from the goal.
        for ring_top, ring_bottom in self.goal_on.items():
            if current_on.get(ring_top) != ring_bottom:
                cost += 1

        # Check "top" relations from the goal.
        for ring, stick in self.goal_top.items():
            if current_top.get(ring) != stick:
                cost += 1

        # Check "bot" relations from the goal.
        for ring, stick in self.goal_bot.items():
            if current_bot.get(ring) != stick:
                cost += 1

        return cost