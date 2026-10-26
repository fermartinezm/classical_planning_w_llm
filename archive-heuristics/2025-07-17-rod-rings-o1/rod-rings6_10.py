from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Remove the surrounding parentheses from a PDDL fact and split it into tokens.
    For example, '(on r2 r4)' -> ['on', 'r2', 'r4'].
    """
    return fact[1:-1].split()

class rod_ringsHeuristic(Heuristic):
    """
    Summary:
    This domain-dependent heuristic for the rod-rings domain estimates the cost
    (in terms of number of actions) required to achieve all goal conditions that
    reference rings. The estimate is computed by counting unsatisfied goal facts
    and returning a simple function of that count.

    Assumptions:
    1. Each ring can appear in one or more goal facts, such as (top ?r ?s),
       (bot ?r ?s), or (on ?r ?r2).  
    2. A single ring may appear in multiple subgoals (e.g., both (top r2 s1)
       and (on r2 r4)), and each subgoal contributes to the heuristic cost if it
       is not satisfied in the current state.
    3. There are no static facts that influence the heuristic in a sophisticated
       way, so we only focus on the goal conditions referencing each ring.
    4. The heuristic is not admissible; it purposefully overestimates the
       number of required actions by assigning a fixed penalty per unsatisfied
       subgoal in order to reduce the risk of underestimating.

    Heuristic Initialization:
    1. Extract all goal conditions from the planning task.
    2. For each goal condition that involves a ring, record that subgoal in a
       ring-specific collection. This allows us to track which subgoals each ring
       needs to fulfill.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state satisfies all goals. If yes, the heuristic is 0.
    2. Otherwise, for each ring:
       a) Retrieve the set of subgoals that mention this ring.
       b) Count how many of those subgoals are not yet satisfied in the current state.
       c) Multiply the number of unsatisfied subgoals by a small constant (2) to
          approximate the likely sequence of domain actions needed (because each
          change of position typically involves picking up or putting down a ring).
    3. Sum these costs across all rings to compute the final heuristic value.
    """

    def __init__(self, task):
        """
        Store the subgoals mentioning each ring in a dictionary, keyed by ring name.
        Also store all goal facts to quickly check if the goal is fully satisfied.
        """
        self.goals = task.goals
        self.static_facts = task.static  # Usually empty for rod-rings.

        # Map each ring to the set of goal facts in which it appears.
        self.ring_goals = {}

        for fact in self.goals:
            parts = get_parts(fact)
            predicate = parts[0]

            # Identify the ring involved in the subgoal.
            # (top rX sY) => ring is parts[1]
            # (bot rX sY) => ring is parts[1]
            # (holding rX) => ring is parts[1]
            # (on rX rY) => ring X is parts[1]
            if predicate in ("top", "bot", "holding"):
                ring = parts[1]
            elif predicate == "on":
                ring = parts[1]
            else:
                # Ignore any goal fact that doesn't reference a ring as its first argument
                continue

            if ring not in self.ring_goals:
                self.ring_goals[ring] = set()
            self.ring_goals[ring].add(fact)

    def __call__(self, node):
        """
        Calculate the heuristic value for the given state by counting the number
        of unsatisfied ring-related subgoals. A penalty of 2 is applied for each
        unsatisfied subgoal to reflect the multi-step nature of ring manipulation.
        """
        # If all goals are satisfied, return 0 immediately.
        if self.goals <= node.state:
            return 0

        cost = 0

        # For each ring, count how many of its goal facts are unsatisfied.
        for ring, subgoals in self.ring_goals.items():
            unsatisfied_count = sum(1 for g in subgoals if g not in node.state)
            # Assign a penalty of 2 actions per unsatisfied subgoal.
            cost += 2 * unsatisfied_count

        return cost