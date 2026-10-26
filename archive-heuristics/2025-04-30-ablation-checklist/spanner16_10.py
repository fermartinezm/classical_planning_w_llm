from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Used for infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    goal nuts. It simplifies travel costs and focuses on the sequence of
    picking up a usable spanner and tightening a nut for each goal nut.

    # Assumptions:
    - Each usable spanner can tighten exactly one nut.
    - The man can only carry one spanner at a time.
    - Travel between any two relevant locations (man, spanner, nut) is approximated
      as a fixed cost of 1 action (a single 'walk' step). This is a relaxation
      and makes the heuristic non-admissible but fast.
    - The cost of pickup and tighten actions is 1 each.
    - The problem involves a single man.

    # Heuristic Initialization
    - Identify the set of nuts that are specified in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man and whether he is currently carrying a usable spanner.
    2. Count the number of loose nuts in the current state that are also goal nuts.
    3. Count the total number of usable spanners available in the current state
       (either carried by the man or on the ground).
    4. If the number of usable spanners is less than the number of loose goal nuts,
       the problem is unsolvable from this state; return infinity.
    5. If there are no loose goal nuts, the goal is reached; return 0.
    6. Estimate the cost for the *first* loose goal nut:
       - If the man is already carrying a usable spanner:
         Cost = 1 (walk to nut location) + 1 (tighten) = 2 actions.
       - If the man is *not* carrying a usable spanner:
         Cost = 1 (walk to a usable spanner) + 1 (pickup) + 1 (walk to nut location) + 1 (tighten) = 4 actions.
    7. Estimate the cost for each *subsequent* loose goal nut (after the first one is tightened):
       - After tightening a nut, the man is at that nut's location and is no longer
         carrying a *usable* spanner (as it was consumed).
       - To tighten the next nut, the man must:
         Walk to a *new* usable spanner (1) + Pickup the spanner (1) +
         Walk to the next nut's location (1) + Tighten the nut (1).
       - Cost per subsequent nut = 4 actions.
    8. The total heuristic is the sum of the cost for the first nut and the
       estimated cost for all remaining nuts.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the set of goal nuts from the task definition.
        """
        self.goal_nuts = set()
        # Parse the goal conditions to find all nuts that need to be tightened.
        for goal in task.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened":
                nut = args[0]
                self.goal_nuts.add(nut)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.
        """
        state = node.state  # The current world state as a frozenset of facts.

        # 1. Identify the man and if he's carrying a usable spanner.
        # We assume there is only one man. We try to find him by looking for
        # the object involved in a 'carrying' fact first, as only men can carry.
        # If no 'carrying' fact exists, we fall back to looking for an object
        # in an 'at' fact that doesn't appear to be a nut or spanner based on name.
        # This fallback is fragile without explicit type information in the state facts.
        man_name = None
        man_carrying_usable = False
        carried_spanner = None

        for fact in state:
             if match(fact, "carrying", "*", "*"):
                 man_name = get_parts(fact)[1]
                 carried_spanner = get_parts(fact)[2]
                 break # Found the man and what he's carrying

        if man_name is None:
             # Fallback: Try to find the man from an 'at' fact.
             # This is a heuristic guess based on common PDDL naming conventions.
             for fact in state:
                 if match(fact, "at", "*", "*"):
                     parts = get_parts(fact)
                     obj_name = parts[1]
                     # Simple check: if the name doesn't contain 'nut' or 'spanner', assume it's the man.
                     if 'nut' not in obj_name.lower() and 'spanner' not in obj_name.lower():
                         man_name = obj_name
                         break
             # If man_name is still None, we cannot identify the man, which is unexpected.
             # Return infinity as the state is likely problematic or unsolvable by this heuristic.
             if man_name is None:
                  return math.inf

        # Check if the spanner the man is carrying (if any) is usable.
        if carried_spanner and f"(usable {carried_spanner})" in state:
             man_carrying_usable = True

        # 2. Count loose goal nuts in the current state.
        # These are the nuts that are currently 'loose' AND are in the goal set.
        loose_goal_nuts_in_state = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "loose", "*") and get_parts(fact)[1] in self.goal_nuts
        }
        num_loose_goal_nuts = len(loose_goal_nuts_in_state)

        # If there are no loose goal nuts, the goal is satisfied for all nuts.
        if num_loose_goal_nuts == 0:
            return 0

        # 3. Count usable spanners available (carried or on ground).
        # These are spanners for which the '(usable ?s)' predicate is true.
        usable_spanners_in_state = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "usable", "*")
        }
        num_usable_spanners = len(usable_spanners_in_state)

        # 4. Check solvability based on usable spanners.
        # If we need to tighten N nuts, we need N usable spanners in total throughout the plan.
        if num_usable_spanners < num_loose_goal_nuts:
             # Not enough usable spanners exist in the state to tighten all required nuts.
             # This state is unsolvable. Return infinity.
             return math.inf

        # 5. Calculate heuristic based on sequential actions with simplified costs.
        # This models the process of tightening nuts one by one.

        # Cost for the first nut:
        # This depends on whether the man starts already carrying a usable spanner.
        if man_carrying_usable:
            # Man has usable spanner -> needs to go to a nut and tighten it.
            # Simplified cost: 1 (walk to nut) + 1 (tighten) = 2 actions.
            cost_first_nut = 2
        else:
            # Man does not have a usable spanner -> needs to get one, go to a nut, and tighten.
            # Simplified cost: 1 (walk to spanner) + 1 (pickup) + 1 (walk to nut) + 1 (tighten) = 4 actions.
            cost_first_nut = 4

        # Cost for subsequent nuts (if any):
        # After tightening the first nut, the man is at that nut's location and
        # the spanner used is no longer usable. For each remaining nut, the man
        # must repeat the process of getting a *new* usable spanner and going
        # to the next nut.
        # Simplified cost per subsequent nut: 1 (walk to spanner) + 1 (pickup) +
        # 1 (walk to nut) + 1 (tighten) = 4 actions.
        num_remaining_nuts = num_loose_goal_nuts - 1
        cost_remaining_nuts = num_remaining_nuts * 4

        # Total heuristic is the sum of the cost for the first nut and the cost for all subsequent nuts.
        total_heuristic = cost_first_nut + cost_remaining_nuts

        return total_heuristic

