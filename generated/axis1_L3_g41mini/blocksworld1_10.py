from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The full fact string, e.g., "(road locatable3)".
    - `args`: The pattern parts to match, with wildcard '*' supported.
    - Returns True if fact matches the pattern; else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain (transport) domain.

    Summary
    -------
    This heuristic estimates the number of actions required to reach the goal state,
    based on the current configuration of the "road", "capacity-predecessor", "capacity",
    "in", and "at" predicates.

    The transport domain models movement of "road" objects through locations indicated by "at"
    predicates, with capacity constraints represented through "capacity" and
    "capacity-predecessor" predicates, and temporary holding indicated by "in".

    The goal involves certain facts about roads present, specific "at" relations, and
    "capacity-predecessor" facts holding for some objects.

    Assumptions
    -----------
    - The "drive" action requires a road with capacity and a capacity-predecessor.
    - "pick-up" and "drop" actions transfer capacity state and road predicates between objects.
    - The "drive-2" action moves road objects across locations under capacity constraints.
    - The state of capacity/predecessor can only be changed through pick-up, drop, or drive actions.
    - The heuristic does not have to be admissible; it aims for effective guidance and efficiency.
    - No static facts are present in current tasks (static set is empty).
    - The domain objects represent locations or mobile entities ("road", "at", "capacity-predecessor" refer to these).

    Heuristic Initialization
    ------------------------
    - Extract and store the set of goal facts for fast membership checks.
    - No static facts extraction is needed as domain static is empty.
    - No specialized preprocessing since the domain uses only a fixed small set of predicate types.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For the current state:

    1. If the current state satisfies all goal facts, return 0.

    2. Count the number of discrepancies for each predicate type relevant to the goal:
       - Number of goal `(road ?v)` facts missing in the state.
       - Number of goal `(capacity-predecessor ?v)` facts missing.
       - Number of goal `(capacity)` fact missing (0 or 1 fact).
       - Number of goal `(at ?v1 ?v2)` facts missing.

    3. Count the number of unwanted facts present in the state that contradict the goal:
       - `(road ?v)` facts present but not in goal.
       - `(capacity-predecessor ?v)` facts present but not in goal.
       - `(capacity)` fact present but not in goal.
       - `(at ?v1 ?v2)` facts present but not in goal.

    4. Analyze "in" facts:
       - If any objects are currently "in" (held), assume it will cost at least one action per "in" object to drop or move them appropriately.

    5. Aggregate these counts and multiply by small weights to reflect the typical action costs:
       - Each missing or extra predicate roughly corresponds to at least one action to fix.
       - "in" objects likely need at least one pick/drop action each.

    6. Return the weighted sum as a heuristic estimate of the number of required actions.

    This heuristic is efficient to compute (counts of facts) and captures mismatches between current and goal states,
    thus guiding the greedy best-first search more effectively than trivial heuristics.

    It is not guaranteed admissible but aims for realistic estimates to minimize search expansions.
    """

    def __init__(self, task):
        self.goals = task.goals
        # As static is empty, no static info is processed

        # Precompute sets of goal facts by predicate for fast membership
        self.goal_road = set(fact for fact in self.goals if fact.startswith("(road "))
        self.goal_capacity_predecessor = set(
            fact for fact in self.goals if fact.startswith("(capacity-predecessor ")
        )
        self.goal_capacity = set(fact for fact in self.goals if fact == "(capacity)")
        self.goal_at = set(fact for fact in self.goals if fact.startswith("(at "))

    def __call__(self, node):
        state = node.state

        # If goal satisfied, heuristic is 0
        if self.goals <= state:
            return 0

        # Helper lambdas to separate facts by predicate types in current state:
        road_facts = set(f for f in state if f.startswith("(road "))
        cap_pred_facts = set(f for f in state if f.startswith("(capacity-predecessor "))
        capacity_fact = set(f for f in state if f == "(capacity)")
        at_facts = set(f for f in state if f.startswith("(at "))
        in_facts = set(f for f in state if f.startswith("(in "))

        # Count missing goals:
        missing_roads = len(self.goal_road - road_facts)
        missing_cap_preds = len(self.goal_capacity_predecessor - cap_pred_facts)
        missing_capacity = 0 if self.goal_capacity <= capacity_fact else 1 if len(self.goal_capacity) > 0 else 0
        missing_ats = len(self.goal_at - at_facts)

        # Count extra facts not in goal that may need removal:
        extra_roads = len(road_facts - self.goal_road)
        extra_cap_preds = len(cap_pred_facts - self.goal_capacity_predecessor)
        extra_capacity = 0 if len(self.goal_capacity) > 0 else 1 if len(capacity_fact) > 0 else 0
        extra_ats = len(at_facts - self.goal_at)

        # Count number of "in" facts (held items), each likely requiring at least 1 pick/drop action
        in_count = len(in_facts)

        # Heuristic weights (tuned to typical action costs):
        ROAD_WEIGHT = 2
        CAP_PREDECESSOR_WEIGHT = 2
        CAPACITY_WEIGHT = 1
        AT_WEIGHT = 3
        IN_WEIGHT = 2
        EXTRA_WEIGHT = 1  # Penalty for extra unwanted facts

        heuristic_value = 0

        # Add costs for missing goal facts (must add or fix)
        heuristic_value += missing_roads * ROAD_WEIGHT
        heuristic_value += missing_cap_preds * CAP_PREDECESSOR_WEIGHT
        heuristic_value += missing_capacity * CAPACITY_WEIGHT
        heuristic_value += missing_ats * AT_WEIGHT

        # Add costs for extra facts that must be removed or changed
        heuristic_value += (extra_roads + extra_cap_preds) * EXTRA_WEIGHT
        heuristic_value += extra_capacity * EXTRA_WEIGHT
        heuristic_value += extra_ats * EXTRA_WEIGHT

        # Add cost to handle "in" facts (items currently held that need resolving)
        heuristic_value += in_count * IN_WEIGHT

        # Ensure heuristic is at least 1 if not goal to avoid zero for non-goal states
        if heuristic_value == 0:
            heuristic_value = 1

        return heuristic_value
