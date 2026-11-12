from heuristics.heuristic_base import Heuristic
from typing import List, Set


def get_parts(fact: str) -> List[str]:
    """
    Remove surrounding parentheses from a PDDL fact string and split into parts.
    Example: "(at obj loc)" -> ["at", "obj", "loc"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A lightweight, domain-dependent heuristic for the "content-portion" (secret_domain)
    PDDL domain. The heuristic estimates the number of additional actions required to
    satisfy all goal facts by (1) checking which goal facts are already true in the
    current state and (2) for each unsatisfied goal, estimating a small, finite number
    of actions required based on which operator(s) in the domain can produce that goal
    and how many of the corresponding precondition predicate-types are currently missing
    (in the union of the current state and static facts).

    Assumptions
    -----------
    - The heuristic does not attempt full grounding or exact variable binding reasoning.
      Instead it reasons on the level of predicate names: it checks whether required
      predicate types (e.g., "served", "at_kitchen_bread", "no_gluten_sandwich",
      "at_kitchen_bread-2") appear anywhere in the current state or in the static facts.
    - If an operator that produces a goal has all its predicate-type preconditions
      present somewhere in state+static, we assume the cost to produce the goal is
      roughly 1 (one operator application). If some of the predicate-type preconditions
      are missing we add a small penalty per missing predicate-type (each representing
      extra work to enable the operator).
    - The heuristic is intentionally permissive (non-admissible) and favors a small,
      informative estimate to guide greedy best-first search with low computation cost.

    Heuristic Initialization
    ------------------------
    - self.goals:
        The set of goal fact strings (task.goals).
    - self.static_facts:
        The set of static fact strings (task.static) that are true in every state.
    - self.static_predicates:
        A set of predicate names that appear in static facts for quick lookup.
    - self.producer_preconditions:
        A mapping from goal-predicate-name (string) to a list of predicate-names that
        commonly appear as preconditions for operators that add facts with that goal
        predicate. These lists were derived from inspecting the domain operators:
          * "no_gluten_bread-2" is typically produced by move_tray-2 and needs
            predicates such as at_kitchen_bread, no_gluten_sandwich, at_kitchen_bread-2, served.
          * "no_gluten_sandwich-2" is typically produced by serve_sandwich-2 and needs
            at_kitchen_bread, no_gluten_sandwich, not_allergic_gluten, served.
        If a goal predicate is not in the mapping we fall back to a small default cost.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    When the heuristic is called on a node:
    1. If all goal facts are already present in the state's fact set, return 0.
    2. Build a quick index of predicate-names present in the union of the current state
       and static facts. This is simply a set of the first token of each fact string.
    3. For each goal fact that is not present in the current state:
       a. Extract the goal predicate name (first token).
       b. Look up a representative list of precondition predicate-names that can
          produce that goal (self.producer_preconditions). If none known, use a
          small default producer list (empty).
       c. Count how many of those representative predicate-names are missing from
          the state+static predicate set. Each missing predicate-type charges a small
          penalty (1 per missing type).
       d. The local estimated cost for producing that goal is estimated as:
            base_cost_for_operator_application (1) + missing_predicate_count
          (so if all precondition predicate-types are available somewhere, cost ≈ 1).
    4. Sum the per-goal estimates to produce the overall heuristic value and return it.
    5. The heuristic therefore yields 0 exactly on goal states, is finite for solvable
       states (we always produce a finite integer), and is designed to be cheap to compute.

    Notes
    -----
    - This heuristic is intentionally simple and fast. It is domain-dependent (it
      leverages knowledge of which operators produce which predicates) but does not
      attempt costly grounding or exact parameter unification checks. The objective
      is to provide an informative guidance signal for greedy best-first search
      while keeping computation overhead minimal.
    """

    def __init__(self, task):
        # Goals that must be achieved (strings like "(no_gluten_bread-2 content-portion2 tray2)")
        self.goals: Set[str] = set(task.goals)
        # Static facts that never change (set of fact strings)
        self.static_facts: Set[str] = set(task.static)

        # Build set of predicate names present in static facts for fast presence checks.
        self.static_predicates: Set[str] = set()
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts:
                self.static_predicates.add(parts[0])

        # Mapping: goal-predicate -> representative list of predicate-names
        # that typically appear in preconditions of operators that add that goal predicate.
        # These mappings were inspected from the domain description.
        self.producer_preconditions = {
            # "no_gluten_bread-2" is produced by move_tray-2 (and related sequences that
            # produce the at_kitchen_bread-2 facts). Representative preconditions:
            "no_gluten_bread-2": [
                "at_kitchen_bread",
                "no_gluten_sandwich",
                "at_kitchen_bread-2",
                "served",
            ],
            # "no_gluten_sandwich-2" is produced by serve_sandwich-2:
            "no_gluten_sandwich-2": [
                "at_kitchen_bread",
                "no_gluten_sandwich",
                "not_allergic_gluten",
                "served",
            ],
            # Other potential goal predicates could be handled similarly if needed.
        }

        # A small default base cost used if a goal predicate is not present in the map.
        self.default_base_cost = 2

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given search node.

        Parameters
        ----------
        - node: a planner node object with attribute `state` (a frozenset of PDDL fact strings).

        Returns
        -------
        - An integer heuristic estimate (0 for goal states).
        """
        state: Set[str] = set(node.state)

        # 1) If already a goal state, return 0.
        if self.goals <= state:
            return 0

        # 2) Build the set of predicate-names that appear in the union of state and static.
        # This gives a cheap indicator of which predicate-types are already available.
        present_predicates: Set[str] = set(self.static_predicates)  # start with static
        for fact in state:
            parts = get_parts(fact)
            if parts:
                present_predicates.add(parts[0])

        total_cost = 0

        # 3) For each goal fact not in the state, estimate cost.
        for goal_fact in self.goals:
            if goal_fact in state:
                continue  # already satisfied

            parts = get_parts(goal_fact)
            if not parts:
                # malformed fact; charge a conservative default
                total_cost += self.default_base_cost
                continue

            goal_pred = parts[0]

            # If we know a representative producer precondition set for this goal predicate,
            # use it to estimate how many predicate-types are missing.
            if goal_pred in self.producer_preconditions:
                required_pred_types = self.producer_preconditions[goal_pred]
                # Count how many of these representative predicate names are missing.
                missing = sum(1 for p in required_pred_types if p not in present_predicates)
                # Cost = 1 application of producer operator + penalty for each missing predicate-type.
                local_cost = 1 + missing
            else:
                # Unknown goal predicate: a conservative small estimate (non-zero).
                # We assume at least one operator application is required, possibly with
                # one supporting fact missing.
                local_cost = self.default_base_cost

            total_cost += local_cost

        # Return finite integer estimate.
        return int(total_cost)