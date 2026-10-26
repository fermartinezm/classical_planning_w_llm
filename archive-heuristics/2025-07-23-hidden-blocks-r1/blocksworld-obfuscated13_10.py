from heuristics.heuristic_base import Heuristic

class secret_domainHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain.

    Summary:
    This heuristic estimates the number of actions required to achieve the goal state by:
      1. Counting missing 'on' relations, base, and top predicates in the current state.
      2. Adding costs for converting objects to required states.
      3. Adjusting for the global flag state when conversions are needed.

    Assumptions:
      - The goal conditions are independent of the global flag state.
      - Objects in the middle state (q71n8mtz) can be converted to base/top with one action.
      - Objects in invalid states (e.g., top-only when base is required) incur a high penalty.
      - The global flag must be set for oecve6pr actions; each such action unsets the flag.

    Heuristic Initialization:
      - Extracts goal conditions for base, top, and 'on' relations during initialization.

    Step-By-Step Thinking for Computing Heuristic:
      1. Identify missing base, top, and 'on' relations by comparing current state with goal.
      2. For missing base:
            - If the object is in the middle state, add 1 (dgdzvgpm or m82i1lr3 cost).
            - If the object is top-only, add a high penalty (invalid state).
      3. For missing top:
            - If the object is in the middle state, add 1 (dgdzvgpm cost).
            - If the object is base-only, add a high penalty (invalid state).
      4. For each missing 'on' relation:
            - Add 1 (m82i1lr3 cost).
            - If the object to place has both base and top, add 1 (oecve6pr cost to convert to middle).
      5. Calculate flag cost:
            - Each oecve6pr action requires the flag. If the flag is currently set, subtract one from the total oecve6pr count (since the first action doesn't need an extra set).
            - The flag cost equals the adjusted oecve6pr count.
      6. Sum costs: missing_on + cost_base + cost_top + cost_oecve6pr + cost_flag.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        self.goal_base = set()
        self.goal_top = set()
        self.goal_on = set()

        for fact in self.goals:
            parts = fact[1:-1].split()
            if not parts:
                continue
            predicate = parts[0]
            if predicate == "dhjk1eyy":
                if len(parts) >= 2:
                    obj = parts[1]
                    self.goal_base.add(obj)
            elif predicate == "xk87au5b":
                if len(parts) >= 2:
                    obj = parts[1]
                    self.goal_top.add(obj)
            elif predicate == "x272hpoe":
                if len(parts) >= 3:
                    obj1 = parts[1]
                    obj2 = parts[2]
                    self.goal_on.add((obj1, obj2))

    def __call__(self, node):
        state = node.state

        missing_base = set()
        for obj in self.goal_base:
            if f'(dhjk1eyy {obj})' not in state:
                missing_base.add(obj)

        missing_top = set()
        for obj in self.goal_top:
            if f'(xk87au5b {obj})' not in state:
                missing_top.add(obj)

        missing_on = set()
        for (obj1, obj2) in self.goal_on:
            if f'(x272hpoe {obj1} {obj2})' not in state:
                missing_on.add((obj1, obj2))

        cost_on = len(missing_on)

        cost_base = 0
        for obj in missing_base:
            if f'(q71n8mtz {obj})' in state:
                cost_base += 1
            elif f'(xk87au5b {obj})' in state and f'(dhjk1eyy {obj})' not in state:
                cost_base += 10000

        cost_top = 0
        for obj in missing_top:
            if f'(q71n8mtz {obj})' in state:
                cost_top += 1
            elif f'(dhjk1eyy {obj})' in state and f'(xk87au5b {obj})' not in state:
                cost_top += 10000

        cost_oecve6pr = 0
        for (x, y) in missing_on:
            if f'(dhjk1eyy {x})' in state and f'(xk87au5b {x})' in state:
                cost_oecve6pr += 1

        current_flag = '(hxtpdpff)' in state
        cost_flag = cost_oecve6pr
        if current_flag and cost_oecve6pr > 0:
            cost_flag = max(0, cost_oecve6pr - 1)

        total_cost = cost_on + cost_base + cost_top + cost_oecve6pr + cost_flag
        return total_cost
