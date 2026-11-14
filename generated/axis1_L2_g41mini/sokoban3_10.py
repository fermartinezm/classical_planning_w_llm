from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(odqd6scj obj1)".
    - `args`: The expected pattern with wildcards `*` allowed.
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain.

    Summary
    -------
    This heuristic estimates the number of actions required to reach the goal state by focusing on
    the key predicate `(wuqjn5r5 ?ndr6y ?el0fk)` present in the goal and the transition of `(odqd6scj ?el0fk)`
    facts along a chain of objects connected by the static relation `(k21e3pdq ?el0fk ?ey5ma ?a4o94)`.
    The heuristic measures the distance along this chain for the goal object from the current object
    holding `(wuqjn5r5 ?ndr6y ?el0fk)`, taking into account that the operator `gww2bqr6` moves the 
    `odqd6scj` condition along the chain, and the operator `yaz3jy7e` can re-activate predicates and 
    move the `wuqjn5r5` relation accordingly.

    Assumptions
    -----------
    - The goal consists of `(wuqjn5r5 ?ndr6y ?el0fk)`.
    - The key transitions involve moving the `odqd6scj` predicate along z1mgqxkc objects via the chain defined by 
      `(k21e3pdq ...)`.
    - `gww2bqr6` moves the `odqd6scj` predicate from one z1mgqxkc object to its linked successor, deactivating the old and activating the new.
    - `yaz3jy7e` can move the `wuqjn5r5` relation from one z1mgqxkc object to another along with re-activation of `ajcpzwcf` predicates.
    - The heuristic does not need to be admissible, so it approximates the number of actions by counting 
      minimal required transitions to propagate the necessary predicates.

    Heuristic Initialization
    ------------------------
    - Extract the goal `(wuqjn5r5 ?ndr6y ?el0fk)` facts: identify `ndr6y_goal` (of type kukxix7x) and the `el0fk_goal` (of type z1mgqxkc).
    - From static facts `(k21e3pdq ?el0fk ?ey5ma ?a4o94)`, build a directed graph (mapping each ?el0fk to a list of successors ?ey5ma).
    - Extract all `ajcpzwcf` objects initially from the state for future reasoning.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Check if the current state is a goal state: return 0 if so.
    2. Identify the current `(wuqjn5r5 ndr6y ?el0fk)` fact to find where `ndr6y` is currently linked.
       If not present, heuristic is at least 1 (need to establish it).
    3. Find the current object `el0fk_start` linked to `ndr6y` (`wuqjn5r5 ndr6y el0fk_start`).
    4. Compute the shortest path length (in terms of `k21e3pdq` edges) from `el0fk_start` to `el0fk_goal`.
       This length approximates the minimal number of `gww2bqr6` actions to move `odqd6scj` along the chain,
       plus possibly some `yaz3jy7e` actions to move or reactivate `wuqjn5r5` accordingly.
    5. Count how many `ajcpzwcf` predicates are currently true, how many are false but should be true at the goal related to the chain (approximate).
       Since `yaz3jy7e` requires `ajcpzwcf` to move `wuqjn5r5`, each missing `ajcpzwcf` may require an extra action.
    6. Add a small fixed cost if the `odqd6scj` predicate is not at the goal object currently, since it needs to be moved along the chain.
    7. Sum these components as the heuristic estimate.

    The heuristic hence balances:
    - Distance along the static chain (via `k21e3pdq`) between where `wuqjn5r5` currently points to and the goal.
    - The number of needed actions to move or re-activate predicates (`odqd6scj` and `ajcpzwcf`).
    """

    def __init__(self, task):
        # Store goals (expected to contain a single wuqjn5r5 predicate).
        self.goals = task.goals
        self.static = task.static

        # Extract goal for wuqjn5r5: (wuqjn5r5 ?ndr6y_goal ?el0fk_goal)
        wuq_goal = None
        for goal in self.goals:
            if match(goal, "wuqjn5r5", "*", "*"):
                parts = get_parts(goal)
                self.ndr6y_goal = parts[1]  # kukxix7x object in goal
                self.el0fk_goal = parts[2]  # z1mgqxkc object in goal
                wuq_goal = goal
                break
        else:
            # If no wuqjn5r5 goal found, fallback heuristic
            self.ndr6y_goal = None
            self.el0fk_goal = None

        # Build adjacency list from static facts k21e3pdq ?el0fk ?ey5ma ?a4o94
        # Directed edges from ?el0fk -> ?ey5ma
        self.graph = {}
        for fact in self.static:
            if match(fact, "k21e3pdq", "*", "*", "*"):
                parts = get_parts(fact)
                src, dst = parts[1], parts[2]
                self.graph.setdefault(src, []).append(dst)

        # Precompute the set of all ajcpzwcf objects appearing in static, to estimate their needed counts
        self.all_ajcpzwcf_objects = set()
        for fact in task.facts:
            # task.facts include all possible facts in domain, filter those with ajcpzwcf
            if match(fact, "ajcpzwcf", "*"):
                parts = get_parts(fact)
                self.all_ajcpzwcf_objects.add(parts[1])

    def __call__(self, node):
        state = node.state

        # If goal is already reached, cost is 0.
        if self.goals <= state:
            return 0

        # Find which el0fk the wuqjn5r5 relation currently points to for self.ndr6y_goal.
        wuqjn5r5_facts = [fact for fact in state if match(fact, "wuqjn5r5", self.ndr6y_goal, "*")]
        if not wuqjn5r5_facts:
            # If wuqjn5r5 relation not present, estimate cost as distance unknown but at least 1.
            # We assume need to establish this relation eventually.
            return 5  # Arbitrary small heuristic, since likely re-activation is needed.

        # Get the current el0fk where wuqjn5r5(ndr6y_goal, el0fk_current)
        el0fk_start = get_parts(wuqjn5r5_facts[0])[2]

        # Breadth-first search to find shortest path length from el0fk_start to el0fk_goal in self.graph
        from collections import deque

        if el0fk_start == self.el0fk_goal:
            dist = 0
        else:
            dist = None
            queue = deque([(el0fk_start, 0)])
            visited = {el0fk_start}
            while queue:
                current, steps = queue.popleft()
                for neighbor in self.graph.get(current, []):
                    if neighbor == self.el0fk_goal:
                        dist = steps + 1
                        queue.clear()
                        break
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, steps + 1))
            if dist is None:
                # No path found, return a large heuristic
                dist = 50

        # Count ajcpzwcf predicates true in the state (these are needed for the yaz3jy7e action)
        ajcpzwcf_true = sum(1 for fact in state if match(fact, "ajcpzwcf", "*"))

        # Count ajcpzwcf predicates false but appearing in all_ajcpzwcf_objects
        ajcpzwcf_all_count = len(self.all_ajcpzwcf_objects)
        ajcpzwcf_false = ajcpzwcf_all_count - ajcpzwcf_true

        # Check presence of odqd6scj at goal object
        odqd6scj_goal_present = any(match(fact, "odqd6scj", self.el0fk_goal) for fact in state)

        # If odqd6scj not at goal, need to move it along chain: add cost proportional to dist
        odqd6scj_move_cost = dist if not odqd6scj_goal_present else 0

        # Moving wuqjn5r5 along chain via yaz3jy7e likely requires ajcpzwcf predicates to be true, so each missing ajcpzwcf adds cost
        ajcpzwcf_reactivation_cost = ajcpzwcf_false

        # Total heuristic estimate balances:
        #  - distance to move odqd6scj along the chain
        #  - number of ajcpzwcf predicates to (re)activate
        #  - minimal base cost of 1 for any remaining needed action
        heuristic_value = odqd6scj_move_cost + ajcpzwcf_reactivation_cost + 1

        # Ensure heuristic is finite and non-negative
        return max(heuristic_value, 0)
