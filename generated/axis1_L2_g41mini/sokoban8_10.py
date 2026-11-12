from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain PDDL domain.

    Summary
    -------
    This heuristic estimates the number of actions required to achieve goals
    involving the predicate (wuqjn5r5 ?k ?z). The goal fact requires that a
    specific kukxix7x object is related to a specific z1mgqxkc object with wuqjn5r5.

    The domain actions move and manipulate facts related to odqd6scj, ajcpzwcf,
    wuqjn5r5, and the relation k21e3pdq which links z1mgqxkc pairs with a yrvdqfln constant.

    The heuristic measures progress by how many goal wuqjn5r5 facts are already true,
    and how many "steps" it takes to propagate odqd6scj (which can be seen as some form of location/state marker)
    through the chain of k21e3pdq links, while also considering ajcpzwcf predicates that appear
    to gate the applicability of actions.

    Assumptions
    -----------
    - The only goal facts are of the form (wuqjn5r5 ?k ?z).
    - k21e3pdq describes some kind of adjacency or connectivity between z1mgqxkc objects,
      parameterized by a yrvdqfln label.
    - odqd6scj marks a particular z1mgqxkc as "active" or "current".
    - ajcpzwcf is a boolean marker on z1mgqxkc objects,
      which likely controls the applicability of domain actions.
    - wuqjn5r5 relates kukxix7x objects to z1mgqxkc objects;
      goal demands specific such pairs.
    - Only two actions exist, and their effects propagate odqd6scj and wuqjn5r5,
      while toggling ajcpzwcf on certain objects.
    - Each "hop" through k21e3pdq edges corresponds roughly to one or two actions.
    - The heuristic ignores the kukxix7x parameters except in wuqjn5r5,
      instead focusing on z1mgqxkc objects related to the goal.

    Heuristic Initialization
    ------------------------
    - Extract all k21e3pdq facts from static facts and organize them as a directed labeled graph:
      keys: z1mgqxkc object (source), values: list of (target, yrvdqfln) tuples.
    - Extract all goal facts of form (wuqjn5r5 kukxix7x_obj z1mgqxkc_obj).
      Focus the heuristic on the z1mgqxkc_obj that must be reached with wuqjn5r5.
    - Store the static facts for quick reference.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Check if the current state is a goal state: if all goal wuqjn5r5 facts hold, return 0.
    2. Extract which (wuqjn5r5 ?k ?z) facts currently hold; determine which goal targets are still missing.
    3. Find the current z1mgqxkc object that has odqd6scj predicate in the state.
       This represents the "current position" or "active node" for propagation.
    4. Use a BFS in the k21e3pdq graph to find shortest paths from the current odqd6scj location
       to each goal target z1mgqxkc that does not yet have the goal wuqjn5r5.
    5. Each edge traversal can be approximated to cost 1 or 2 actions, since gww2bqr6 moves odqd6scj along edges,
       and yaz3jy7e propagates wuqjn5r5 and ajcpzwcf with more complex conditions.
    6. Estimate the heuristic cost as:
       - For each goal target not yet reached:
         minimal distance in edges from current odqd6scj location to that node * 2
         (two actions roughly per edge to propagate odqd6scj and then do a wuqjn5r5 move)
       - If no path exists, assume a large penalty.
    7. Return the minimal sum of these costs as the heuristic value to guide the search.
    """

    def __init__(self, task):
        """Initialize heuristic data structures from static and goal facts."""
        self.goals = task.goals  # set of goal facts (strings)
        self.static = task.static

        # Build a graph from k21e3pdq facts:
        # map from z1mgqxkc source to list of (target, yrvdqfln) edges
        self.k21_graph = {}
        for fact in self.static:
            if match(fact, "k21e3pdq", "*", "*", "*"):
                parts = get_parts(fact)
                src, tgt, lbl = parts[1], parts[2], parts[3]
                if src not in self.k21_graph:
                    self.k21_graph[src] = []
                self.k21_graph[src].append((tgt, lbl))

        # Extract the set of goal target z1mgqxkc for wuqjn5r5 facts:
        # goal facts are (wuqjn5r5 kukxix7x_obj z1mgqxkc_obj)
        self.goal_targets = set()
        for fact in self.goals:
            if match(fact, "wuqjn5r5", "*", "*"):
                goal_parts = get_parts(fact)
                self.goal_targets.add(goal_parts[2])  # third part is target z1mgqxkc

    def __call__(self, node):
        """
        Compute an estimate of remaining actions to reach goal state.

        Returns 0 if goal reached.

        Otherwise, estimates minimal actions by BFS distances on k21e3pdq edges from current odqd6scj location
        to each goal target z1mgqxkc that still lacks wuqjn5r5 in state,
        multiplied by 2 to roughly count steps needed for action sequences.
        """
        state = node.state

        # Goal test: if all goal facts are true in state, heuristic = 0
        if self.goals <= state:
            return 0

        # Extract all currently true wuqjn5r5 facts in the state
        current_wuqjn5r5_targets = {
            get_parts(fact)[2]
            for fact in state
            if match(fact, "wuqjn5r5", "*", "*")
        }

        # Determine which goal targets are not yet reached by wuqjn5r5
        missing_targets = self.goal_targets - current_wuqjn5r5_targets
        if not missing_targets:
            return 0  # all goals met

        # Find the current location of odqd6scj (should be exactly one)
        odqd6scj_locs = [get_parts(fact)[1] for fact in state if match(fact, "odqd6scj", "*")]
        if not odqd6scj_locs:
            # If none found, large heuristic (state far from goal)
            return 1000
        current_loc = odqd6scj_locs[0]

        # Perform BFS from current_loc in k21e3pdq graph to find shortest distances to missing targets
        from collections import deque

        visited = set()
        queue = deque([(current_loc, 0)])
        distances = {}

        while queue and missing_targets:
            node_loc, dist = queue.popleft()
            if node_loc in visited:
                continue
            visited.add(node_loc)

            if node_loc in missing_targets:
                distances[node_loc] = dist
                missing_targets.remove(node_loc)
                if not missing_targets:
                    break

            for (neighbor, _) in self.k21_graph.get(node_loc, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))

        # For targets not found in graph BFS assign a large distance
        max_penalty = 1000
        heuristic = 0
        for target in self.goal_targets:
            if target in distances:
                # The factor 2 assumes 2 actions per graph edge: one to move odqd6scj, one to do wuqjn5r5 propagation
                heuristic += 2 * distances[target]
            else:
                # unreachable target in graph -> high penalty
                heuristic += max_penalty

        return heuristic
