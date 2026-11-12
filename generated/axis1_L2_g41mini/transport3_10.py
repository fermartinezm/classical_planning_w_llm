from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.

    Example:
        '(yarbdfgc obj1 obj2)' -> ['yarbdfgc', 'obj1', 'obj2']
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate obj1 obj2)".
    - `args`: The expected pattern to match; wildcards '*' allowed.
    Returns True if matches, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (mtfkspuf) domain.

    Summary
    -------
    This heuristic estimates the minimal number of actions required to reach the goal state(s) from the current state.
    It focuses primarily on achieving the goal facts which are all predicates of the form (jft9ynda ?gpvob ?pk9o1), i.e.,
    association of a val3ypio object with a hmhptaxe object. The heuristic exploits the structure of the domain's
    operators that allow changing these associations through transfer actions, and it uses the static facts 
    (z6o4oitz) defining possible transitions among hmhptaxe objects.

    Assumptions
    -----------
    - The goal is formed only by conjunctions of (jft9ynda ?gpvob ?pk9o1).
    - Static facts (z6o4oitz) define a symmetric graph allowing transitions between hmhptaxe objects.
    - Operators sw2fg9ql and vyn92k55 manipulate the (jft9ynda), (yarbdfgc) and (uypgj7u3) predicates, enabling key "moves."
    - Action k12p0czo changes the relationship (jft9ynda ?pk9o1 ?xbsk8) to (jft9ynda ?pk9o1 ?r3kn5) following the static graph.
    - We ignore the r86varlt objects when estimating cost as they are intermediate in preconditions but do not appear
      explicitly in goal or affect movable predicates directly.
    - The heuristic is not guaranteed to be admissible but aims for good guidance in greedy best-first search by 
      approximating required "moves".

    Heuristic Initialization
    ------------------------
    - Extract the set of goal (jft9ynda) pairs (g83opnlj, hmhptaxe) the heuristic must realize.
    - Construct a graph from static facts (z6o4oitz) representing transitions between hmhptaxe objects.
    - Precompute shortest distances between all relevant hmhptaxe nodes using BFS.
    - Identify all objects of types g83opnlj and hmhptaxe that appear in goals and static facts.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. For each g83opnlj object in the goal (the first argument of jft9ynda), identify the current hmhptaxe object it is associated with in the state:
       - This association is given by predicates (jft9ynda ?gpvob ?pk9o1) with ?gpvob of type val3ypio (including g83opnlj subtype) and ?pk9o1 of type hmhptaxe.
       - If the g83opnlj target is not currently associated with any hmhptaxe, assume it must be established via some action costing at least one move.

    2. Compute the shortest distance in terms of static graph edges (z6o4oitz relations) from the current associated hmhptaxe to the goal hmhptaxe for that g83opnlj.
       - If they already match, distance is zero.
       - Otherwise, the heuristic counts the number of k12p0czo actions (or similar) needed to move from current to target.

    3. For each g83opnlj, add a cost of 1 if the jft9ynda predicate in the goal differs from the current (because actions sw2fg9ql or vyn92k55 are needed to update related (yarbdfgc) or (uypgj7u3) predicates, representing "switching" or "handover" steps).

    4. Sum all these estimated costs over all g83opnlj objects appearing in the goal.

    5. Additionally, account heuristically for any (yarbdfgc) mismatches that are prerequisites for goal transformations, by counting the number of goal-relevant (yarbdfgc) facts not yet true.

    6. The heuristic is zero if and only if the current state satisfies the goal.

    7. The heuristic remains finite due to BFS distances over static graph and bounded differences between current and goal assignments.

    This heuristic exploits the problem structure:
    - Aligning g83opnlj objects to the correct hmhptaxe targets.
    - Using static transitions that reflect possible moves of those assignments.
    - Counting necessary update actions to true goal facts.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract goal facts for fast membership
        self.goal_jft9ynda = set(
            fact for fact in self.goals if fact.startswith("(jft9ynda ")
        )
        self.goal_yarbdfgc = set(
            fact for fact in self.goals if fact.startswith("(yarbdfgc ")
        )

        # Build adjacency graph from static z6o4oitz facts (hmhptaxe <-> hmhptaxe edges)
        self.graph = {}
        for fact in self.static:
            if fact.startswith("(z6o4oitz "):
                parts = get_parts(fact)
                # parts = [predicate, node1, node2]
                _, n1, n2 = parts
                self.graph.setdefault(n1, set()).add(n2)
                self.graph.setdefault(n2, set()).add(n1)

        # Cache for shortest paths between hmhptaxe nodes, to avoid recomputation
        self._dist_cache = {}

        # Extract the set of g83opnlj objects referenced in goals (first argument of jft9ynda)
        self.goal_gpvobs = set()
        for fact in self.goal_jft9ynda:
            _, gpvob, hmhptaxe_obj = get_parts(fact)
            self.goal_gpvobs.add(gpvob)

    def shortest_distance(self, start, goal):
        """
        Compute shortest distance between two hmhptaxe nodes in static graph using BFS.
        Caches results for efficiency.

        Returns an integer >= 0 or None if no path exists.
        """
        if start == goal:
            return 0
        if (start, goal) in self._dist_cache:
            return self._dist_cache[(start, goal)]
        if (goal, start) in self._dist_cache:
            return self._dist_cache[(goal, start)]

        from collections import deque

        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            current, dist = queue.popleft()
            for neighbor in self.graph.get(current, []):
                if neighbor == goal:
                    self._dist_cache[(start, goal)] = dist + 1
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        # no path found
        self._dist_cache[(start, goal)] = None
        return None

    def __call__(self, node):
        """
        Estimate the minimal number of required actions to reach a goal state from node.state.
        """

        state = node.state

        # Quick check if goal achieved
        if self.goals <= state:
            return 0

        # Helper to get current jft9ynda assignments for each g83opnlj object in goal
        current_jft9ynda_map = {}  # g83opnlj -> hmhptaxe
        for fact in state:
            if fact.startswith("(jft9ynda "):
                parts = get_parts(fact)
                if len(parts) == 3:
                    predicate, gpvob, hmhptaxe_obj = parts
                    if gpvob in self.goal_gpvobs:
                        current_jft9ynda_map[gpvob] = hmhptaxe_obj

        # Heuristic estimate accumulator
        total_cost = 0

        # For each g83opnlj in goal, estimate cost to move association to goal hmhptaxe
        for goal_fact in self.goal_jft9ynda:
            _, gpvob, goal_hmhptaxe = get_parts(goal_fact)
            current_hmhptaxe = current_jft9ynda_map.get(gpvob, None)

            # If not currently assigned, assign cost of at least 1 to establish (e.g., via sw2fg9ql or vyn92k55)
            if current_hmhptaxe is None:
                total_cost += 2  # Assuming minimal cost 2: one to create association and one to confirm
                continue

            # If current matches goal, no distance cost for movement on this gpvob
            if current_hmhptaxe == goal_hmhptaxe:
                dist = 0
            else:
                dist = self.shortest_distance(current_hmhptaxe, goal_hmhptaxe)
                if dist is None:
                    # No path in the static graph, consider a large cost
                    dist = 10

            # The dist corresponds roughly to k12p0czo moves needed to translate jft9ynda to goal hmhptaxe.

            # Additional cost of 1 for needed update switching (sw2fg9ql or vyn92k55), as they remove and add (yarbdfgc) and (uypgj7u3) facts
            update_cost = 1 if (current_hmhptaxe != goal_hmhptaxe or
                               f"(jft9ynda {gpvob} {goal_hmhptaxe})" not in state) else 0

            total_cost += dist + update_cost

        # Count how many goal (yarbdfgc) facts are missing, each likely needs an action in effect updating those predicates
        missing_yarbdfgc = [fact for fact in self.goal_yarbdfgc if fact not in state]
        total_cost += len(missing_yarbdfgc)

        # Return heuristic estimate. It is zero if and only if the goal is reached.
        return total_cost
