from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The full fact string e.g. "(predicate arg1 arg2)".
    - `args`: Expected predicate and arguments, supports wildcard '*' matching.
    - Returns True if matches, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (mtfkspuf).

    Summary
    -------
    The heuristic estimates the number of actions needed to achieve the goal
    facts, primarily focusing on the predicate `jft9ynda` being true for certain
    object pairs, since the domain's goal conditions involve positive `jft9ynda` facts.
    It uses information about static relations and current state bindings to estimate
    minimal steps by reasoning about required changes to predicates `jft9ynda` and `yarbdfgc`.

    Assumptions
    -----------
    - Goals are conjunctions of positive facts mostly involving `jft9ynda` predicate.
    - The operator `sw2fg9ql` switches `yarbdfgc` and `jft9ynda` relations between objects.
    - The operator `k12p0czo` changes the `jft9ynda` relation of a `hodp29q0` object to a related `hmhptaxe`.
    - The operator `vyn92k55` appears to revert effects of `sw2fg9ql` or create a link between `g83opnlj` and `hodp29q0`.
    - Static facts describe connectivity via `z6o4oitz` which relate `hmhptaxe` objects.
    - The heuristic seeks a light cost estimate of required moves of `jft9ynda` relations to match goals, considering minimal steps to align intermediate predicates (`yarbdfgc`) and links (`uypgj7u3`).

    Heuristic Initialization
    ------------------------
    - Extract goals (predicates and their arguments) and focus on goal facts with `jft9ynda`.
    - Parse static facts related to `z6o4oitz` to build a graph of relation connections among `hmhptaxe` objects.
    - No need to consider other predicates in static as none directly impact heuristic cost.
    - Store goal pairs for `jft9ynda`, to check which pairs still need to be established.
    - Cache static connectivity graph for quick shortest-path lookups between related objects.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify which goal `jft9ynda` facts are not currently satisfied in the given state.
    2. For each missing goal `(jft9ynda ?pk9o1 ?oqsrn)`, estimate cost to achieve this:
       a) Check current `jft9ynda` targets of `?pk9o1`. If different, a "replacement" action (`k12p0czo`) is needed.
       b) Determine minimal path distance between current and goal `hmhptaxe` bindings via the `z6o4oitz` graph (reflecting pointer updates).
    3. Similarly, check if `yarbdfgc` predicates need to be adjusted for correct `hodp29q0` to `r86varlt` bindings to enable eventual `jft9ynda` changes (connected to `sw2fg9ql` and `vyn92k55` operators).
    4. Count the number of required replacement actions and path "updates" (each costs at least one action).
    5. Include costs for establishing or removing `uypgj7u3` link predicates needed for some actions.
    6. The heuristic sums all minimal required changes for unsatisfied goals, including connectivity and intermediate predicate fixes.
    7. If current state already satisfies all goals, heuristic returns 0.
    8. The heuristic is not guaranteed admissible but balances efficiency and informativeness to reduce expanded nodes.

    This approach yields a finite and efficiently computable heuristic estimate by leveraging
    the domain structure and static connectivity to approximate minimum operator sequences.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Build graph of hmhptaxe connectivity from static (z6o4oitz ?x ?y)
        self.connectivity = {}
        for fact in self.static:
            if match(fact, "z6o4oitz", "*", "*"):
                parts = get_parts(fact)
                src, tgt = parts[1], parts[2]
                self.connectivity.setdefault(src, set()).add(tgt)

        # Precompute all pairs shortest paths among hmhptaxe using BFS to estimate minimal links traversed for changing jft9ynda relations
        # This will help to estimate minimal actions needed for k12p0czo operator sequences.
        self.shortest_paths = {}

        def bfs(start):
            visited = {start: 0}
            queue = [start]
            while queue:
                current = queue.pop(0)
                for neighbor in self.connectivity.get(current, ()):
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            return visited

        # Compute shortest paths from each node that appears in connectivity keys
        nodes = set(self.connectivity.keys())
        for neighbors in self.connectivity.values():
            nodes.update(neighbors)
        for node in nodes:
            self.shortest_paths[node] = bfs(node)

        # Extract all goal jft9ynda facts for quick reference
        self.goal_jft9ynda = {
            (parts[1], parts[2])
            for fact in self.goals
            if match(fact, "jft9ynda", "*", "*")
            for parts in [get_parts(fact)]
        }

        # Extract initial goal y arbdfgc facts for guidance (may assist but optional)
        self.goal_yarbdfgc = {
            (parts[1], parts[2])
            for fact in self.goals
            if match(fact, "yarbdfgc", "*", "*")
            for parts in [get_parts(fact)]
        }

    def __call__(self, node):
        state = node.state

        # Quick goal check: if all goals hold, heuristic = 0
        if self.goals <= state:
            return 0

        # Get current jft9ynda assignments indexed by first argument (hodp29q0)
        current_jft9ynda = {}
        for fact in state:
            if match(fact, "jft9ynda", "*", "*"):
                parts = get_parts(fact)
                current_jft9ynda[parts[1]] = parts[2]

        # Get current yarbdfgc assignments indexed by first argument (hodp29q0)
        current_yarbdfgc = {}
        for fact in state:
            if match(fact, "yarbdfgc", "*", "*"):
                parts = get_parts(fact)
                current_yarbdfgc[parts[1]] = parts[2]

        # Get uypgj7u3 links (g83opnlj <-> hodp29q0), sets of connected pairs
        uypgj7u3_links = set()
        for fact in state:
            if match(fact, "uypgj7u3", "*", "*"):
                parts = get_parts(fact)
                uypgj7u3_links.add((parts[1], parts[2]))

        total_cost = 0

        # Heuristic part 1:
        # Estimate cost for achieving goal jft9ynda relations.
        for (pk9o1_goal, hmhptaxe_goal) in self.goal_jft9ynda:
            hmhptaxe_current = current_jft9ynda.get(pk9o1_goal)
            if hmhptaxe_current != hmhptaxe_goal:
                # Replacement or adjustment needed via k12p0czo operator possibly multiple times along z6o4oitz path.

                # If no current assignment, assume 1 step to establish
                if hmhptaxe_current is None:
                    # Minimal cost = 1 for k12p0czo action to set to goal hmhptaxe
                    total_cost += 1
                else:
                    # Find shortest path length between current and goal hmhptaxe bindings
                    dist_map = self.shortest_paths.get(hmhptaxe_current, {})
                    dist = dist_map.get(hmhptaxe_goal, None)
                    if dist is None:
                        # No path found, pessimistically assign a large cost (but finite)
                        dist = 5  # arbitrary finite upper bound; can be tuned
                    # Each step likely corresponds to one action (k12p0czo)
                    total_cost += dist

        # Heuristic part 2:
        # Align yarbdfgc relations if needed for enabling operators that rely on their matching with jft9ynda.
        # For each goal yarbdfgc fact, if current state differs, add some cost estimate.

        for (pk9o1_goal, r86varlt_goal) in self.goal_yarbdfgc:
            r86varlt_current = current_yarbdfgc.get(pk9o1_goal)
            if r86varlt_current != r86varlt_goal:
                # Changing yarbdfgc probably requires sw2fg9ql or vyn92k55 actions, cost at least 1
                total_cost += 1

        # Heuristic part 3:
        # Consider missing uypgj7u3 links that might be required,
        # since these links are added by vyn92k55 and removed by sw2fg9ql and appear relevant for goal jft9ynda changes.

        # We can consider the number of required uypgj7u3 links to be added or removed by checking goal-related (g83opnlj, hodp29q0) pairs.
        # But since uypgj7u3 does not appear in goals explicitly, we instead reward states missing necessary links for jft9ynda updates.
        # Conservative approach: add cost proportional to number of jft9ynda facts that need link support.

        for (pk9o1_goal, hmhptaxe_goal) in self.goal_jft9ynda:
            # For each pk9o1_goal that is also hodp29q0,
            # find g83opnlj objects u with uypgj7u3(u, pk9o1_goal) in state.
            # If none exist, assume cost 1 to establish required link before sw2fg9ql or vyn92k55.
            found_link = any(
                link[1] == pk9o1_goal
                for link in uypgj7u3_links
            )
            if not found_link:
                total_cost += 1

        # Heuristic by design never returns 0 unless goals fully satisfied.
        # Return the computed cost estimate.
        return total_cost
