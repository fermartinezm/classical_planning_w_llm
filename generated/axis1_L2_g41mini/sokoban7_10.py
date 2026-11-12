from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract components from a PDDL fact string by stripping parentheses and splitting on spaces.
    Example: "(k21e3pdq beg400qv dj19p99h oxpphcgv)" -> ["k21e3pdq", "beg400qv", "dj19p99h", "oxpphcgv"]
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check whether a fact matches a pattern, supporting wildcards.

    :param fact: The fact string (e.g., "(k21e3pdq a b c)").
    :param args: Pattern components, e.g. ("k21e3pdq", "*", "b", "*").
    :return: True if the fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(p, a) for p, a in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the 'secret_domain' domain.

    Summary
    -------
    This heuristic estimates the number of actions required to reach a goal state by 
    focusing on the main goal predicate (wuqjn5r5) and the related predicates that indicate 
    intermediate states (odqd6scj, ajcpzwcf, wuqjn5r5) as well as transitions allowed by the 
    two actions defined in the domain. The core idea is to estimate how far the state is from 
    achieving the wuqjn5r5 facts of the goal by propagating and counting intermediate steps 
    needed to transform odqd6scj and ajcpzwcf facts.

    Assumptions
    -----------
    - The goal consists only of wuqjn5r5 facts, each relating a kukxix7x object and a z1mgqxkc object.
    - Achieving a goal wuqjn5r5 fact requires bringing the z1mgqxkc object into the ajcpzwcf predicate state, and transferring odqd6scj as described by the operators.
    - The static facts (k21e3pdq) define valid transitions between z1mgqxkc objects under yrvdqfln labels.
    - The two operators gww2bqr6 and yaz3jy7 change predicates following a linked-chain logic through odqd6scj and ajcpzwcf facts with related z1mgqxkc objects.
    - The heuristic is not required to be admissible but should guide the search efficiently.

    Heuristic Initialization
    ------------------------
    - Extract the wuqjn5r5 facts from the goal as pairs of (kukxix7x_object, z1mgqxkc_object) to satisfy.
    - Extract from static facts the relation k21e3pdq(z1mgqxkc_from, z1mgqxkc_to, yrvdqfln_label), used as directed edges in a transition graph.
    - Represent k21e3pdq facts as adjacency dictionaries keyed by (current_z1mgqxkc, yrvdqfln) with reachable z1mgqxkc nodes.
    - Keep track of all ajcpzwcf and odqd6scj predicates present in the current state.

    Step-By-Step Thinking for Computing Heuristic
    ----------------------------------------------
    1. Check if the current state satisfies all goal wuqjn5r5 facts; if yes, heuristic is 0.
    2. For each goal wuqjn5r5(k, z), check whether it already holds.
       If not, the heuristic assesses how many steps are required to "move" the odqd6scj predicate from its current position to the goal position z, and set ajcpzwcf predicates accordingly to enable wuqjn5r5 to be true.
    3. The heuristic constructs a simplified distance estimate on the graph formed by k21e3pdq edges:
       - Use backward breadth-first search (BFS) or shortest path from each goal z to any odqd6scj currently true, connected by k21e3pdq, counting the minimum number of operator steps required.
       - Each gww2bqr6 action moves odqd6scj from one z1mgqxkc node to another.
       - Each yaz3jy7 action swaps ajcpzwcf predicates and changes wuqjn5r5 and odqd6scj in a more complex way, so taking into account ajcpzwcf facts reduces under- or over-estimation.
    4. Combine those distances to estimate the minimal number of actions to achieve all goal wuqjn5r5.
    5. Since not all exact cost computation is trivial, the heuristic uses the minimal path lengths (in terms of involved k21e3pdq edges) plus penalties for missing ajcpzwcf predicates related to the path, to form an efficient and effective heuristic estimate.
    6. Return the sum of these distances over all goal wuqjn5r5 facts as the heuristic.

    This heuristic is optimistic enough to guide search efficiently while reflecting the structure of the problem induced by the two domain operators.
    """

    def __init__(self, task):
        self.goals = task.goals

        # Parse goal wuqjn5r5 facts into tuples (kukxix7x_obj, z1mgqxkc_obj)
        self.goal_wuqjn5r5 = []
        for g in self.goals:
            parts = get_parts(g)
            if parts[0] == "wuqjn5r5":
                self.goal_wuqjn5r5.append((parts[1], parts[2]))

        # Build a graph from static k21e3pdq facts: edges from z1mgqxkc -> z1mgqxkc with label yrvdqfln
        # We'll represent as: adjacency[(src_z1mgqxkc, yrvdqfln_label)] = set of dst_z1mgqxkc
        self.adjacency = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "k21e3pdq":
                src, dst, label = parts[1], parts[2], parts[3]
                key = (src, label)
                if key not in self.adjacency:
                    self.adjacency[key] = set()
                self.adjacency[key].add(dst)

    def __call__(self, node):
        state = node.state

        # Quick goal check: heuristic is zero if all goal facts hold
        if self.goals <= state:
            return 0

        # Extract current predicates of interest from the state
        # odqd6scj: set of z1mgqxkc objects
        odqd6scj_current = {
            get_parts(fact)[1] for fact in state if fact.startswith("(odqd6scj ")
        }

        # ajcpzwcf: set of z1mgqxkc objects
        ajcpzwcf_current = {
            get_parts(fact)[1] for fact in state if fact.startswith("(ajcpzwcf ")
        }

        # wuqjn5r5: dict mapping kukxix7x_obj -> set of z1mgqxkc objs where predicate true
        wuqjn5r5_map = {}
        for fact in state:
            if fact.startswith("(wuqjn5r5 "):
                parts = get_parts(fact)
                k_obj, z_obj = parts[1], parts[2]
                if k_obj not in wuqjn5r5_map:
                    wuqjn5r5_map[k_obj] = set()
                wuqjn5r5_map[k_obj].add(z_obj)

        # Our goal: For each (k_obj, z_obj) in goal_wuqjn5r5,
        # check if waqjn5r5(k_obj, z_obj) holds; if yes cost=0 for it
        # else estimate cost to make it true:

        # Plan:
        # Action gww2bqr6 moves odqd6scj from some z1mgqxkc node to another:
        # Preconditions: odqd6scj ?q3jjz and k21e3pdq(?q3jjz ?xo2y4 ?clx7v) and ajcpzwcf ?xo2y4
        # Effect: odqd6scj no longer at ?q3jjz, but at ?xo2y4
        #
        # Action yaz3jy7 moves ajcpzwcf and wuqjn5r5 predicates in a way that rotates the related ajcpzwcf, wuqjn5r5, odqd6scj sets.
        #
        # So roughly:
        # - To get wuqjn5r5(k_obj, z_obj), ajcpzwcf must hold for z_obj.
        # - To get ajcpzwcf on z_obj, may have to flip ajcpzwcf using yaz3jy7 actions.
        # - To have odqd6scj on some z1mgqxkc, may have to move it via gww2bqr6 actions along edges k21e3pdq and ajcpzwcf states.
        #
        # Let's use a simplified approximation:
        # - From odqd6scj_current nodes, find minimum number of gww2bqr6 moves to z_obj in goal.
        # - If ajcpzwcf(z_obj) currently false, add 1 step at least (to flip ajcpzwcf at that node).
        # - Also if wuqjn5r5(k_obj,z_obj) currently false, minimum is 1 yaz3jy7 action (which produces wuqjn5r5 and modifies ajcpzwcf).
        #
        # We approximate the cost for each goal wuqjn5r5 fact by:
        # steps_to_move_odqd6scj + ajcpzwcf_fix_cost + 1 (for yaz3jy7)
        #
        # If no path between the odqd6scj current nodes and the goal z_obj, give a high but finite cost.

        max_cost = 1000  # Arbitrary large finite cost for unreachable goals

        # Precompute reverse adjacency for backward BFS from goal nodes: from dst to src with label matches.
        # We'll ignore label for heuristic simplicity, just consider edges as from src -> dst.
        reverse_adj = {}
        for (src, label), dsts in self.adjacency.items():
            for dst in dsts:
                if dst not in reverse_adj:
                    reverse_adj[dst] = set()
                reverse_adj[dst].add(src)

        def min_steps_to_odqd6scj(goal_z):
            """
            Compute minimal number of gww2bqr6 moves needed to get odqd6scj from one of the current
            odqd6scj nodes to goal_z by backward search on transition graph.

            Returns an integer >= 0 or None if unreachable.
            """
            from collections import deque

            if goal_z in odqd6scj_current:
                return 0

            visited = set()
            queue = deque()
            queue.append((goal_z, 0))
            visited.add(goal_z)

            while queue:
                current, dist = queue.popleft()
                # If this node currently has odqd6scj, return distance (number of steps)
                if current in odqd6scj_current:
                    return dist
                # Reverse neighbors
                for pred in reverse_adj.get(current, []):
                    if pred not in visited:
                        visited.add(pred)
                        queue.append((pred, dist + 1))
            return None

        heuristic_sum = 0

        for (k_obj, goal_z) in self.goal_wuqjn5r5:
            # If wuqjn5r5(k_obj, goal_z) true -> cost 0 for this goal
            if k_obj in wuqjn5r5_map and goal_z in wuqjn5r5_map[k_obj]:
                continue

            # Compute min steps to move odqd6scj to goal_z by repeated gww2bqr6-like moves
            steps_to_odqd6scj = min_steps_to_odqd6scj(goal_z)
            if steps_to_odqd6scj is None:
                # no way to move odqd6scj to goal_z, give large estimate but finite
                steps_to_odqd6scj = max_cost // 3

            # ajcpzwcf must hold for goal_z to apply gww2bqr6 and produce wuqjn5r5 on goal_z after yaz3jy7
            ajcpzwcf_missing_penalty = 0 if goal_z in ajcpzwcf_current else 1

            # Must do at least one yaz3jy7 action to transform predicates and generate wuqjn5r5 fact.
            yaz3jy7_cost = 1

            # Sum estimated cost for this goal
            cost_for_goal = steps_to_odqd6scj + ajcpzwcf_missing_penalty + yaz3jy7_cost

            heuristic_sum += cost_for_goal

        # Heuristic value must be positive if not goal state
        return max(heuristic_sum, 1)
