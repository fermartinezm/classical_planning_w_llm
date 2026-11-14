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
    A domain-dependent heuristic for the secret_domain (u2gh8ts5) domain.

    Summary
    -------
    This heuristic estimates the number of actions required to reach a goal state from the current state.
    The goal in this domain is to have a certain (wuqjn5r5 ?ndr6y ?el0fk) fact true.
    The domain contains two actions that manipulate the unary predicate odqd6scj and ajcpzwcf on z1mgqxkc objects,
    as well as wuqjn5r5 relations involving kukxix7x and z1mgqxkc objects.
    The heuristic leverages:
      1. That the goal requires wuqjn5r5 for a fixed kukxix7x constant and some z1mgqxkc object,
      2. The crucial chain of switching odqd6scj predicates from one z1mgqxkc object to another using action gww2bqr6,
      3. The limited accessible transitions captured by static facts of k21e3pdq,
      4. Maintaining ajcpzwcf as a "flag" set on objects to control applicability of actions.

    Assumptions
    -----------
    - The goal is always a single wuqjn5r5 fact with a fixed kukxix7x object constant.
    - Only one odqd6scj is true at a time, representing a "current" location or mark.
    - The transitions between z1mgqxkc objects are governed via k21e3pdq static facts indexed by the yrvdqfln object,
      which functions like a "label" on edges.
    - The ajcpzwcf predicate marks which z1mgqxkc objects are "flagged" to allow transitions.
    - The heuristic does not need to be admissible but must be informative enough to reduce expansions.

    Heuristic Initialization
    ------------------------
    - Extract:
      - The goal wuqjn5r5 fact’s kukxix7x and z1mgqxkc objects (fixed).
      - Sets of ajcpzwcf objects (these control some states).
      - The unique object currently having odqd6scj true.
      - The static transition graph given by k21e3pdq facts of shape
        (k21e3pdq source target label), representing possible moves labeled by yrvdqfln.
      - A reverse index from a z1mgqxkc to all sources reachable by one edge,
        to efficiently compute distances.
      - The goal fact's second argument (target z1mgqxkc for wuqjn5r5).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the state already satisfies the goal (contains the goal wuqjn5r5 fact), return 0.
    2. Identify current odqd6scj location: must move this marker progressively to the target z1mgqxkc in the goal wuqjn5r5 fact.
    3. ajcpzwcf flags on z1mgqxkc objects control applicability of action yaz3jy7e, which "teleports" or upgrades wuqjn5r5 and ajcpzwcf predicates along edges.
    4. The heuristic simulates a shortest path calculation from the current odqd6scj node to the goal target node along available transitions:
       - Only transitions allowed by static k21e3pdq facts.
       - Paths need available ajcpzwcf "flags" or must flip them by spending one action.
    5. Because "distance" in terms of odqd6scj flips is crucial, the heuristic approximates:
       - Distance from current odqd6scj node to the goal target node in the transition graph.
       - The number of ajcpzwcf objects missing on the path (because yaz3jy7e requires ajcpzwcf true on the new node).
    6. The heuristic sums:
       - The distance (minimal number of gww2bqr6 or yaz3jy7e transitions) to move odqd6scj to goal node,
       - Plus potential costs for flipping ajcpzwcf on needed nodes,
       - Plus at least one action (yaz3jy7e) to finalize wuqjn5r5 at the goal node.
    7. It does not attempt full state-space search but uses a BFS over static graph and ajcpzwcf presence to estimate steps.
    8. The heuristic thus bounds the minimal required actions assuming ideal availability of ajcpzwcf flags.

    This heuristic balances informativeness and computational efficiency, ensuring finite estimates on solvable states
    and zero only on goal states.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract k21e3pdq edges: (source_z1mgqxkc, target_z1mgqxkc, label_yrvdqfln)
        self.edges = []  # list of triples (src, tgt, label)
        for fact in task.static:
            if match(fact, "k21e3pdq", "*", "*", "*"):
                parts = get_parts(fact)
                self.edges.append((parts[1], parts[2], parts[3]))

        # Build graph adjacency (directed) from source to target ignoring labels (for shortest path search).
        # We'll also index edges by source to target with label for reachability checks.
        self.graph = {}
        for src, tgt, label in self.edges:
            self.graph.setdefault(src, set()).add(tgt)

        # Extract goal wuqjn5r5 fact, assuming 1 goal fact only and of the form (wuqjn5r5 kukxix7x z1mgqxkc)
        # We get the kukxix7x constant (fixed) and the goal z1mgqxkc object.
        self.goal_wuqjn5r5 = None
        for goal_fact in self.goals:
            if match(goal_fact, "wuqjn5r5", "*", "*"):
                parts = get_parts(goal_fact)
                self.goal_wuqjn5r5 = (parts[1], parts[2])
                break
        assert self.goal_wuqjn5r5 is not None, "Goal must contain wuqjn5r5 fact"

        self.goal_kukxix7x = self.goal_wuqjn5r5[0]  # fixed kukxix7x object in goal
        self.goal_z1mgqxkc = self.goal_wuqjn5r5[1]  # target z1mgqxkc node for wuqjn5r5 goal

    def __call__(self, node):
        state = node.state

        # 1. If goal is reached, heuristic=0
        if self.goals <= state:
            return 0

        # 2. Extract current odqd6scj node (should be unique)
        odqd_nodes = [get_parts(f)[1] for f in state if match(f, "odqd6scj", "*")]
        if not odqd_nodes:
            # No current odqd6scj: must activate it first - heuristic at least 1
            # But since odqd6scj is involved in all transitions, count 1
            odqd_node = None
        else:
            odqd_node = odqd_nodes[0]

        # 3. Extract set of ajcpzwcf flagged nodes
        ajcp_nodes = {get_parts(f)[1] for f in state if match(f, "ajcpzwcf", "*")}

        # 4. Extract current wuqjn5r5 for fixed kukxix7x from the state (should be unique)
        current_wuqjn5r5_nodes = {get_parts(f)[2] for f in state if match(f, "wuqjn5r5", self.goal_kukxix7x, "*")}
        # Usually only one node with wuqjn5r5 for this kukxix7x exists
        # If none, heuristic will consider from odqd node and ajcp flags

        # Note: wuqjn5r5 predicate is moved by yaz3jy7e action setting at new node, so target is goal_z1mgqxkc

        # Early heuristic: if wuqjn5r5 already at goal node, return 0 (goal check above covers most)
        if self.goal_z1mgqxkc in current_wuqjn5r5_nodes:
            return 0

        # 5. We want to estimate minimal actions to:
        #    - Move odqd6scj from current location to goal_z1mgqxkc by edges.
        #    - Have ajcpzwcf flag at next node before movement (yaz3jy7e requires ajcpzwcf on target node).
        #    - Possibly flip ajcpzwcf flags by spending one action (yaz3jy7e flips ajcpzwcf flags on two nodes).
        #    - Finally execute yaz3jy7e to get wuqjn5r5 at goal node.

        # Approach:
        # We'll perform a BFS on the graph from odqd_node to goal_z1mgqxkc,
        # tracking ajcpzwcf presence to estimate cost of raising flag if missing,
        # and count steps as sum of action costs needed per edge traversal.

        # For states without odqd6scj node defined, return heuristic 1 as minimal effort (arbitrary choice).

        if odqd_node is None:
            return 1

        from collections import deque

        # State per BFS: (current_node, ajcp_flag_present_at_current)
        # Because ajcpzwcf controls applicability of action yaz3jy7e,
        # being flagged on current or neighbor impacts cost.

        # We aim to reach goal_z1mgqxkc node.

        # Track visited: dictionary mapping (node, ajcp_flag_boolean) -> cost
        visited = {}

        # Initial ajcpzwcf flag presence on odqd_node
        ajcp_on_start = odqd_node in ajcp_nodes

        queue = deque()
        queue.append((odqd_node, ajcp_on_start, 0))
        visited[(odqd_node, ajcp_on_start)] = 0

        minimal_cost = None

        while queue:
            curr_node, ajcp_flag, cost_so_far = queue.popleft()

            if curr_node == self.goal_z1mgqxkc:
                # We estimate cost to get odqd6scj at goal node.
                # After reaching, one yaz3jy7e action at least is needed to finalize wuqjn5r5 update.
                final_cost = cost_so_far + 1  # +1 for yaz3jy7e action to set wuqjn5r5 at goal node
                if minimal_cost is None or final_cost < minimal_cost:
                    minimal_cost = final_cost
                continue

            # Try to traverse edges from curr_node
            for neighbor in self.graph.get(curr_node, []):
                neighbor_flag = neighbor in ajcp_nodes

                # Action costs:
                # gww2bqr6 can move odqd6scj when ajcpzwcf at target is set (?), 
                # but per domain definition, gww2bqr6 requires ajcpzwcf at target (in preconditions),
                # because precondition includes (ajcpzwcf ?xo2y4) in gww2bqr6's preconditions [actually no].
                #
                # Reviewing domain actions:
                # gww2bqr6 preconditions:
                #   (odqd6scj ?q3jjz)
                #   (k21e3pdq ?q3jjz ?xo2y4 ?clx7v)
                #   (ajcpzwcf ?xo2y4)
                # So target node must be ajcpzwcf flagged to use gww2bqr6.
                #
                # yaz3jy7e action allows:
                #   switches wuqjn5r5 from old to new locations and flips ajcpzwcf,
                #   allowing to move ajcpzwcf flag to relevant next node.
                #
                # So in heuristic, traveling from curr_node to neighbor requires:
                # - If ajcpzwcf at neighbor is True: one action (gww2bqr6) to move odqd6scj.
                # - If ajcpzwcf at neighbor is False: first a yaz3jy7e action to "flip" flags + one gww2bqr6 action.
                #
                # Each yaz3jy7e flips ajcpzwcf from old to new node, must be considered, but for heuristic simple counting:
                # assume 2 steps if ajcpzwcf at neighbor missing (one yaz3jy7e + one gww2bqr6),
                # else 1 step if ajcpzwcf present (gww2bqr6 only).

                # To be even more effective, ajcpzwcf flag is actually flipped by yaz3jy7e,
                # so after flipping, neighbor will have ajcpzwcf, potentially enabling further moves.

                if neighbor_flag:
                    step_cost = 1  # Just gww2bqr6
                    next_flag = True  # ajcpzwcf at neighbor remains True
                else:
                    # Need yaz3jy7e to flip ajcpzwcf flag here before moving odqd6scj
                    # Yaz3jy7e switches ajcpzwcf from some node to another and also moves wuqjn5r5.
                    # After yaz3jy7e, ajcpzwcf on curr_node gone, neighbor gets it.
                    # Hence, ajcpzwcf at curr_node lost, so after this move ajcpzwcf is on neighbor only.
                    # So after this move ajcpzwcf flag will be True on neighbor, False on curr_node.
                    step_cost = 2  # yaz3jy7e + gww2bqr6
                    next_flag = True

                new_cost = cost_so_far + step_cost

                state_key = (neighbor, next_flag)
                if state_key not in visited or new_cost < visited[state_key]:
                    visited[state_key] = new_cost
                    queue.append((neighbor, next_flag, new_cost))

        # If no path found, return a large heuristic value to encourage pruning.
        # But must be finite for solvable states; here we return a large fixed upper bound.
        if minimal_cost is None:
            return 1000

        return minimal_cost
