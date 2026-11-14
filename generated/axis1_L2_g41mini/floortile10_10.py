from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with support for wildcards.

    Args:
        fact (str): The fact string, e.g. "(predicate arg1 arg2)".
        args (str): Pattern elements, e.g. "predicate", "*", "obj".
    
    Returns:
        bool: True if fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (hi4l5zmb).

    Summary
    -------
    This heuristic estimates the number of actions required to achieve all goal facts of the form
    (cb6fw2xd <object_nh1sy8v9> <object_qekv4465>) — the goal predicate — from the current state.
    It operates by analyzing which goal facts are not yet satisfied and estimating the cost
    to transform the state accordingly.

    Assumptions
    -----------
    - The only goals are conjunctions of (cb6fw2xd ?nh1sy8v9 ?qekv4465) facts.
    - Facts and effects depend heavily on the spatial relations between nh1sy8v9 objects and qekv4465 resources.
    - Transporting or "moving" the key entities between nh1sy8v9 objects is central.
    - The predicate (fguw2142 ?nh1sy8v9) marks some "position" or "free" property which actions manipulate.
    - (ypt9mrgu ?k9b7lknr ?nh1sy8v9) relates entities of type k9b7lknr to nh1sy8v9 objects and signify ownership or presence.
    - Actions move or swap the presence of these entities or their associated resources.
    - The heuristic focuses on how many (cb6fw2xd ... ) goal facts remain false in state, estimating 1 or 2 actions per fact not achieved depending on related conditions.
    
    Heuristic Initialization
    ------------------------
    - Extract the goal facts involving cb6fw2xd.
    - Extract static binary relations s4jrpk2b, vom4pfig, vjcqgdgn, ga4l9do0 that define adjacency or connectivity between nh1sy8v9 objects.
    - Extract facts marking which qekv4465 objects satisfy (fezojtj3).
    - Prepare data structures to quickly test connectivity and presence.
    - This information allows identifying if an entity is plausibly reachable or how many steps propagate from current positions.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify all goal facts of form (cb6fw2xd nh1sy8v9 qekv4465).

    2. For each such goal fact, check if it currently holds in the state. If yes, no cost for that fact.

    3. For goal facts not in the state:
       a) Check if the nh1sy8v9 object currently has the key property (fguw2142).
       b) Check if the respective k9b7lknr entity associated with the qekv4465 resource can be identified or is currently related through predicates ypt9mrgu or qp2t6u4s.
       c) Check distance or adjacency between the nh1sy8v9 object and the location of the entity or the resource using static predicates:
          - s4jrpk2b, vom4pfig, vjcqgdgn, ga4l9do0 represent different neighborhood relations or connectivity.
       d) Depending on presence and adjacency, estimate how many actions to "bring" or "carry" the resource into the goal nh1sy8v9 object for cb6fw2xd to hold.

    4. The heuristic value sums, for all unmet goal facts, estimated minimal costs, usually 1-3 actions per goal element depending on connectivity and current holding state. 
       This is a domain-dependent informed guess minimizing node expansions, not guaranteed admissible.

    5. Return 0 if all goal facts are satisfied (goal state), else sum the estimates.
    """

    def __init__(self, task):
        # Save goal facts with predicate cb6fw2xd
        self.goals_cb6fw2xd = {
            fact for fact in task.goals if match(fact, "cb6fw2xd", "*", "*")
        }

        static = task.static

        # Store connectivity relations for quick adjacency testing:
        self.s4jrpk2b = set(
            (get_parts(fact)[1], get_parts(fact)[2]) for fact in static if match(fact, "s4jrpk2b", "*", "*")
        )
        self.vom4pfig = set(
            (get_parts(fact)[1], get_parts(fact)[2]) for fact in static if match(fact, "vom4pfig", "*", "*")
        )
        self.vjcqgdgn = set(
            (get_parts(fact)[1], get_parts(fact)[2]) for fact in static if match(fact, "vjcqgdgn", "*", "*")
        )
        self.ga4l9do0 = set(
            (get_parts(fact)[1], get_parts(fact)[2]) for fact in static if match(fact, "ga4l9do0", "*", "*")
        )

        # Static property indicating qekv4465 objects of type fezojtj3
        self.fezojtj3 = {get_parts(fact)[1] for fact in static if match(fact, "fezojtj3", "*")}

    def __call__(self, node):
        state = node.state

        # Quick membership sets for faster lookup
        state_fguw2142 = {get_parts(fact)[1] for fact in state if match(fact, "fguw2142", "*")}
        state_ypt9mrgu = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "ypt9mrgu", "*", "*")}
        state_qp2t6u4s = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "qp2t6u4s", "*", "*")}
        state_cb6fw2xd = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "cb6fw2xd", "*", "*")}

        # For heuristic calculation:
        # For every goal (cb6fw2xd ?nh1sy8v9 ?qekv4465) fact not yet true, estimate cost:
        # Cost depends on whether the qekv4465 object belongs to "fezojtj3",
        # whether nh1sy8v9 objects have fguw2142 (availability),
        # and on adjacency relations linking nh1sy8v9 objects that hold ypt9mrgu relation for some k9b7lknr entities associated with the qekv4465.
        # Steps:
        # - To achieve (cb6fw2xd nh1sy8v9 qekv4465), the qekv4465 resource must be "placed" at nh1sy8v9, which requires transporting it or enabling ownership (ypt9mrgu).
        # We estimate action counts from:
        # (a) distance or adjacency from current owner to goal position,
        # (b) availability of "tokens" fguw2142 to enable actions.
        # Because domain predicates and actions are complex, we use conservative heuristic:
        # For each missing goal fact: cost starts at 1 (to produce cb6fw2xd).
        # Add 1 more if the qekv4465 resource is not currently associated with the nh1sy8v9 via any of the predicates indicating ownership or presence (ypt9mrgu or qp2t6u4s).
        # Add 1 more if the goal nh1sy8v9 object is not currently "free" (fguw2142), implying extra moves to free it.
        # Add an adjacency penalty of 0 or 1 based on whether any adjacency relation links the goal nh1sy8v9 to an nh1sy8v9 which has the resource or ownership.
        # This sum yields a domain-informed heuristic that favors states closer to moving needed resources into goals.

        total_cost = 0

        for goal_fact in self.goals_cb6fw2xd:
            _, nh_obj, qekv_obj = get_parts(goal_fact)

            # If goal fact is already satisfied, no cost.
            if (nh_obj, qekv_obj) in state_cb6fw2xd:
                continue

            cost = 1  # Base cost to achieve cb6fw2xd at least.

            # Check if qekv_obj is a known fezojtj3 object (some resource)
            is_fezo = qekv_obj in self.fezojtj3

            # Check if nh_obj currently has fguw2142 property (indicates "free" or "active")
            free_at_nh = nh_obj in state_fguw2142
            if not free_at_nh:
                cost += 1  # Need to free or swap fguw2142 at nh_obj to place resource.

            # Check if there exists some k9b7lknr entity en18p so that (ypt9mrgu en18p nh_obj) and (qp2t6u4s en18p qekv_obj) hold currently.
            # That is: does the owner of qekv_obj appear at nh_obj, enabling placement directly?
            owner_at_nh = False
            for en18p, stq0j in state_ypt9mrgu:
                if stq0j == nh_obj and (en18p, qekv_obj) in state_qp2t6u4s:
                    owner_at_nh = True
                    break

            if not owner_at_nh:
                # If owner not at goal location, need additional actions to move or transfer ownership.
                cost += 1

                # Additionally, check adjacency relations linking nh_obj to other nh1sy8v9 objects that do have the resource:
                # For any nhx linked to nh_obj via s4jrpk2b, vom4pfig, vjcqgdgn, or ga4l9do0,
                # if owner is at nhx and fguw2142 is at nhx, then cost could be less.
                adjacency_found = False
                # Collect all nh1sy8v9 neighbors based on static adjacency relations:
                neighbors = set()
                neighbors.update(n2 for (n1, n2) in self.s4jrpk2b if n1 == nh_obj)
                neighbors.update(n1 for (n1, n2) in self.s4jrpk2b if n2 == nh_obj)
                neighbors.update(n2 for (n1, n2) in self.vom4pfig if n1 == nh_obj)
                neighbors.update(n1 for (n1, n2) in self.vom4pfig if n2 == nh_obj)
                neighbors.update(n2 for (n1, n2) in self.vjcqgdgn if n1 == nh_obj)
                neighbors.update(n1 for (n1, n2) in self.vjcqgdgn if n2 == nh_obj)
                neighbors.update(n2 for (n1, n2) in self.ga4l9do0 if n1 == nh_obj)
                neighbors.update(n1 for (n1, n2) in self.ga4l9do0 if n2 == nh_obj)

                # Check if owner with resource and free token is at any neighbor
                for nhx in neighbors:
                    if nhx in state_fguw2142:
                        for en18p, stq0j in state_ypt9mrgu:
                            if stq0j == nhx and (en18p, qekv_obj) in state_qp2t6u4s:
                                adjacency_found = True
                                break
                    if adjacency_found:
                        break

                if not adjacency_found:
                    cost += 1  # Further penalty for longer transport (no adjacency shortcut)

            total_cost += cost

        return total_cost
