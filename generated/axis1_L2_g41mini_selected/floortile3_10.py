from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by stripping parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a pattern (with wildcards '*').

    - fact: string representing a fact, e.g. '(predicate arg1 arg2)'.
    - args: pattern parts, e.g. 'predicate', '*', 'arg2'.
    Return True if all parts match the pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain (hi4l5zmb).

    Summary
    -------
    This heuristic estimates the number of actions needed to achieve all goal facts of the form
    `(cb6fw2xd ?stq0j - nh1sy8v9 ?tlfqu - qekv4465)` in the goal state. Achieving such a goal fact
    represents placing a certain object of type nh1sy8v9 with a qekv4465 object via an action
    effect that requires a series of steps involving certain conditions on "fguw2142" and "ypt9mrgu"
    predicates.

    The heuristic tries to approximate how many moves or re-locations are needed to get each goal
    fact `(cb6fw2xd stq0j tlfqu)` true from the current state by analyzing:
    - Whether the `(cb6fw2xd)` fact exists already (0 cost),
    - If not, whether the `stq0j` location/object currently holds the "fguw2142" flag (some
      "free" marker) indicating it can be used as a staging point,
    - Whether the `ypt9mrgu ?en18p ?stq0j` relationship holds, i.e., which `k9b7lknr` object is
      associated with the location,
    - How close or reachable the needed conditions are via static predicates `ga4l9do0`, `s4jrpk2b`,
      `vom4pfig`, `vjcqgdgn` which relate nh1sy8v9 objects.

    By counting how many objects have not yet reached their corresponding goal `(cb6fw2xd ...)`
    condition and estimating the minimal steps needed to bring their state to the goal, a heuristic
    estimate is computed.

    Assumptions
    -----------
    - The goals only contain `(cb6fw2xd nh1sy8v9 qekv4465)` facts whose precursor facts must be
      manipulated by the actions described.
    - `fguw2142 ?stq0j` is a "free" or "empty" marker used for moving or swapping positions.
    - `ypt9mrgu ?en18p ?stq0j` indicates that object of type k9b7lknr is at nh1sy8v9 location.
    - Static relations such as `ga4l9do0`, `s4jrpk2b`, `vom4pfig`, and `vjcqgdgn` express
      adjacency or connectivity, relevant for reasoning about how to bring objects closer.
    - The heuristic permits non-admissible estimates prioritizing fast guidance.

    Heuristic Initialization
    ------------------------
    - Extract the set of goal `(cb6fw2xd stq0j tlfqu)` facts to track which `(nh1sy8v9, qekv4465)` pairs
      need to be achieved.
    - Store static connectivity relations for adjacency checks:
      - `ga4l9do0` and `s4jrpk2b` are relations between nh1sy8v9 objects.
      - `vjcqgdgn` and `vom4pfig` also map nh1sy8v9 relations, indicating possible moves.
    - Store all `(fezojtj3 <qekv4465>)` facts for relevant qekv4465 objects involved.
    - Build reverse indices for quick checking adjacency from any nh1sy8v9 object.
    - Identify which `k9b7lknr` objects are involved via `ypt9mrgu` relations.

    Step-By-Step Thinking for Computing Heuristic
    -----------------------------------------------
    1. Quick goal check: If current state includes all goal `(cb6fw2xd ...)` facts, heuristic = 0.
    2. For each goal `(cb6fw2xd stq0j tlfqu)` not yet achieved:
       a) Check if `(fguw2142 stq0j)` holds in state; if not, estimate at least 1 action to free it.
       b) Check which `k9b7lknr` object is currently linked to `stq0j` via `(ypt9mrgu en18p stq0j)`.
       c) If `k9b7lknr` currently linked is not the "goal carrier" or mismatched, estimate cost to relocate.
       d) Use static adjacency (`ga4l9do0`, `s4jrpk2b`, `vom4pfig`, `vjcqgdgn`)
          to estimate the minimal number of moves/swaps needed to setup this `(cb6fw2xd ...)` goal.
    3. Sum heuristic costs over all missing `(cb6fw2xd ...)` goal facts.
    4. Return sum as heuristic estimate.

    This approach balances accuracy and efficiency by using static adjacency information and current
    state configurations to guess a lower bound of actions needed. It aims to expand fewer search nodes
    by preferring states closer to establishing `(cb6fw2xd ...)` facts.

    Note: Details like the exact minimal number of moves are approximated by connectivity counts and
    presence of free spots due to complexity of exact distance computations.
    """

    def __init__(self, task):
        self.goals = task.goals  # Goal facts (a frozenset)
        self.static = task.static  # Static facts (a frozenset)

        # Extract goal (cb6fw2xd nh1sy8v9 qekv4465) facts
        self.goal_cb6fw2xd = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "cb6fw2xd" and len(parts) == 3:
                self.goal_cb6fw2xd.add((parts[1], parts[2]))

        # Extract static adjacency relations between nh1sy8v9 objects
        # We will treat these as undirected edges in a graph
        self.adj_ga4l9do0 = set()
        self.adj_s4jrpk2b = set()
        self.adj_vjcqgdgn = set()
        self.adj_vom4pfig = set()

        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "ga4l9do0":
                self.adj_ga4l9do0.add((parts[1], parts[2]))
                self.adj_ga4l9do0.add((parts[2], parts[1]))
            elif pred == "s4jrpk2b":
                self.adj_s4jrpk2b.add((parts[1], parts[2]))
                self.adj_s4jrpk2b.add((parts[2], parts[1]))
            elif pred == "vjcqgdgn":
                self.adj_vjcqgdgn.add((parts[1], parts[2]))
            elif pred == "vom4pfig":
                self.adj_vom4pfig.add((parts[1], parts[2]))

        # We also need to map which locations/types each k9b7lknr object is linked to
        # Extract all k9b7lknr objects that appear in ypt9mrgu relations in static (if any)
        self.k9b7lknr_objects = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "ypt9mrgu" and len(parts) == 3:
                self.k9b7lknr_objects.add(parts[1])

        # Extract fezojtj3 qekv4465 facts present in static (possible markers for qekv4465)
        self.fezojtj3 = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "fezojtj3" and len(parts) == 2:
                self.fezojtj3.add(parts[1])

    def __call__(self, node):
        state = node.state

        # If goal is reached, heuristic is 0
        if self.goals <= state:
            return 0

        # Build maps from current state:
        # - fguw2142 (free) sets for nh1sy8v9 objects
        free_nh1sy8v9 = set(
            parts[1]
            for fact in state
            if (parts := get_parts(fact))[0] == "fguw2142" and len(parts) == 2
        )

        # Map from nh1sy8v9 location to k9b7lknr object for ypt9mrgu
        location_to_carrier = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "ypt9mrgu" and len(parts) == 3:
                en18p, stq0j = parts[1], parts[2]
                location_to_carrier[stq0j] = en18p

        # Set of achieved cb6fw2xd facts currently in state
        achieved_cb6fw2xd = set(
            (parts[1], parts[2])
            for fact in state
            if (parts := get_parts(fact))[0] == "cb6fw2xd" and len(parts) == 3
        )

        heuristic_estimate = 0

        # For each goal (stq0j, tlfqu), estimate the cost if not achieved yet
        for stq0j, tlfqu in self.goal_cb6fw2xd:
            if (stq0j, tlfqu) in achieved_cb6fw2xd:
                # Already achieved, no cost
                continue

            # Base cost estimate per missing goal predicate
            cost = 0

            # 1) Check if stq0j is free (fguw2142)
            if stq0j not in free_nh1sy8v9:
                # Need to free it first, so at least one action
                cost += 1

            # 2) Check which k9b7lknr object is currently at stq0j
            current_carrier = location_to_carrier.get(stq0j, None)

            # 3) Heuristic: If no carrier or wrong one, estimate moves needed to relocate
            # We do not know the exact en18p that should carry tlfqu, but we know at goal
            # cb6fw2xd exists only if corresponding carrier moved item. We use presence of any carrier as a hint.

            if current_carrier is None:
                # No carrier at location, at least 2 moves: get carrier here + place tlfqu
                cost += 2
            else:
                # Carrier present: estimate cost to put correct tlfqu here
                # If carrier changed from the initial distribution could require 1 move
                # We approximate this by adding 1 if carrier is present (assuming still need to set tlfqu)
                cost += 1

            # 4) Additional cost estimation based on adjacency
            # Check if stq0j is adjacent to any free location (free_nh1sy8v9) for quick moves
            adjacent_free_found = any(
                (stq0j, adj) in self.adj_ga4l9do0 or
                (stq0j, adj) in self.adj_s4jrpk2b or
                (stq0j, adj) in self.adj_vjcqgdgn or
                (stq0j, adj) in self.adj_vom4pfig
                for adj in free_nh1sy8v9
            )

            if not adjacent_free_found:
                # If no adjacent free location, more costly to rearrange, add penalty
                cost += 1

            heuristic_estimate += cost

        # Return the heuristic estimate as an integer >= 0
        return heuristic_estimate
