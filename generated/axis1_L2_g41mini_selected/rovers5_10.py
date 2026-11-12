from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate obj1 obj2)".
    - `args`: The expected pattern with wildcards `*` allowed.
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain domain (rb7qr1fs).

    Summary:
        This heuristic estimates the minimal number of actions required to reach
        the goal state by focusing on reaching key facts related to the goal predicates
        'ixsb7u9c' and 'y01m0fhm', which appear prominently in the goal descriptions.
        It analyzes the current state to identify how many required 'ixsb7u9c' and 'y01m0fhm'
        goal facts are still missing, and uses a heuristic cost model based on the domain's
        actions to estimate step costs for achieving these facts.

    Assumptions:
        - The goals always consist mainly of fulfilling predicates of the types 'ixsb7u9c' and 'y01m0fhm'.
        - Achieving 'ixsb7u9c' facts appears to require producing certain intermediate facts like 'r1frsf5x'
          for matching objects, and satisfying certain static relationships.
        - Achieving 'y01m0fhm' facts corresponds loosely to enabling 'y01m0fhm' predicates on certain objects,
          potentially requiring activation of 'w1yp6zl7' and related predicates.
        - Static facts (like 'b2mwd6x4' relationships) and unary predicates like 'jckjsxca' are used as connectivity or membership info.
        - Actions support moving or toggling relations between objects of different types; the heuristic
          estimates costs by counting unfulfilled goal facts and the number of required intermediate steps.

    Heuristic Initialization:
        - Extract and memorize goal facts separately by their predicate ('ixsb7u9c' or 'y01m0fhm').
        - Extract static facts to build quick lookup dictionaries for:
          * Static connections between objects of type rb7qr1fs and uwq7ubjf (from 'b2mwd6x4' and 'wx95011q' facts),
            which relate to movements or linking uwq7ubjf objects within rb7qr1fs.
          * Maps from uwq7ubjf objects to their connected uwq7ubjf neighbors (from 'wx95011q') to estimate reachability.
        - Store static membership of objects in rb7qr1fs clusters via predicates like 'bszmiquc' and 'a298chgk'.

    Step-By-Step Thinking for Computing Heuristic:
        1. If the current state already satisfies all goals, return 0.
        2. Count how many 'ixsb7u9c' goal facts are not satisfied in the current state.
           Each missing 'ixsb7u9c' fact roughly requires:
             - Producing the corresponding 'r1frsf5x' fact, which normally requires:
               > Satisfying 'wfrx4exs' with a tom0ncko object and rb7qr1fs cluster.
               > Proper connections via 'jckjsxca' (membership), 'iaub97vi', and static connectivity.
             - Leading a rough cost estimate of about 3-5 actions per missing 'ixsb7u9c' goal fact.
        3. Count how many 'y01m0fhm' goal facts are not satisfied.
           Each missing 'y01m0fhm' fact often requires enabling 'w1yp6zl7' for some connected uwq7ubjf,
           plus toggling interrelated predicates. We approximate 3 actions per such missing fact.
        4. Consider also connected structure costs to estimate group synchronization or movements
           needed for enabling multiple goals: for example, the number of distinct clusters or groups
           active that influence goals.
        5. Sum the estimates ensuring the heuristic is finite and reasonably informed.

        Ultimately, this heuristic is a weighted sum of the number of missing key goal facts,
        scaled by approximate minimal action counts to achieve the respective predicates.
        It does not perform heavy search or exact inference but uses static connectivity
        and known goal predicates to guide the greedy best-first search effectively.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract goals by predicate type
        self.goal_ixsb7u9c = set()
        self.goal_y01m0fhm = set()
        for fact in self.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "ixsb7u9c":
                self.goal_ixsb7u9c.add(fact)
            elif pred == "y01m0fhm":
                self.goal_y01m0fhm.add(fact)

        # Static maps for quick lookup
        # Map rb7qr1fs -> set of uwq7ubjf connected by b2mwd6x4
        self.rb7qr1fs_to_uwq7ubjf = {}
        for fact in self.static:
            if match(fact, "b2mwd6x4", "*", "*", "*"):
                parts = get_parts(fact)
                r, u1, u2 = parts[1], parts[2], parts[3]
                self.rb7qr1fs_to_uwq7ubjf.setdefault(r, set()).add(u1)
                self.rb7qr1fs_to_uwq7ubjf.setdefault(r, set()).add(u2)

        # Map uwq7ubjf -> set of connected uwq7ubjf from wx95011q
        self.uwq7ubjf_connected = {}
        for fact in self.static:
            if match(fact, "wx95011q", "*", "*"):
                parts = get_parts(fact)
                u1, u2 = parts[1], parts[2]
                self.uwq7ubjf_connected.setdefault(u1, set()).add(u2)
                self.uwq7ubjf_connected.setdefault(u2, set()).add(u1)

        # Cache all rb7qr1fs clusters with 'bszmiquc' or 'a298chgk' (we assume they represent active/belonging clusters)
        self.bszmiquc_set = {get_parts(f)[1] for f in self.static if match(f, "bszmiquc", "*")}
        self.a298chgk_set = {get_parts(f)[1] for f in self.static if match(f, "a298chgk", "*")}

    def __call__(self, node):
        state = node.state

        # Goal reached? heuristic 0
        if self.goals <= state:
            return 0

        # Count missing ixsb7u9c goal facts
        missing_ixsb7u9c = 0
        for fact in self.goal_ixsb7u9c:
            if fact not in state:
                missing_ixsb7u9c += 1

        # Count missing y01m0fhm goal facts
        missing_y01m0fhm = 0
        for fact in self.goal_y01m0fhm:
            if fact not in state:
                missing_y01m0fhm += 1

        # Heuristic estimate parameters (chosen conservatively)
        # Each ixsb7u9c missed requires approx 4 actions: creating intermediate facts, enabling related predicates.
        cost_per_ixsb7u9c = 4
        # Each y01m0fhm missed requires approx 3 actions: enabling w1yp6zl7 and toggling related predicates.
        cost_per_y01m0fhm = 3

        # Consider possible multiplier for complexity based on number of distinct rb7qr1fs clusters active
        # Count how many clusters (rb7qr1fs) currently have membership predicates in state ('jckjsxca')
        clusters_active = set()
        for fact in state:
            if match(fact, "jckjsxca", "*", "*"):
                parts = get_parts(fact)
                clusters_active.add(parts[1])

        # Add a multiplier to scale costs mildly by cluster involvement, capped at 2
        cluster_factor = min(2, 1 + 0.1 * len(clusters_active))

        heuristic_value = cluster_factor * (
            missing_ixsb7u9c * cost_per_ixsb7u9c + missing_y01m0fhm * cost_per_y01m0fhm
        )

        # Ensure heuristic is at least 1 (if non-goal) to avoid zero cost traps except exact goal
        return max(1, int(heuristic_value))
