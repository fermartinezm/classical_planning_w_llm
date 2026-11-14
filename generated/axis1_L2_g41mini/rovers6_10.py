from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact, removing parentheses and splitting by space."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Supports wildcard matching using fnmatch.

    :param fact: fact string, e.g. "(predicate arg1 arg2)"
    :param args: pattern tokens to match against fact components
    :return: True if match, else False
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (rb7qr1fs).

    Summary
    -------
    This heuristic estimates the number of actions required to reach the goal state by analyzing:
    - For each (r6gw2, pmq6p) goal fact "(ixsb7u9c r6gw2 pmq6p)", how many such goals remain unsatisfied.
    - It tries to estimate the minimum steps to produce these goals by examining the related intermediate predicates 
      and dependencies on known static relations and current state facts.
    - The heuristic computes a weighted sum of missing goal predicate instances plus penalties for unmet prerequisite conditions.

    Assumptions
    -----------
    - The main goals are facts with predicate "ixsb7u9c" specifying a pair of objects (r6gw2 and pmq6p).
    - Achieving each such goal requires establishing corresponding preconditions as can be inferred from operators.
    - Static facts and predicates related to "b2mwd6x4" and "wx95011q" represent connection or reachability relations that allow moving/manipulating objects.
    - Each missing goal fact roughly costs one or more actions, plus additional cost for prerequisite conditions.
    - The heuristic is not guaranteed admissible but attempts to accurately reflect subgoal dependencies to reduce expansion.

    Heuristic Initialization
    ------------------------
    - Parse and store the set of goal "ixsb7u9c" facts and their argument pairs.
    - Extract static relations from static facts that can help infer connectivity or preconditions:
      * "b2mwd6x4" relations (rb7qr1fs, uwq7ubjf, uwq7ubjf)
      * "wx95011q" relations (uwq7ubjf, uwq7ubjf)
      * Other static unary and binary predicates used by operators preconditions.
    - These static sets help quickly check relational conditions pertinent to actions affecting the goals.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify all goal predicate facts for "ixsb7u9c", i.e., (r6gw2, pmq6p) pairs to achieve.
    2. Check which goal facts are already satisfied in the current state (cost 0 for those).
    3. For each unsatisfied goal fact:
       a. Check if prerequisite facts (like "r1frsf5x" linking goal rb7qr1fs to objects) exist in the current state.
       b. Verify whether required static connectivity facts like b2mwd6x4 and wx95011q hold to ensure the goal is reachable.
       c. Estimate the minimum number of steps to produce the goal:
          - At least 1 action to add the goal fact itself.
          - Add 1 extra for each missing prerequisite or intermediate predicate that the operators require to establish the goal fact.
    4. Sum these estimated costs for all missing goal facts.
    5. Return the sum as heuristic value.

    This approach mirrors dependencies observed in the domain actions, e.g., transferring objects, establishing intermediate predicates,
    and the goal requirements on predicate "ixsb7u9c".

    The heuristic returns zero if the state already satisfies all goal conditions.
    """

    def __init__(self, task):
        # Extract all goal "ixsb7u9c" facts and store pairs for fast lookup.
        self.goal_ixsb7u9c_pairs = set()
        for fact in task.goals:
            parts = get_parts(fact)
            if parts[0] == "ixsb7u9c" and len(parts) == 3:
                self.goal_ixsb7u9c_pairs.add((parts[1], parts[2]))

        static_facts = task.static

        # Extract relevant static relations for fast membership tests.
        # Relations b2mwd6x4(rb7qr1fs, uwq7ubjf, uwq7ubjf) and wx95011q(uwq7ubjf, uwq7ubjf)
        self.b2mwd6x4 = set()
        self.wx95011q = set()
        self.a8x3330w = set()   # unary predicate for uwq7ubjf (used in some preconditions)
        self.p6u7tzz0 = set()   # unary predicate for zp2z4mnx (often used in preconditions and effects)
        self.a298chgk = set()   # unary predicate for rb7qr1fs
        self.bszmiquc = set()   # unary predicate for rb7qr1fs
        self.rrhxwbyt = set()   # binary predicate (ggaeyv41, uwq7ubjf)
        self.ipivzc5v = set()   # unary predicate uwq7ubjf
        self.z6yvwl7n = set()   # binary predicate (zp2z4mnx, rb7qr1fs)
        self.y01m0fhm = set()   # unary predicate uwq7ubjf
        self.df hfijzr = set()  # unary predicate rb7qr1fs
        self.r1frsf5x = set()   # ternary predicate (rb7qr1fs, dbk5i2a5, chxau8fy)

        # Fill predicate sets from static facts only those predicates that appear in static.
        # Note that some relevant predicates can also appear in state; only static predicates given here.
        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "b2mwd6x4" and len(parts) == 4:
                self.b2mwd6x4.add((parts[1], parts[2], parts[3]))
            elif pred == "wx95011q" and len(parts) == 3:
                self.wx95011q.add((parts[1], parts[2]))
            elif pred == "a8x3330w" and len(parts) == 2:
                self.a8x3330w.add(parts[1])
            elif pred == "p6u7tzz0" and len(parts) == 2:
                self.p6u7tzz0.add(parts[1])
            elif pred == "a298chgk" and len(parts) == 2:
                self.a298chgk.add(parts[1])
            elif pred == "bszmiquc" and len(parts) == 2:
                self.bszmiquc.add(parts[1])
            elif pred == "rrhxwbyt" and len(parts) == 3:
                self.rrhxwbyt.add((parts[1], parts[2]))
            elif pred == "ipivzc5v" and len(parts) == 2:
                self.ipivzc5v.add(parts[1])
            elif pred == "z6yvwl7n" and len(parts) == 3:
                self.z6yvwl7n.add((parts[1], parts[2]))
            elif pred == "y01m0fhm" and len(parts) == 2:
                self.y01m0fhm.add(parts[1])
            elif pred == "dfhfijzr" and len(parts) == 2:
                self.df hfijzr.add(parts[1])
            elif pred == "r1frsf5x" and len(parts) == 4:
                self.r1frsf5x.add((parts[1], parts[2], parts[3]))

        # Store the task goals as a fast lookup frozenset.
        self.goals = task.goals

    def __call__(self, node):
        state = node.state

        # Fast membership test of predicates in current state.
        state_facts = state

        def fact_in_state(predicate, *args):
            """Check if a fact of form (predicate args...) is in state."""
            fact = f"({predicate} {' '.join(args)})"
            return fact in state_facts

        # Check goals: (ixsb7u9c r6gw2 pmq6p)
        # Estimate cost by counting how many goals are missing, plus prerequisites per missing goal.

        cost = 0

        for (r6gw2, pmq6p) in self.goal_ixsb7u9c_pairs:
            goal_fact = f"(ixsb7u9c {r6gw2} {pmq6p})"
            if goal_fact in state_facts:
                # Goal satisfied, no cost.
                continue

            # Base cost: need at least 1 action to achieve this goal fact's predicate
            subcost = 1

            # Check prerequisite facts involving r6gw2 and pmq6p:
            # Must have (r1frsf5x rb7qr1fs r6gw2 pmq6p) for some rb7qr1fs to allow achieving ixsb7u9c

            # Extract all rb7qr1fs objects in state that satisfy r1frsf5x for these args
            supporting_rbs = set()
            for (rb, dbk5, chx) in self.r1frsf5x:
                if dbk5 == r6gw2 and chx == pmq6p and fact_in_state("r1frsf5x", rb, dbk5, chx):
                    supporting_rbs.add(rb)

            # If no rb7qr1fs supports this, higher cost (need to create or repair prerequisites)
            if not supporting_rbs:
                # Add penalty for missing r1frsf5x prerequisite
                subcost += 1
            else:
                # For each supporting rb7qr1fs, check if some further prerequisites hold in state:
                # Must have (jckjsxca rb7qr1fs uwq7ubjf) and (cs7680d7 dbk5i2a5 uwq7ubjf) for some uwq7ubjf
                # The operator dlx472ba precondition and effect suggests a chain involving these predicates,
                # so if missing, add penalty.

                found_valid = False
                for rb in supporting_rbs:
                    # Find uwq7ubjf linked by jckjsxca (rb, w)
                    js_w = [parts[2] for fact in state_facts if fact.startswith("(jckjsxca ") for parts in [get_parts(fact)] if parts[1] == rb]
                    # For each such uwq7ubjf w, check if cs7680d7 includes (dbk5i2a5, w)
                    for w in js_w:
                        has_cs = any(
                            fact.startswith("(cs7680d7 ")
                            and get_parts(fact)[1] == r6gw2
                            and get_parts(fact)[2] == w
                            for fact in state_facts
                        )
                        if has_cs:
                            found_valid = True
                            break
                    if found_valid:
                        break
                if not found_valid:
                    # Missing intermediate preconditions, add penalty
                    subcost += 1

            # Check connectivity predicates required by action xvldpwsb, such as b2mwd6x4 and wx95011q between uwq7ubjf objects
            # These impose that to move jckjsxca predicate bindings, certain relations must hold statically and in state

            # Try to estimate missing transfers of jckjsxca facts by checking needed move connectivity:
            # Check if for each rb7qr1fs in supporting_rbs, the required pairs of (uwq7ubjf, uwq7ubjf) exist in b2mwd6x4 (static)
            # and in wx95011q (static or state) for relevant pairs of uwq7ubjf

            # Gather all uwq7ubjf related to rb7qr1fs via jckjsxca
            related_uwq7ubjf = set()
            for fact in state_facts:
                parts = get_parts(fact)
                if parts[0] == "jckjsxca" and parts[1] in supporting_rbs:
                    related_uwq7ubjf.add(parts[2])

            # For each pair of related_uwq7ubjf, check if connections exist in b2mwd6x4 (rb7qr1fs, uwq7ubjf, uwq7ubjf)
            # and wx95011q (uwq7ubjf, uwq7ubjf) static predicates
            # Count missing relations as penalty

            missing_connectivity = 0
            related_list = list(related_uwq7ubjf)

            for i in range(len(related_list)):
                for j in range(len(related_list)):
                    if i == j:
                        continue
                    u1 = related_list[i]
                    u2 = related_list[j]

                    # Check if (rb7qr1fs, u1, u2) in b2mwd6x4 for some rb7qr1fs in supporting_rbs
                    if not any((rb, u1, u2) in self.b2mwd6x4 for rb in supporting_rbs):
                        missing_connectivity += 1

                    # Check if (u1, u2) in wx95011q (either static or in current state)
                    if (u1, u2) not in self.wx95011q and f"(wx95011q {u1} {u2})" not in state_facts:
                        missing_connectivity += 1

            # Add a fraction of missing connectivity to penalty to not overshoot heuristic too much
            subcost += missing_connectivity * 0.5

            # Sum cost for this goal predicate instance
            cost += subcost

        # If all goals satisfied, heuristic = 0
        if cost == 0:
            return 0

        # Return ceil to ensure integer heuristic but allowing fractional additions internally
        return int(cost + 0.9999)
