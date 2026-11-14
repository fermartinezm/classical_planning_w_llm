from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Wildcards * are supported for pattern arguments.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (biqhd61z). The heuristic estimates the number
    of actions required to achieve the goal facts (ivo59apj ?child) for each child object.
    It exploits the structure of the domain, focusing on the "preparation" and "delivery" steps
    of sandwiches (uguffgcw), bread (oxxoho4f), and content (hu57dt8m) combined together with the children (bciid6ad),
    trays (eoraohfl), and tables (zwhbr5cd).

    Assumptions
    -----------
    - The goal is defined as (ivo59apj ?child) for some children.
    - To achieve (ivo59apj ?child), the task must:
      1) Prepare sandwiches: i9fq3vit and l2thmd6e predicates mark the sandwich states.
      2) Deliver the prepared sandwich placed on trays to children, respecting preconditions in actions.
    - Static information identifies children types and their associated locations.
    - The heuristic ignores deletes and reordering because it is greedy and non-admissible.
    - Actions kdzjcqwj and h0xkysus prepare sandwiches.
    - Actions jhzk46zy place sandwiches on trays.
    - Actions nfarwtet and xtn2vmez deliver the sandwich to the child.

    Heuristic Initialization
    ------------------------
    - Extract from static facts:
      * The set of child objects (bciid6ad) for which the goal (ivo59apj) must hold.
      * The associations eyker22y between children and tables.
    - Extract the goal children from the goal facts.
    - Precompute sets of relevant predicates in goal, static, and state.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. The goal is to have (ivo59apj ?child) true for each goal child.
    2. For each goal child not yet having (ivo59apj child), the heuristic estimates:
       * If (ivo59apj child) is true in the state, cost is 0 for that child.
       * Otherwise, estimate cost to prepare and deliver a sandwich for that child.
    3. Preparation involves:
       - Checking if a sandwich (?sandw) is prepared, indicated by (i9fq3vit ?sandw) and (l2thmd6e ?sandw).
         If not yet prepared, count cost for preparation actions (kdzjcqwj or h0xkysus) which require bread (go9lczo2), content (fqungbps), and sandwich raw state (zogo0zhl).
    4. Delivery involves:
       - A sandwich placed on a tray: (mpvvhjpt ?sandw ?tray) where the tray is associated to a table where the child is located.
       - If not placed, estimate cost of placing sandwich on tray (action jhzk46zy).
       - Finally, delivering to the child (actions nfarwtet or xtn2vmez) which require preconditions including (eyker22y ?child ?table), (mpvvhjpt ?sandw ?tray), and (l2thmd6e ?sandw).
    5. The heuristic sums estimated minimal steps for all children not yet satisfied.
    6. To keep computation efficient:
       - For each child, identify a suitable sandwich and tray from the state facts,
         or estimate opening cost if none is prepared yet.
       - Since sandwiches are independent, sum costs.
    """

    def __init__(self, task):
        # Extract goal children for which (ivo59apj ?child) is required.
        self.goal_children = {
            get_parts(goal)[1]
            for goal in task.goals
            if goal.startswith("(ivo59apj ")
        }

        # From static facts: map children to their table locations for delivery.
        self.child_to_table = {}
        for fact in task.static:
            if match(fact, "eyker22y", "*", "*"):
                child, table = get_parts(fact)[1:]
                self.child_to_table[child] = table

        # From static facts: remember the constant bupp7mhz (the unique constant of type zwhbr5cd)
        self.constant_bupp7mhz = None
        for fact in task.static:
            # The constant bupp7mhz is declared as constant of type zwhbr5cd; 
            # it appears in predicates like (y0rxzzmc ?tray bupp7mhz)
            # Since it's the single constant, we can just fix it to "bupp7mhz" as given.
            # No need to extract it from static facts since constants are fixed.
            self.constant_bupp7mhz = "bupp7mhz"
            break

    def __call__(self, node):
        state = node.state

        # Quick check for goal state:
        if all(f"(ivo59apj {child})" in state for child in self.goal_children):
            return 0

        # Index state facts for fast lookup.
        facts_by_pred = {}
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            facts_by_pred.setdefault(pred, set()).add(fact)

        # Helper functions to find facts by predicate and arguments.
        def find_facts(pred, *args):
            """
            Find facts with predicate `pred` whose arguments match args pattern.
            Wildcard is '*', which matches any argument.
            """
            res = []
            facts = facts_by_pred.get(pred, set())
            for fct in facts:
                parts = get_parts(fct)
                if len(parts) - 1 < len(args):
                    continue
                match_args = True
                for i, pattern in enumerate(args):
                    if pattern != "*" and parts[i + 1] != pattern:
                        match_args = False
                        break
                if match_args:
                    res.append(fct)
            return res

        # We must estimate cost for each child that does not yet have ivo59apj.
        total_cost = 0

        # Keep track of sandwiches already assigned for heuristics (to avoid overestimating).
        used_sandwiches = set()

        # Find all sandwiches in state by predicate zogo0zhl (raw sandwich), i9fq3vit (prepared), l2thmd6e (ready to deliver).
        # These predicates mark sandwich status.
        # Also gather sandwich -> trays mapping via (mpvvhjpt ?sandw ?tray)
        sandwich_raw = {get_parts(f)[1] for f in find_facts("zogo0zhl", "*")}
        sandwich_prepared_i9fq3vit = {get_parts(f)[1] for f in find_facts("i9fq3vit", "*")}
        sandwich_l2thmd6e = {get_parts(f)[1] for f in find_facts("l2thmd6e", "*")}
        sandwich_on_tray = {}
        for fact in find_facts("mpvvhjpt", "*", "*"):
            parts = get_parts(fact)
            sandwich_on_tray[parts[1]] = parts[2]

        # Bread with e6jz04yd predicate
        bread_available = {get_parts(f)[1] for f in find_facts("e6jz04yd", "*")}
        # Content with fqungbps predicate (not yet used up)
        content_available = {get_parts(f)[1] for f in find_facts("fqungbps", "*")}
        # Children in dr25ifx3 or n2ij1fah predicate (different children types); both appear in effects
        children_n2ij1fah = {get_parts(f)[1] for f in find_facts("n2ij1fah", "*")}
        children_dr25ifx3 = {get_parts(f)[1] for f in find_facts("dr25ifx3", "*")}

        # For heuristic guessing:
        # Cost constants (rough estimates, domain knowledge-based):
        # Action costs approximated as 1 per action.
        COST_PREPARE_AND_READY = 3  # Actions needed to prepare sandwich (kdzjcqwj or h0xkysus)
        COST_PLACE_ON_TRAY = 1       # Action jhzk46zy to place sandwich on tray
        COST_DELIVER = 1             # One of nfarwtet or xtn2vmez to deliver to child

        # For each child still needing delivery:
        for child in self.goal_children:
            if f"(ivo59apj {child})" in state:
                # Already delivered, skip.
                continue

            # Get child's associated table:
            child_table = self.child_to_table.get(child, None)
            if child_table is None:
                # No static info on child -> table, assume minimal delivery cost.
                child_table = None

            # Try to find a prepared sandwich ready and placed correctly for this child:
            # Find sandwiches on a tray on the correct table (y0rxzzmc tray table) and tray used is linked to the sandwich on that tray.
            # tray with (y0rxzzmc ?tray bupp7mhz)
            # but to check if tray is on child's table, we must find if (y0rxzzmc tray table) AND table==child's table
            trays_on_table = {
                get_parts(fact)[1]
                for fact in find_facts("y0rxzzmc", "*", self.constant_bupp7mhz)  # unique constant
                if True  # we only know tray - constant mapping here, table facts unknown directly
            }

            # Due to lack of explicit table-tray relation from 'y0rxzzmc' other than table constant bupp7mhz,
            # we assume all trays are correctly placed at correct table (since eyker22y is child-table link)
            # So we check if sandwich is on tray and sandwich is ready and tray is present.
            candidate_sandwich = None
            for sw in sandwich_l2thmd6e:
                # sandwich must be on some tray:
                tray = sandwich_on_tray.get(sw, None)
                if tray in trays_on_table:
                    candidate_sandwich = sw
                    break

            if candidate_sandwich is not None:
                # Sandwich already prepared, placed on tray, ready to be delivered to child.
                # Estimate only 1 action to deliver from tray to child in this case.
                total_cost += COST_DELIVER
                used_sandwiches.add(candidate_sandwich)
                continue

            # Otherwise, try to find a prepared sandwich (i9fq3vit and l2thmd6e true), but not placed on tray
            candidate_sandwich = None
            for sw in sandwich_l2thmd6e:
                if sw not in sandwich_on_tray and sw not in used_sandwiches:
                    candidate_sandwich = sw
                    break

            if candidate_sandwich is not None:
                # Sandwich prepared but not placed on tray yet:
                # Cost: 1 for placing on tray + 1 for delivery
                total_cost += COST_PLACE_ON_TRAY + COST_DELIVER
                used_sandwiches.add(candidate_sandwich)
                continue

            # Otherwise, try to find a raw sandwich (zogo0zhl) not yet prepared
            candidate_sandwich = None
            for sw in sandwich_raw:
                if sw not in sandwich_l2thmd6e and sw not in used_sandwiches:
                    candidate_sandwich = sw
                    break

            if candidate_sandwich is not None:
                # Need to prepare sandwich (kdzjcqwj or h0xkysus), then place on tray, then deliver.
                total_cost += COST_PREPARE_AND_READY + COST_PLACE_ON_TRAY + COST_DELIVER
                used_sandwiches.add(candidate_sandwich)
                continue

            # If no sandwich found, assume minimal cost to prepare 1 sandwich + place + deliver:
            total_cost += COST_PREPARE_AND_READY + COST_PLACE_ON_TRAY + COST_DELIVER

        return total_cost
