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
    - Returns True if the fact matches the pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (biqhd61z) domain.

    Summary:
    This heuristic estimates the minimum number of actions still required to achieve all goal facts
    of the form (ivo59apj ?child - bciid6ad). These goal facts appear to represent completion for each
    child object. Achieving (ivo59apj child) requires enabling the corresponding "child" fact by
    completing a sequence of related subgoals involving sandwiches, trays, breads, and contents. 

    The heuristic decomposes the problem into counting the number of "children" not yet solved, and
    estimating the approximate action steps needed to complete each child's goal based on the current state.
    It is domain-dependent and closely follows the action effects and the relevant predicates.

    Assumptions:
    - Each goal is of the form (ivo59apj child), where child is a bciid6ad object.
    - To achieve (ivo59apj child), a sandwich (uguffgcw) must be prepared with certain properties,
      placed on a tray (eoraohfl), and all required predicates must be satisfied.
    - The involved predicates: (zogo0zhl sandwich), (l2thmd6e sandwich), (i9fq3vit sandwich),
      (mpvvhjpt sandwich tray), (eyker22y child table), (y0rxzzmc tray table).
    - The heuristic counts how many chidren are missing the goal fact (ivo59apj child) and evaluates
      for each missing child how close the state is to constructing its sandwich and placing it correctly.
    - Each step roughly corresponds to preparing the sandwich, combining ingredients, transferring it
      to the tray, and finally finishing the child's goal.

    Heuristic Initialization:
    - Extract the set of children that appear in any child-related static facts ((n2ij1fah child) or (dr25ifx3 child))
      to determine the universe of children.
    - Extract the set of trays known via the (y0rxzzmc tray bupp7mhz) predicate from static facts.
    - Extract static relationship of (eyker22y child table) mapping each child to its table.
    - Extract the constant bupp7mhz and store it for tray-to-table checks.
    - Extract all breads and contents from static facts (predicates like (go9lczo2 bread) and (fqungbps content))
      and track which breads satisfy (e6jz04yd) and which contents satisfy (odd6wyd1).

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify children for which (ivo59apj child) is not yet true in the current state.
       These children still require finishing.
    2. For each unfinished child, estimate the minimal steps to make (ivo59apj child) true by:
       a) Checking if there exists a sandwich (uguffgcw) that can be completed and assigned to that child.
          - Sandwiches must be (zogo0zhl), then marked (i9fq3vit), and then placed onto a tray (mpvvhjpt).
          - Sandwiches in different states reflect progress in preparing the sandwich:
            - (zogo0zhl sandwich) indicates raw sandwich,
            - (l2thmd6e sandwich) indicates intermediate progress,
            - (i9fq3vit sandwich) indicates sandwich ready to be put on tray.
       b) The bread (oxxoho4f) used for sandwich needs to satisfy (go9lczo2) and (e6jz04yd) predicates.
       c) The content (hu57dt8m) for sandwich needs to satisfy (fqungbps) and (odd6wyd1) predicates.
       d) The tray (eoraohfl) needs to be on the correct table (zwhbr5cd), verified by (y0rxzzmc tray table)
          with constant bupp7mhz.
       e) Sequence of actions starts from sandwiches being raw, adding content and bread (actions kdzjcqwj/h0xkysus),
          then placing sandwich on tray (jhzk46zy), and finally completing child's goal (nfarwtet/xtn2vmez).
    3. Approximate the heuristic cost per child by counting missing sandwich preparation steps and transfers:
       - If no sandwich in i9fq3vit state for the child: add cost for preparing it.
       - If no sandwich assigned to tray: add cost for putting sandwich on tray.
       - If final goal (ivo59apj child) missing: add cost for finishing it.
    4. Sum estimated costs over all unfinished children.
    5. Return 0 if all goals satisfied; else return the sum.
    6. Heuristic is not guaranteed admissible but aims to accurately estimate the number of actions left.

    Note: The heuristic avoids complex search and only uses current state facts and static information for efficiency.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        self.bupp7mhz = "bupp7mhz"

        # Extract all children from static facts (those that appear with n2ij1fah or dr25ifx3 predicates)
        self.children = set()
        for fact in self.static:
            if match(fact, "n2ij1fah", "*"):
                parts = get_parts(fact)
                self.children.add(parts[1])
            elif match(fact, "dr25ifx3", "*"):
                parts = get_parts(fact)
                self.children.add(parts[1])

        # Extract child's table via (eyker22y child table)
        self.child_to_table = {}
        for fact in self.static:
            if match(fact, "eyker22y", "*", "*"):
                parts = get_parts(fact)
                child = parts[1]
                table = parts[2]
                self.child_to_table[child] = table

        # All trays from static and init are of type eoraohfl with y0rxzzmc predicate to bupp7mhz
        self.all_trays = set()
        for fact in self.static:
            if match(fact, "y0rxzzmc", "*", self.bupp7mhz):
                parts = get_parts(fact)
                self.all_trays.add(parts[1])

        # All breads from static facts that have (go9lczo2 bread)
        self.all_breads = set()
        for fact in self.static:
            if match(fact, "go9lczo2", "*"):
                self.all_breads.add(get_parts(fact)[1])

        # All contents from static facts that have (fqungbps content)
        self.all_contents = set()
        for fact in self.static:
            if match(fact, "fqungbps", "*"):
                self.all_contents.add(get_parts(fact)[1])

        # Extract breads that satisfy (e6jz04yd)
        self.e6jz04yd_breads = set()
        for fact in self.static:
            if match(fact, "e6jz04yd", "*"):
                self.e6jz04yd_breads.add(get_parts(fact)[1])

        # Extract contents that satisfy (odd6wyd1)
        self.odd6wyd1_contents = set()
        for fact in self.static:
            if match(fact, "odd6wyd1", "*"):
                self.odd6wyd1_contents.add(get_parts(fact)[1])

    def __call__(self, node):
        state = node.state

        # Quick check: if goal reached, heuristic is zero
        if self.goals <= state:
            return 0

        # --- Gather current state facts of interest ---
        # Track which children have their goal (ivo59apj child) true
        child_goal_satisfied = set()
        for fact in state:
            if match(fact, "ivo59apj", "*"):
                child_goal_satisfied.add(get_parts(fact)[1])

        # Children still to be completed
        children_to_do = self.children - child_goal_satisfied

        # Track sandwiches (uguffgcw) and their predicates states:
        # We want to know:
        # - which sandwiches are (zogo0zhl) raw
        # - which sandwiches are (l2thmd6e) intermediate
        # - which sandwiches are (i9fq3vit) ready to be placed on tray

        sandwiches_zogo0zhl = set()
        sandwiches_l2thmd6e = set()
        sandwiches_i9fq3vit = set()
        sandwiches_mpvvhjpt = set()  # sandwiches placed on tray

        # Map sandwich to tray (for mpvvhjpt)
        sandwich_to_tray = {}

        for fact in state:
            if match(fact, "zogo0zhl", "*"):
                sandwiches_zogo0zhl.add(get_parts(fact)[1])
            if match(fact, "l2thmd6e", "*"):
                sandwiches_l2thmd6e.add(get_parts(fact)[1])
            if match(fact, "i9fq3vit", "*"):
                sandwiches_i9fq3vit.add(get_parts(fact)[1])
            if match(fact, "mpvvhjpt", "*", "*"):
                parts = get_parts(fact)
                sandwich, tray = parts[1], parts[2]
                sandwiches_mpvvhjpt.add(sandwich)
                sandwich_to_tray[sandwich] = tray

        # Track trays and their tables currently (y0rxzzmc tray table)
        tray_to_table = {}
        for fact in state:
            if match(fact, "y0rxzzmc", "*", "*"):
                parts = get_parts(fact)
                tray, table = parts[1], parts[2]
                tray_to_table[tray] = table

        # Track breads currently with (go9lczo2 bread) - these seem to be "available" breads
        breads_go9lczo2 = set()
        for fact in state:
            if match(fact, "go9lczo2", "*"):
                breads_go9lczo2.add(get_parts(fact)[1])

        # Track contents currently with (fqungbps content) - these are available contents
        contents_fqungbps = set()
        for fact in state:
            if match(fact, "fqungbps", "*"):
                contents_fqungbps.add(get_parts(fact)[1])

        # Track breads with (e6jz04yd bread)
        breads_e6jz04yd = set()
        for fact in state:
            if match(fact, "e6jz04yd", "*"):
                breads_e6jz04yd.add(get_parts(fact)[1])

        # Track contents with (odd6wyd1 content)
        contents_odd6wyd1 = set()
        for fact in state:
            if match(fact, "odd6wyd1", "*"):
                contents_odd6wyd1.add(get_parts(fact)[1])

        # Heuristic cost accumulator
        cost = 0

        # Estimated costs for partial steps:
        # Each missing step roughly counts as 1 action or group of actions:
        #  - Preparing sandwich (kdzjcqwj or h0xkysus) correspond to combining bread and content,
        #    turning (zogo0zhl) sandwiches into (i9fq3vit) and (l2thmd6e).
        #  - Putting sandwich on tray (jhzk46zy).
        #  - Finalizing child's goal (nfarwtet or xtn2vmez).

        # Cost constants as rough estimations:
        PREPARE_SANDWICH_COST = 1     # kdzjcqwj or h0xkysus action(s)
        PLACE_ON_TRAY_COST = 1        # jhzk46zy action
        FINALIZE_FOR_CHILD_COST = 1   # nfarwtet or xtn2vmez action

        # For each unfinished child, estimate cost:
        for child in children_to_do:

            # Step 1: Check if a sandwich has been prepared for this child
            # This means a sandwich in state i9fq3vit (ready) or mpvvhjpt (placed on tray) for this child's tray/table

            # Find trays at child's table
            child_table = self.child_to_table.get(child)
            trays_at_child_table = {tray for tray, table in tray_to_table.items() if table == child_table}

            # Track if there is a sandwich on tray at child's table:
            sandwich_on_tray = False
            for sandwich in sandwiches_mpvvhjpt:
                tray = sandwich_to_tray.get(sandwich)
                if tray in trays_at_child_table:
                    # Also check if the sandwich belongs to "active" set (i.e. has l2thmd6e or not zogo0zhl)
                    # but we cannot identify sandwich-child relation directly, so we approximation here
                    sandwich_on_tray = True
                    break

            # Step 2: Check if there is a sandwich ready (i9fq3vit) but not yet placed on tray at child's table
            sandwich_ready_not_on_tray = False
            if not sandwich_on_tray:
                # Check if any sandwich in i9fq3vit state that is not in mpvvhjpt can correspond to child
                # Since we cannot uniquely assign sandwich to child, we assume that presence of any ready sandwich that
                # can be put on child's tray reduces cost.
                if len(sandwiches_i9fq3vit - sandwiches_mpvvhjpt) > 0:
                    sandwich_ready_not_on_tray = True

            # Step 3: Check if no sandwich ready at all for child
            sandwich_prepared = sandwich_on_tray or sandwich_ready_not_on_tray

            # Step 4: Check presence of raw sandwiches (zogo0zhl)
            has_raw_sandwich = len(sandwiches_zogo0zhl) > 0

            # Estimate cost per child:

            # If final goal missing (ivo59apj child), at least 1 action needed
            cost_per_child = FINALIZE_FOR_CHILD_COST

            if sandwich_on_tray:
                # Sandwich is ready and placed on tray, only finalize needed
                pass  # cost remains 1
            elif sandwich_ready_not_on_tray:
                # Need to place sandwich on tray plus finalizing
                cost_per_child += PLACE_ON_TRAY_COST
            elif has_raw_sandwich:
                # Need to prepare sandwich (kdzjcqwj or h0xkysus), place on tray, finalize
                cost_per_child += PREPARE_SANDWICH_COST + PLACE_ON_TRAY_COST
            else:
                # No sandwich at all for child, assume all three steps needed and possibly pick bread/content
                # This is a rough upper bound:
                cost_per_child += PREPARE_SANDWICH_COST + PLACE_ON_TRAY_COST

            cost += cost_per_child

        return cost
