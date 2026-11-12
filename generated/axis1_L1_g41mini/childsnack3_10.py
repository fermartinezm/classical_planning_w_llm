from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate obj1 obj2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain domain.

    Summary
    -------
    This heuristic estimates the number of actions required to achieve the goal
    conditions from a given state, guiding a greedy best-first search. The goal 
    is to have one or more `ivo59apj` predicates true for certain `bciid6ad` objects 
    (called children in examples). These predicates represent the final achievement 
    for each such object.

    Assumptions
    -----------
    - Each `bciid6ad` object must achieve the goal predicate `(ivo59apj <child>)`.
    - To achieve `(ivo59apj ?db93w)`, certain predicates involving `mpvvhjpt`, `eyker22y`, 
      `l2thmd6e`, and `y0rxzzmc` must hold.
    - The domain uses a sequence of four main stages per object to reach the goal:
        1) Prepare a `uguffgcw` object (like a sandwich) such that `(i9fq3vit ?yam4s)` holds 
           by applying `kdzjcqwj` or `h0xkysus` actions if needed (depending on predicates on 
           `bread` and `content` objects).
        2) Use `jhzk46zy` action to consume the `i9fq3vit` predicate to establish `(mpvvhjpt ?yam4s ?zw4bi)`.
        3) Apply either `nfarwtet` or `xtn2vmez` to achieve `(ivo59apj ?db93w)`.
        4) The predicate `y0rxzzmc` links `eoraohfl` objects with `zwhbr5cd` objects (the constant 
           `bupp7mhz`), which are critical in final steps.

    Heuristic Initialization
    ------------------------
    - Extract all static facts from the task, in particular:
        * The mapping `(eyker22y ?db93w ?upvbj)` associates each `bciid6ad` object with a 
          unique `zwhbr5cd` location.
        * The constant `bupp7mhz` of type `zwhbr5cd` is used in predicates `(y0rxzzmc ?zw4bi bupp7mhz)`.
    - Store the set of `bciid6ad` objects that appear in the goal predicate `(ivo59apj ?db93w)`.
    - Extract static objects to recognize these domain entities.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each `bciid6ad` object `child` that requires `(ivo59apj child)` in goal:
    
    1. Check if the goal is already satisfied, i.e. `(ivo59apj child)` is in the current state.
       - If yes, no cost for this child.
       - Otherwise, estimate cost to achieve it.
    
    2. To achieve `(ivo59apj child)`, we need to have `(mpvvhjpt ?yam4s ?zw4bi)` for some `uguffgcw` and 
       `eoraohfl` objects, plus `(eyker22y child ?upvbj)` and `(y0rxzzmc ?zw4bi ?upvbj)` facts.
       
       - The predicate `(eyker22y child ?upvbj)` is static and known.
       - From `(y0rxzzmc ?zw4bi bupp7mhz)`, we know which trays are linked to the constant `bupp7mhz`.
       
    3. Check whether any `(mpvvhjpt ?yam4s ?zw4bi)` exists in the state for this `child`'s `eyker22y` location.
       - If yes, the cost to get `(ivo59apj child)` is one action (either `nfarwtet` or `xtn2vmez`).
    
    4. If no such `(mpvvhjpt ?yam4s ?zw4bi)` yet:
       - Check if we have `(i9fq3vit ?yam4s)` for some `uguffgcw`.
         * If yes, then one action `jhzk46zy` is needed to establish `(mpvvhjpt ?yam4s ?zw4bi)`.
       - If no `(i9fq3vit ?yam4s)` exists, then we need to apply `kdzjcqwj` or `h0xkysus` first.
         * These two actions differ in that `kdzjcqwj` requires additional predicates about `content` and `bread`.
       
       - Since the predicate `zogo0zhl ?yam4s` indicates sandwiches not yet processed,
         assume that each `uguffgcw` with `zogo0zhl` corresponds to one needed preparation action.
       
    5. Thus, the estimated cost per child is the sum of:
       - 0 if `(ivo59apj child)` in state.
       - Otherwise, cost to produce `(mpvvhjpt ?yam4s ?zw4bi)` + 1 for final step action producing `(ivo59apj child)`.
    
    6. For multiple children, sum individual costs.
    
    7. Because some sandwiches may be shared or reusable, the heuristic errs on the side of summing independent costs but is still efficient.

    Note:
    - This heuristic is not admissible; it aims for efficiency and guidance.
    - The heuristic returns 0 exactly for goal states.
    - It is efficiently computable by simple scans over facts without deep search.
    """

    def __init__(self, task):
        """Extract static info and initialize the heuristic."""
        self.goals = task.goals
        self.static = task.static

        # Extract children for which (ivo59apj child) is a goal
        self.goal_children = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "ivo59apj" and len(parts) == 2:
                self.goal_children.add(parts[1])

        # Map each child (bciid6ad) to its respective zwhbr5cd location from static facts (eyker22y)
        self.child_to_location = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "eyker22y" and len(parts) == 3:
                child, location = parts[1], parts[2]
                self.child_to_location[child] = location

        # Identify which trays (eoraohfl) are linked to bupp7mhz (a constant of type zwhbr5cd)
        self.bupp7mhz_constant = "bupp7mhz"
        self.tray_to_upvbj = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "y0rxzzmc" and len(parts) == 3 and parts[2] == self.bupp7mhz_constant:
                tray = parts[1]
                self.tray_to_upvbj[tray] = self.bupp7mhz_constant
        # We do not rely on tray_to_upvbj directly, but in state will check presence of y0rxzzmc facts.

    def __call__(self, node):
        state = node.state

        # Fast goal check: if goal already reached, heuristic = 0
        if self.goals <= state:
            return 0

        cost = 0

        # Index some helpful predicates for efficiency by creating sets:
        # (ivo59apj ?child)
        ivo59apj_facts = set(
            get_parts(f)[1] for f in state if match(f, "ivo59apj", "*")
        )
        # All (mpvvhjpt ?yam4s ?zw4bi)
        mpvvhjpt_pairs = set(
            (get_parts(f)[1], get_parts(f)[2]) for f in state if match(f, "mpvvhjpt", "*", "*")
        )
        # All (i9fq3vit ?yam4s)
        i9fq3vit_set = set(get_parts(f)[1] for f in state if match(f, "i9fq3vit", "*"))
        # All (y0rxzzmc ?zw4bi bupp7mhz)
        y0rxzzmc_bupp7mhz = set(
            get_parts(f)[1]
            for f in state
            if match(f, "y0rxzzmc", "*", self.bupp7mhz_constant)
        )
        # Sandwiches (uguffgcw) that are unprocessed: (zogo0zhl ?yam4s)
        zogo0zhl_set = set(get_parts(f)[1] for f in state if match(f, "zogo0zhl", "*"))
        # All children currently have goal? or not is checked by ivo59apj_facts.

        # For each child that needs to achieve (ivo59apj child):
        for child in self.goal_children:
            if child in ivo59apj_facts:
                # Goal already achieved, no cost here
                continue

            # child location from static facts, needed for matching
            child_loc = self.child_to_location.get(child, None)
            if child_loc is None:
                # Missing static info: assume at least 1 step cost (can't be zero)
                cost += 1
                continue

            # We want to check if mpvvhjpt is established for some sandwich and tray
            # Find if there exists any (mpvvhjpt ?yam4s ?zw4bi) where zw4bi is in y0rxzzmc linked to bupp7mhz 
            # and y0rxzzmc matches (zw4bi bupp7mhz) in state, with upvbj=constant from eyker22y(child, upvbj)
            # so make sure (y0rxzzmc ?zw4bi ?upvbj) in state, but here upvbj is constant "bupp7mhz" in y0rxzzmc facts

            # Because y0rxzzmc predicate links tray ?zw4bi to constant bupp7mhz, and eyker22y(child, upvbj) links child to table ?upvbj,
            # but upvbj for child is not necessarily bupp7mhz constant, so check if child's upvbj == bupp7mhz? 
            # From static facts eyker22y always link child -> some table (upvbj), but y0rxzzmc links trays -> bupp7mhz constant,
            # so it's not guaranteed child's upvbj == bupp7mhz, might differ.
            # The action nfarwtet and xtn2vmez check that eyker22y(child, upvbj) and y0rxzzmc(tray, upvbj) both hold same upvbj,
            # so upvbj in y0rxzzmc(tray, upvbj) must match child's upvbj to enable these actions.
            # Since the only bupp7mhz constant given, and y0rxzzmc facts in state only link trays to bupp7mhz, 
            # we assume upvbj for all children is bupp7mhz (the examples show this).
            # So assume child's upvbj == bupp7mhz constant.

            # Check if child's upvbj equals constant bupp7mhz
            # we need child's upvbj for that:
            child_upvbj = None
            for fact in self.static:
                parts = get_parts(fact)
                if parts[0] == "eyker22y" and parts[1] == child:
                    child_upvbj = parts[2]
                    break
            if child_upvbj is None:
                # Missing info, assume nonzero cost
                cost += 1
                continue

            # Check if child's upvbj equals bupp7mhz:
            upvbj_matches_bupp7mhz = child_upvbj == self.bupp7mhz_constant
            if not upvbj_matches_bupp7mhz:
                # No path to goal by given actions (since y0rxzzmc links trays only to bupp7mhz)
                # We can assume high cost, or 1 minimum cost.
                cost += 1
                continue

            # Find if there exists an (mpvvhjpt sandwich tray) where tray is a tray that satisfies
            # (y0rxzzmc tray bupp7mhz) -- i.e. tray in y0rxzzmc_bupp7mhz
            mpvvhjpt_for_child = False
            for (sandwich, tray) in mpvvhjpt_pairs:
                if tray in y0rxzzmc_bupp7mhz:
                    mpvvhjpt_for_child = True
                    break

            if mpvvhjpt_for_child:
                # Only one action needed: nfarwtet or xtn2vmez produces (ivo59apj child)
                cost += 1
                continue

            # If mpvvhjpt not achieved, check if any sandwich is ready (i9fq3vit sandwich) to generate mpvvhjpt
            if len(i9fq3vit_set) > 0:
                # One action jhzk46zy needed + one final action nfarwtet or xtn2vmez
                cost += 2
                continue

            # Else, must prepare sandwich first (kdzjcqwj or h0xkysus)
            # Count how many sandwiches unprocessed: zogo0zhl_set size
            # We'll assume at least one is required to prepare for this child.
            if len(zogo0zhl_set) > 0:
                # One action to prepare sandwich + one action to produce mpvvhjpt + one to get ivo59apj
                cost += 3
            else:
                # No sandwich available or ready, assume 2 actions minimal
                cost += 2

        return cost
