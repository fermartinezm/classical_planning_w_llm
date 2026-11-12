from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return components of a PDDL fact string '(pred a b ...)' as a list."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Return True if `fact` matches the pattern (`*` supported) where
    pattern elements correspond to predicate and arguments.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (biqhd61z) problem.
    The domain models preparing sandwiches and delivering them to children:
      - Sandwiches can be assembled (i9fq3vit), placed on trays (mpvvhjpt), and
        then delivered to children, producing the goal fact (ivo59apj child).
    This heuristic estimates the number of actions still required to satisfy
    all goal child-serving facts (ivo59apj ...). It is intentionally
    non-admissible and aims to be informative for greedy best-first search.

    Assumptions
    -----------
    - Each (unsatisfied) child requires a single sandwich to reach its goal.
    - Serving a child normally requires three kinds of actions:
        1) producing an assembled sandwich (kdzjcqwj or h0xkysus) or using an
           already assembled sandwich (i9fq3vit),
        2) placing the sandwich onto a tray (jhzk46zy),
        3) executing the serve action (nfarwtet or xtn2vmez).
      We therefore treat a baseline of ~3 actions per child.
    - There may be additional tray-movement actions (whgyhnl0) if a tray is not
      currently at the child's table. We add a small penalty for moving trays.
    - Some children require a special sandwich property (l2thmd6e). The heuristic
      attempts to prefer already-assembled sandwiches with that property or
      to count the extra resource requirement when creating them.
    - The heuristic is optimistic about action concurrency but pessimistic about
      missing domain resources by applying small penalties if resources are
      absent in the current state. This keeps estimates finite and often useful.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic extracts static information from the task:
    - self.goals: the set of goal facts (strings).
    - child_to_table: mapping child -> table extracted from static eyker22y facts.
    - children_n2: set of children that have the n2ij1fah predicate (these
      children require the sandwich to have the l2thmd6e mark).
    - children_dr: set of children that have the dr25ifx3 predicate (they can
      be served using xtn2vmez; tracked for completeness).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a node (state), the heuristic does the following:

    1. Quick goal check:
       - If all goal facts are already true in the state, return 0.

    2. Parse dynamic facts in the state to get:
       - assembled_sandwiches: set of sandwiches with i9fq3vit.
       - l2_sandwiches: subset of assembled sandwiches that already have
         l2thmd6e.
       - raw_sandwiches: set of sandwich objects with zogo0zhl (raw material).
       - breads and breads_with_e6: sets of bread objects (go9lczo2) and those
         additionally annotated with e6jz04yd (favorable for producing l2).
       - contents and contents_with_odd: content objects (fqungbps) and those
         annotated with odd6wyd1 (favorable for producing l2).
       - placed list: list of (sandwich, tray) tuples from mpvvhjpt facts.
       - tray_locations: mapping tray -> table from y0rxzzmc facts.
       - trays_at_bupp: boolean indicating whether any tray currently sits at
         the special constant bupp7mhz (jhzk46zy requires a tray at bupp7mhz
         for placing; the heuristic factors this in with a small penalty).
       - already_served: any goal facts already true are ignored (they were
         handled by the quick goal check).

    3. Greedy resource assignment to unsatisfied goals:
       - For each unsatisfied child (we process children in arbitrary order)
         pick the cheapest way to obtain a valid delivered sandwich:
         a) If there exists a placed sandwich on a tray whose tray is already
            at the child's table and the sandwich meets the l2 requirement (if
            the child needs l2), then cost = 1 (just the serve action).
         b) Else if there exists an assembled sandwich that meets the l2
            requirement (if needed), estimate cost =
               place_cost (1) + serve_cost (1) plus a tray-movement penalty:
               - If a tray is currently at the child's table -> no movement cost.
               - Else add 1 for moving a tray to the child's table.
               - Additionally add 1 as penalty if no tray sits at bupp7mhz (the
                 place action requires a tray at bupp7mhz; if none present we
                 optimistically assume one extra whgyhnl0 will be needed).
         c) Else attempt to create a sandwich from raw resources:
               - Base create cost = 1.
               - If the child requires l2, prefer using bread with e6jz04yd and
                 content with odd6wyd1. If those special resources are missing,
                 add +1 penalty per missing special resource (optimistic but
                 finite).
               - If any of bread/content/raw material is missing at all, add a
                 small penalty (+1 per missing type) (optimistic assumption:
                 resources can be obtained later, but we reflect a small cost).
               - After creation, add same placement and serving costs as in (b).
       - When a resource (placed sandwich, assembled sandwich, raw sandwich,
         bread, content) is assigned to a child, it is consumed in our
         internal accounting so later children cannot reuse it.

    4. Sum the per-child costs to return a single numeric heuristic value.

    Rationale
    ---------
    - The heuristic is domain-dependent and tries to reflect the main task
      structure (assemble -> place -> serve) and the special l2 requirement.
    - It uses simple greedy matching of existing helpful facts first (placed
      sandwiches, then assembled ones, then raw resources). This reduces the
      estimate when the state already made progress.
    - The penalties for missing resources, missing trays at child tables, or
      missing trays at bupp7mhz are small integers; this keeps the heuristic
      finite and informative while remaining fast to compute.
    """

    def __init__(self, task):
        """Extract static information and goals from the task."""
        self.goals = set(task.goals)
        self.static = set(task.static)

        # Map child -> table (from eyker22y static facts)
        self.child_to_table = {}
        for fact in self.static:
            if match(fact, "eyker22y", "*", "*"):
                _, child, table = get_parts(fact)
                self.child_to_table[child] = table

        # Children that require l2thmd6e (n2ij1fah)
        self.children_n2 = {get_parts(f)[1] for f in self.static if match(f, "n2ij1fah", "*")}

        # Children that use the other serve action (dr25ifx3) - tracked for completeness
        self.children_dr = {get_parts(f)[1] for f in self.static if match(f, "dr25ifx3", "*")}

    def __call__(self, node):
        """Compute the domain-dependent heuristic estimate for the given state."""
        state = set(node.state)

        # 1) If goals already satisfied, return 0.
        if self.goals <= state:
            return 0

        # 2) Collect dynamic resources and relevant facts from the state.
        # Assembled sandwiches (i9fq3vit)
        assembled = {get_parts(f)[1] for f in state if match(f, "i9fq3vit", "*")}
        # Sandwiches that have l2thmd6e
        l2_sand = {get_parts(f)[1] for f in state if match(f, "l2thmd6e", "*")}
        # Raw sandwiches
        raw_sand = {get_parts(f)[1] for f in state if match(f, "zogo0zhl", "*")}
        # Breads (go9lczo2) and those with e6jz04yd
        breads = {get_parts(f)[1] for f in state if match(f, "go9lczo2", "*")}
        breads_with_e6 = {get_parts(f)[1] for f in state if match(f, "e6jz04yd", "*")}
        # Contents (fqungbps) and those with odd6wyd1
        contents = {get_parts(f)[1] for f in state if match(f, "fqungbps", "*")}
        contents_with_odd = {get_parts(f)[1] for f in state if match(f, "odd6wyd1", "*")}

        # placed sandwiches on trays: mpvvhjpt sandwich tray
        placed = []
        for f in state:
            if match(f, "mpvvhjpt", "*", "*"):
                _, sandw, tray = get_parts(f)
                placed.append((sandw, tray))

        # tray -> table mapping from y0rxzzmc facts
        tray_locations = {}
        for f in state:
            if match(f, "y0rxzzmc", "*", "*"):
                _, tray, table = get_parts(f)
                tray_locations[tray] = table

        # Are there trays at the special bupp7mhz constant?
        trays_at_bupp = any(table == "bupp7mhz" for table in tray_locations.values())

        # Goals: which children still need to be served?
        unsatisfied_children = []
        for g in self.goals:
            if g not in state:
                # expected goal format "(ivo59apj childX)"
                parts = get_parts(g)
                if parts[0] == "ivo59apj" and len(parts) >= 2:
                    unsatisfied_children.append(parts[1])
                else:
                    # generic fallback: if it's not an ivo59apj goal, treat it as a single subgoal
                    # but our domain examples use ivo59apj for goals.
                    if len(parts) >= 2:
                        unsatisfied_children.append(parts[1])

        # Greedy resource pools (we will consume items as we assign them).
        assembled_pool = set(assembled)
        l2_pool = set(l2_sand)
        raw_pool = set(raw_sand)
        bread_pool = set(breads)
        bread_e6_pool = set(breads_with_e6)
        content_pool = set(contents)
        content_odd_pool = set(contents_with_odd)
        placed_pool = list(placed)  # list of (sandw, tray)

        total_cost = 0

        # For each child, choose cheapest option and consume resources.
        for child in unsatisfied_children:
            needs_l2 = child in self.children_n2
            child_table = self.child_to_table.get(child)

            # 1) Check if a placed sandwich is already at the child's table and satisfies l2 if needed.
            chosen = False
            for idx, (sandw, tray) in enumerate(placed_pool):
                tray_table = tray_locations.get(tray)
                if tray_table is None:
                    continue
                if child_table is not None and tray_table == child_table:
                    if not needs_l2 or sandw in l2_pool:
                        # Serve action will finish this child.
                        total_cost += 1  # nfarwtet or xtn2vmez
                        # consume placed sandwich
                        placed_pool.pop(idx)
                        chosen = True
                        break
            if chosen:
                continue

            # 2) Try to use an assembled sandwich
            assembled_choice = None
            for sandw in assembled_pool:
                if not needs_l2 or sandw in l2_pool:
                    assembled_choice = sandw
                    break
            if assembled_choice is not None:
                # place + possibly move + serve
                place_cost = 1  # jhzk46zy
                serve_cost = 1  # nfarwtet/xtn2vmez
                # If tray already at child's table, no move; else assume 1 move needed.
                move_cost = 0 if (child_table and any(tloc == child_table for tloc in tray_locations.values())) else 1
                # If no tray at bupp7mhz exists, add a small penalty (we will need to make a tray available at bupp7mhz)
                bupp_penalty = 0 if trays_at_bupp else 1
                total_cost += place_cost + serve_cost + move_cost + bupp_penalty
                # consume assembled sandwich
                assembled_pool.remove(assembled_choice)
                if assembled_choice in l2_pool:
                    l2_pool.remove(assembled_choice)
                chosen = True
                continue

            # 3) Otherwise, try to create a sandwich from raw resources.
            create_cost = 1  # kdzjcqwj or h0xkysus
            # Check for raw sandwich, bread, and content. Apply small penalties if missing.
            raw_penalty = 0
            bread_penalty = 0
            content_penalty = 0
            special_bread_penalty = 0
            special_content_penalty = 0

            if raw_pool:
                # consume one raw sandwich
                raw_item = next(iter(raw_pool))
            else:
                raw_item = None
                raw_penalty = 1  # optimistic penalty

            if bread_pool:
                bread_item = next(iter(bread_pool))
            else:
                bread_item = None
                bread_penalty = 1

            if content_pool:
                content_item = next(iter(content_pool))
            else:
                content_item = None
                content_penalty = 1

            # If child needs l2, prefer bread_with_e6 and content_with_odd
            if needs_l2:
                if bread_e6_pool:
                    bread_e6_item = next(iter(bread_e6_pool))
                else:
                    bread_e6_item = None
                    special_bread_penalty = 1
                if content_odd_pool:
                    content_odd_item = next(iter(content_odd_pool))
                else:
                    content_odd_item = None
                    special_content_penalty = 1
            else:
                bread_e6_item = None
                content_odd_item = None

            # Total creation penalties
            total_create_penalty = raw_penalty + bread_penalty + content_penalty + special_bread_penalty + special_content_penalty

            # Placement and serving costs same as assembled case
            place_cost = 1
            serve_cost = 1
            move_cost = 0 if (child_table and any(tloc == child_table for tloc in tray_locations.values())) else 1
            bupp_penalty = 0 if trays_at_bupp else 1

            total_cost += create_cost + total_create_penalty + place_cost + serve_cost + move_cost + bupp_penalty

            # Consume resources if present (optimistic consumption)
            if raw_item:
                raw_pool.remove(raw_item)
            if needs_l2:
                # prefer consuming the special bread/content if available
                if bread_e6_item:
                    if bread_e6_item in bread_pool:
                        bread_pool.remove(bread_e6_item)
                    if bread_e6_item in bread_e6_pool:
                        bread_e6_pool.remove(bread_e6_item)
                elif bread_item:
                    bread_pool.remove(bread_item)
                if content_odd_item:
                    if content_odd_item in content_pool:
                        content_pool.remove(content_odd_item)
                    if content_odd_item in content_odd_pool:
                        content_odd_pool.remove(content_odd_item)
                elif content_item:
                    content_pool.remove(content_item)
            else:
                if bread_item:
                    bread_pool.remove(bread_item)
                if content_item:
                    content_pool.remove(content_item)

            # After creation, we implicitly assume the generated sandwich could be l2 (if special resources used).
            # We don't add it to assembled_pool because it is immediately consumed for this child in this estimate.

        # Final heuristic value
        return total_cost