from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split a PDDL fact string into parts."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-specific heuristic for the secret_domain (biqhd61z). The domain models
    preparing sandwiches, placing them on trays and delivering them to children.
    Goals are facts of the form (ivo59apj <child>), which are achieved by
    operator nfarwtet or xtn2vmez. Those operators require a sandwich to be
    on a tray (mpvvhjpt <sandw> <tray>) and the tray to be on the child's table
    (y0rxzzmc <tray> <table>). Some children require a sandwich that has the
    additional property l2thmd6e (these children are declared with the
    static predicate n2ij1fah). Trays can be moved between tables via
    whgyhnl0, sandwiches can be prepared via kdzjcqwj or h0xkysus, and
    sandwiches can be placed on trays with jhzk46zy (jhzk46zy needs the tray
    to be on the special constant bupp7mhz before placing).

    Assumptions
    -----------
    - The heuristic aims for speed and guidance quality, not admissibility.
    - We assume the existence of at least one tray and at least one sandwich
      object in the instance (if not, the heuristic returns a large finite value).
    - Producing a sandwich (i9fq3vit) can be done in one action if
      applicable resources exist; producing the l2thmd6e property requires
      using the kdzjcqwj operator (which needs specific bread/content facts).
    - Moving a tray between tables is counted as one action (whgyhnl0).
    - Placing a sandwich on a tray (jhzk46zy) is one action but needs the
      tray to be at the special table bupp7mhz before placing; creating a
      tray-at-bupp7mhz configuration may require one move.
    - Feeding a child (nfarwtet or xtn2vmez) is a single action once the
      sandwich is on the tray and the tray is on the child's table.

    Heuristic Initialization
    ------------------------
    The constructor extracts the static information that does not change:
    - Which children are of the "n2" kind (n2ij1fah) and which are "dr" kind
      (dr25ifx3). "n2" children require sandwiches with l2thmd6e.
    - For each child we extract the associated table from eyker22y facts
      (child -> table).
    - The set of goal children is extracted from task.goals (facts of the form
      (ivo59apj <child>)).
    - The special constant string "bupp7mhz" used by jhzk46zy is kept as-is
      (the planner represents constants as strings in facts).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state, the heuristic proceeds as follows (high-level):
    1. If all goal facts are already true in the state, return 0.
    2. Parse the current state's facts into useful indexes:
       - sandwiches that are prepared (i9fq3vit), sandwiches that carry
         the l2thmd6e property, sandwiches already placed on trays (mpvvhjpt),
         and where trays currently are located (y0rxzzmc).
       - available breads/contents and other relevant predicates that allow
         producing prepared sandwiches with/without l2thmd6e.
    3. For every child that still lacks (ivo59apj child):
       - Try to find an existing mpvvhjpt sandwich on the child's table that
         satisfies the child's l2 requirement (if any). If found, cost += 1
         (just apply nfarwtet/xtn2vmez).
       - Else try to find an existing mpvvhjpt sandwich on any other table:
         if found and sandwich satisfies l2 requirement, cost += 2
         (move tray to child's table + feed).
       - Else try to reuse a prepared sandwich (i9fq3vit) by:
         a) ensuring a tray can be (or already is) at bupp7mhz (0 or 1 moves),
         b) using jhzk46zy to create mpvvhjpt (1),
         c) possibly moving the tray to the child's table (0 or 1),
         d) applying nfarwtet/xtn2vmez (1).
         This totals 2..4 actions depending on current tray locations and
         target child's table.
       - Else try to prepare a sandwich (kdzjcqwj or h0xkysus):
         - For children that require l2thmd6e prefer kdzjcqwj (it produces
           l2thmd6e) and only if necessary resources (bread with e6jz04yd and
           content with odd6wyd1 and a zogo0zhl sandwich object) are present.
         - Use the produced sandwich as in the previous step to place it on a
           tray and deliver. Account for the produce action (1) plus the
           other steps (placing and moving).
       - If no option is applicable (resource missing), add a large finite
         penalty for that child (to keep heuristic finite but high).
    4. Sum the per-child estimated minimal costs and return the total. This
       yields a fast, admissible-not-required heuristic that captures the main
       structure of required actions (prepare -> place on tray -> move tray ->
       feed) and discriminates children needing the extra l2 property.

    Implementation notes
    --------------------
    - All PDDL facts are parsed by get_parts which strips parentheses and
      splits on whitespace (this matches the planner's internal fact format).
    - The heuristic tries several delivery/production options and picks the
      cheapest feasible one for each child. The estimates are optimistic but
      informed by available objects in the state.
    """

    def __init__(self, task):
        # Goals (set of goal fact strings)
        self.goals = task.goals
        # Static facts (set of strings)
        static_facts = task.static

        # Children sets by type
        self.children_n2 = set()
        self.children_dr = set()
        # child -> table mapping (from eyker22y)
        self.child_to_table = {}

        # Parse static facts for child types and child-table mapping.
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "n2ij1fah" and len(parts) >= 2:
                self.children_n2.add(parts[1])
            elif pred == "dr25ifx3" and len(parts) >= 2:
                self.children_dr.add(parts[1])
            elif pred == "eyker22y" and len(parts) >= 3:
                child, table = parts[1], parts[2]
                self.child_to_table[child] = table

        # extract child names that are goals (goal facts are of form (ivo59apj <child>))
        self.goal_children = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "ivo59apj" and len(parts) >= 2:
                self.goal_children.append(parts[1])

        # The special table constant used by jhzk46zy is present in domain as a constant:
        self.special_table = "bupp7mhz"

        # Large finite penalty used when a child cannot be feasibly fed according
        # to visible resources in the state.
        self.PENALTY_PER_UNFEASIBLE_CHILD = 1000

    def __call__(self, node):
        state = node.state

        # If goals already satisfied -> heuristic 0
        if self.goals <= state:
            return 0

        # Index state facts
        sandwiches_i9 = set()      # sandwiches with i9fq3vit
        sandwiches_l2 = set()      # sandwiches with l2thmd6e
        sandwich_on_tray = {}      # sandwich -> tray (mpvvhjpt)
        trays_loc = {}             # tray -> table (y0rxzzmc)
        breads_go = set()          # go9lczo2 bread
        breads_e6 = set()          # e6jz04yd bread (needed by kdzjcqwj)
        contents_fq = set()        # fqungbps content
        contents_odd = set()       # odd6wyd1 content
        sandwiches_zogo = set()    # zogo0zhl sandwich (raw/unprepared)
        # Track trays present
        trays_set = set()
        # Track any existing prepared sandwich that also currently has l2 property and is on tray
        # (we already have sandwich_on_tray and sandwiches_l2 sets)

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "i9fq3vit" and len(parts) >= 2:
                sandwiches_i9.add(parts[1])
            elif pred == "l2thmd6e" and len(parts) >= 2:
                sandwiches_l2.add(parts[1])
            elif pred == "mpvvhjpt" and len(parts) >= 3:
                sand, tray = parts[1], parts[2]
                sandwich_on_tray[sand] = tray
                trays_set.add(tray)
            elif pred == "y0rxzzmc" and len(parts) >= 3:
                tray, table = parts[1], parts[2]
                trays_loc[tray] = table
                trays_set.add(tray)
            elif pred == "go9lczo2" and len(parts) >= 2:
                breads_go.add(parts[1])
            elif pred == "e6jz04yd" and len(parts) >= 2:
                breads_e6.add(parts[1])
            elif pred == "fqungbps" and len(parts) >= 2:
                contents_fq.add(parts[1])
            elif pred == "odd6wyd1" and len(parts) >= 2:
                contents_odd.add(parts[1])
            elif pred == "zogo0zhl" and len(parts) >= 2:
                sandwiches_zogo.add(parts[1])

        # Is there a tray currently at special_table (bupp7mhz)?
        any_tray_at_special = any(table == self.special_table for table in trays_loc.values())
        any_tray_exists = len(trays_set) > 0

        total_cost = 0

        # Helper to check if a sandwich meets the child's l2 requirement
        def sandwich_satisfies_child(sand, child):
            if child in self.children_n2:
                return sand in sandwiches_l2
            # dr children don't require l2
            return True

        # For each goal child not yet fed, estimate minimal cost
        for child in self.goal_children:
            goal_fact = f"(ivo59apj {child})"
            if goal_fact in state:
                continue  # already satisfied

            child_table = self.child_to_table.get(child, None)

            best_cost = None

            # Option A: existing mpvvhjpt on child's table and sandwich satisfies l2
            for sand, tray in sandwich_on_tray.items():
                tray_table = trays_loc.get(tray)
                if child_table is not None and tray_table == child_table and sandwich_satisfies_child(sand, child):
                    # one action (nfarwtet or xtn2vmez)
                    best_cost = 1
                    break
            if best_cost == 1:
                total_cost += 1
                continue

            # Option B: existing mpvvhjpt on any other table (move tray + feed)
            found_option_b = False
            for sand, tray in sandwich_on_tray.items():
                if sandwich_satisfies_child(sand, child):
                    # need one move (whgyhnl0) to child's table (if it's not already there)
                    # then one feed action
                    found_option_b = True
                    break
            if found_option_b:
                # move + feed = 2 actions
                total_cost += 2
                continue

            # Option C: use an existing prepared sandwich (i9fq3vit) and create mpvvhjpt via jhzk46zy
            # jhzk46zy requires a tray at special_table before placing
            # compute cost to get a tray at special_table (0 if already, 1 if any tray exists, else impossible)
            tray_to_special_cost = 0 if any_tray_at_special else (1 if any_tray_exists else None)

            # If there is an i9fq3vit sandwich that satisfies l2 requirement (or if child does not require l2)
            found_i9_option = False
            for sand in sandwiches_i9:
                if sandwich_satisfies_child(sand, child):
                    # cost components:
                    # - bring a tray to special_table: tray_to_special_cost (0 or 1)
                    # - jhzk46zy to create mpvvhjpt: 1
                    # - move tray from special_table to child's table (if child's table != special_table): 1 if child_table and child_table != special_table else 0
                    # - feed action: 1
                    if tray_to_special_cost is not None:
                        move_to_child = 0
                        if child_table is not None and child_table != self.special_table:
                            move_to_child = 1
                        c = tray_to_special_cost + 1 + move_to_child + 1
                        if best_cost is None or c < best_cost:
                            best_cost = c
                            found_i9_option = True
            if found_i9_option:
                total_cost += best_cost
                continue

            # Option D: prepare a sandwich first (kdzjcqwj or h0xkysus), then as in Option C
            # For children requiring l2, kdzjcqwj is required; it needs:
            #   - some bread in breads_e6
            #   - some content in contents_odd
            #   - some raw sandwich in sandwiches_zogo
            # For children not requiring l2, h0xkysus can produce i9 from any go9lczo2 bread + fqungbps content + zogo0zhl sandwich:
            #   - bread in breads_go
            #   - content in contents_fq
            #   - raw sandwich in sandwiches_zogo
            can_produce_for_l2 = (
                len(breads_e6) > 0 and len(contents_odd) > 0 and len(sandwiches_zogo) > 0
            )
            can_produce_simple = (
                len(breads_go) > 0 and len(contents_fq) > 0 and len(sandwiches_zogo) > 0
            )

            produce_possible = False
            produce_cost = None
            if child in self.children_n2:
                if can_produce_for_l2:
                    # producing via kdzjcqwj gives i9 and l2 in one action
                    produce_possible = True
                    produce_cost = 1
                else:
                    produce_possible = False
            else:
                # child does not require l2; prefer kdzjcqwj if available (it gives l2 but that's harmless);
                # otherwise h0xkysus if simple resources exist.
                if can_produce_for_l2:
                    produce_possible = True
                    produce_cost = 1
                elif can_produce_simple:
                    produce_possible = True
                    produce_cost = 1
                else:
                    produce_possible = False

            if produce_possible:
                # after producing, same costs as Option C but add produce_cost
                if tray_to_special_cost is not None:
                    move_to_child = 0
                    if child_table is not None and child_table != self.special_table:
                        move_to_child = 1
                    c = produce_cost + tray_to_special_cost + 1 + move_to_child + 1
                    # produce + (tray->special if needed) + jhzk46zy + move to child table (if needed) + feed
                    total_cost += c
                    continue

            # If none of the above options are possible with visible resources, give large penalty
            total_cost += self.PENALTY_PER_UNFEASIBLE_CHILD

        return int(total_cost)