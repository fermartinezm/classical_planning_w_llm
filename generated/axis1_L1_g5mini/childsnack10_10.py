from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Remove surrounding parentheses and split a PDDL fact string into its parts.
    Example: "(eyker22y child1 table1)" -> ["eyker22y", "child1", "table1"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic tailored to the secret_domain (biqhd61z).
    The main planning objective in the domain is to produce and deliver sandwiches
    to children (predicate ivo59apj <child>). Delivering a sandwich to a child
    usually requires:
      - a sandwich instance that has been prepared (i9fq3vit),
      - optionally the sandwich having the extra property l2thmd6e (for some child types),
      - placing the sandwich on a tray (mpvvhjpt) while the tray is at a particular table (y0rxzzmc),
      - and then executing a delivery action (nfarwtet or xtn2vmez) for the child.
    This heuristic estimates the number of actions needed to satisfy all remaining
    child-delivery goals from the current state.

    Assumptions
    -----------
    - The goal facts are of the form "(ivo59apj <child>)".
    - Static facts include child-to-table relations (eyker22y) and child types
      (n2ij1fah, dr25ifx3); those are read in the constructor and used as guidance.
    - The builder actions that produce prepared sandwiches are:
        kdzjcqwj (produces i9fq3vit and l2thmd6e under stricter preconditions)
        h0xkysus (produces i9fq3vit without l2thmd6e)
      We count either as a single "prepare" action where appropriate. kdzjcqwj is
      required when l2thmd6e is needed; otherwise h0xkysus suffices.
    - Placing a prepared sandwich on a tray is the jhzk46zy action. It requires the
      tray to be at the special table constant bupp7mhz. Moving a tray between tables
      is whgyhnl0. Delivery to a child is via nfarwtet (for n2ij1fah children) or
      xtn2vmez (for dr25ifx3 children).
    - The heuristic is allowed to be inadmissible (it may be optimistic or pessimistic)
      but aims to remain informative while being efficient to compute.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic extracts relevant static information:
      - child_to_table: mapping child -> table (from eyker22y facts in static info)
      - n2_children: set of children that require l2thmd6e (n2ij1fah)
      - dr_children: set of children that do not require l2thmd6e (dr25ifx3)
    The bupp7mhz constant (tray "home" table used by jhzk46zy) is hard-coded as
    "bupp7mhz" because that is the domain constant.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state the heuristic proceeds as follows:

    1. Quick check for goal: If all goal facts are already true in the state return 0.

    2. Parse current (dynamic) state to extract:
       - which children already have ivo59apj (delivered),
       - which sandwiches are prepared (i9fq3vit),
       - which sandwiches have l2thmd6e,
       - which sandwiches are already on trays (mpvvhjpt) and the tray they are on,
       - where trays currently are (y0rxzzmc tray table),
       - counts of raw sandwich facts (zogo0zhl) and bread/content flags for deciding if
         a kdzjcqwj (l2-producing prepare) is plausible.
       These data structures allow quick checks about available ready-to-deliver resources.

    3. Greedy per-child estimation:
       For each child that still needs delivery:
         a) If there exists a sandwich already on a tray at that child's table and, if the
            child requires l2thmd6e, that sandwich has l2thmd6e, then we only need the
            delivery action (cost = 1). Such a sandwich/tray is consumed for heuristic
            accounting.
         b) Otherwise attempt to use a prepared sandwich (i9fq3vit). To deliver it we
            normally need to:
               - place it on a tray at bupp7mhz (jhzk46zy) -> cost 1
               - move the tray from bupp7mhz to the child's table if needed (whgyhnl0) -> cost 1 if table differs
               - perform the delivery action (nfarwtet or xtn2vmez) -> cost 1
            The heuristic checks whether there are trays already at bupp7mhz or elsewhere
            and increases the estimated cost if a tray must be moved to bupp7mhz first.
            Using/consuming a prepared sandwich reduces availability for subsequent children.
         c) If no prepared sandwich with the needed l2 property exists, the heuristic assumes
            we can prepare one. Preparing is counted as a single action:
               - If the child requires l2thmd6e we assume kdzjcqwj (cost 1) is used.
               - Otherwise h0xkysus (cost 1) suffices.
            After preparing we follow step (b) to place and deliver.
         d) The per-child estimates are computed greedily and shared resources (prepared sandwiches,
            trays at bupp7mhz, general trays) are decremented as they are assigned to children,
            so the heuristic reflects some contention between goals.

    4. Sum the per-child costs and return that value. The heuristic is designed to be:
       - Zero exactly for goal states (point 2).
       - Finite for solvable states (we always count create/prepare steps and moves),
         and efficient to compute because it uses simple set/dict lookups and greedy accounting.
    """

    def __init__(self, task):
        # Store goals (set of strings)
        self.goals = set(task.goals)

        # Extract static child->table (eyker22y), and child type information.
        self.child_to_table = {}
        self.n2_children = set()  # those requiring l2thmd6e (n2ij1fah)
        self.dr_children = set()  # those matching dr25ifx3

        for fact in task.static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "eyker22y" and len(parts) >= 3:
                child, table = parts[1], parts[2]
                self.child_to_table[child] = table
            elif pred == "n2ij1fah" and len(parts) >= 2:
                self.n2_children.add(parts[1])
            elif pred == "dr25ifx3" and len(parts) >= 2:
                self.dr_children.add(parts[1])

        # Constant home table used by jhzk46zy (domain constant declared as bupp7mhz).
        self.home_table = "bupp7mhz"

    def __call__(self, node):
        state = node.state

        # If goals already reached, heuristic is 0.
        if self.goals <= state:
            return 0

        # Dynamic parsing of the state:
        prepared_sandwiches = set()   # sandwiches with i9fq3vit
        l2_sandwiches = set()         # sandwiches with l2thmd6e
        mpvv_on_tray = {}             # tray -> sandwich (for mpvvhjpt sandwich tray)
        trays_at_table = {}           # tray -> table (for y0rxzzmc)
        raw_sandwiches = set()        # zogo0zhl
        bread_set = set()             # go9lczo2 facts - available breads
        bread_has_e6 = set()          # e6jz04yd breadX facts (special bread)
        content_set = set()           # fqungbps content
        content_has_odd = set()       # odd6wyd1 content
        delivered_children = set()    # ivo59apj children already delivered

        # parse state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "i9fq3vit" and len(parts) >= 2:
                prepared_sandwiches.add(parts[1])
            elif pred == "l2thmd6e" and len(parts) >= 2:
                l2_sandwiches.add(parts[1])
            elif pred == "mpvvhjpt" and len(parts) >= 3:
                sandwich, tray = parts[1], parts[2]
                mpvv_on_tray[tray] = sandwich
            elif pred == "y0rxzzmc" and len(parts) >= 3:
                tray, table = parts[1], parts[2]
                trays_at_table[tray] = table
            elif pred == "zogo0zhl" and len(parts) >= 2:
                raw_sandwiches.add(parts[1])
            elif pred == "go9lczo2" and len(parts) >= 2:
                bread_set.add(parts[1])
            elif pred == "e6jz04yd" and len(parts) >= 2:
                bread_has_e6.add(parts[1])
            elif pred == "fqungbps" and len(parts) >= 2:
                content_set.add(parts[1])
            elif pred == "odd6wyd1" and len(parts) >= 2:
                content_has_odd.add(parts[1])
            elif pred == "ivo59apj" and len(parts) >= 2:
                delivered_children.add(parts[1])

        # Count available trays
        trays = set(trays_at_table.keys())
        # Count trays currently at the special home table
        bupp_trays = sum(1 for t, tbl in trays_at_table.items() if tbl == self.home_table)
        total_trays = len(trays)

        # Build quick maps for mpvvhjpt by (table -> list of (tray, sandwich))
        mpvv_by_table = {}
        for tray, sandwich in mpvv_on_tray.items():
            table = trays_at_table.get(tray, None)
            if table is not None:
                mpvv_by_table.setdefault(table, []).append((tray, sandwich))

        # Goals: determine which children still need delivery
        remaining_children = []
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts and parts[0] == "ivo59apj" and len(parts) >= 2:
                child = parts[1]
                if child not in delivered_children:
                    remaining_children.append(child)

        # If child->table mapping wasn't present in static, try to infer from dynamic state
        # (use eyker22y facts in the state if present)
        child_to_table = dict(self.child_to_table)  # copy static mapping
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "eyker22y" and len(parts) >= 3:
                child_to_table[parts[1]] = parts[2]

        # Resources to be consumed greedily as we assign them to children:
        available_prepared = set(prepared_sandwiches)
        available_l2 = set(l2_sandwiches)
        available_raw = set(raw_sandwiches)
        available_breads = set(bread_set)
        available_contents = set(content_set)
        available_breads_e6 = set(bread_has_e6)
        available_contents_odd = set(content_has_odd)
        used_mpvv_trays = set()  # trays whose mpvv we consumed

        total_cost = 0

        # Process more constrained children first (those requiring l2thmd6e)
        remaining_children.sort(key=lambda c: 0 if c in self.n2_children else 1)

        for child in remaining_children:
            # Determine child's table
            child_table = child_to_table.get(child, None)

            # Determine if child requires l2thmd6e
            needs_l2 = child in self.n2_children

            # 1) Check for an existing sandwich already on a tray at the child's table.
            used_existing = False
            if child_table is not None and child_table in mpvv_by_table:
                for tray, sandwich in mpvv_by_table[child_table]:
                    if tray in used_mpvv_trays:
                        continue  # already used in heuristic accounting
                    if needs_l2 and sandwich not in available_l2 and sandwich not in l2_sandwiches:
                        continue  # doesn't have required property
                    # Use this mpvv for delivery
                    used_mpvv_trays.add(tray)
                    total_cost += 1  # delivery action nfarwtet or xtn2vmez
                    used_existing = True
                    break
            if used_existing:
                continue

            # Helper to compute cost of placing a (prepared) sandwich on a tray and delivering
            def place_and_deliver_cost(target_table):
                nonlocal bupp_trays, total_trays
                cost = 0
                # Need a tray at home table to perform jhzk46zy (place requires tray at bupp7mhz)
                if bupp_trays > 0:
                    # Place takes 1
                    cost += 1  # jhzk46zy
                    bupp_trays -= 1
                    # The tray after placement will be at home table; if child's table differs, move it
                    if target_table is not None and target_table != self.home_table:
                        cost += 1  # whgyhnl0 move to child's table
                    # Delivery
                    cost += 1  # nfarwtet / xtn2vmez
                    # After usage the tray will be at child's table; it remains available but we treat it as consumed implicitly.
                    total_trays = max(0, total_trays - 0)  # no change in total trays count here; we've consumed bupp_trays count already
                    return cost
                elif total_trays > 0:
                    # No tray currently at home; we can move some tray to home (1), then place (1),
                    # possibly move to child's table (1) and deliver (1)
                    cost += 1  # move some tray to home
                    cost += 1  # jhzk46zy place
                    if target_table is not None and target_table != self.home_table:
                        cost += 1  # move to child's table
                    cost += 1  # delivery
                    # After this operation we still have same number of trays but we've used one for the sequence
                    total_trays = max(0, total_trays - 1)
                    return cost
                else:
                    # No trays at all in the world (unlikely). Estimate an upper bound:
                    # create/introduce tray (1) + place (1) + move (maybe 1) + deliver (1)
                    cost += 1  # create/assume tray available
                    cost += 1  # jhzk46zy
                    if target_table is not None and target_table != self.home_table:
                        cost += 1
                    cost += 1
                    return cost

            # 2) Try using an available prepared sandwich that meets requirement
            chosen_prepared = None
            for s in list(available_prepared):
                if needs_l2 and s not in available_l2:
                    continue
                chosen_prepared = s
                break
            if chosen_prepared is not None:
                # Use it: place and deliver
                cost = place_and_deliver_cost(child_table)
                total_cost += cost
                # consume prepared sandwich
                available_prepared.remove(chosen_prepared)
                if chosen_prepared in available_l2:
                    available_l2.discard(chosen_prepared)
                continue

            # 3) No prepared sandwich available of required type. Try to prepare one from raw or by using actions.
            #    Preparing counts as 1 (kdzjcqwj if l2 needed, else h0xkysus). We check presence of necessary
            #    bread/content if available; otherwise still count the prepare action.
            can_prepare_l2 = False
            if needs_l2:
                # kdzjcqwj requires e6jz04yd bread and odd6wyd1 content (per domain) — check existence
                if available_breads_e6 and available_contents_odd and available_raw:
                    can_prepare_l2 = True
                elif available_raw:
                    # raw exists but special bread/content might be missing; still allow prepare but it's optimistic
                    can_prepare_l2 = True
                else:
                    # no raw sandwich; we optimistically assume we can still prepare one (count the prepare action)
                    can_prepare_l2 = True
            else:
                # h0xkysus requires go9lczo2 and fqungbps and zogo0zhl; check approximately
                if available_breads and available_contents and available_raw:
                    can_prepare_l2 = True
                elif available_raw:
                    can_prepare_l2 = True
                else:
                    can_prepare_l2 = True

            # Count the prepare action
            total_cost += 1  # prepare (either kdzjcqwj or h0xkysus)
            # If preparation consumes particular resources, we greedily remove one
            if available_raw:
                used_raw = next(iter(available_raw))
                available_raw.remove(used_raw)
            if needs_l2:
                # consume one bread and one content preferentially from the special sets
                if available_breads_e6:
                    available_breads_e6.pop()
                elif available_breads:
                    available_breads.pop()
                if available_contents_odd:
                    available_contents_odd.pop()
                elif available_contents:
                    available_contents.pop()
                # After prepare, we have a prepared sandwich with l2
                # represent it as newly available prepared sandwich
                # use a synthetic name to avoid collision with existing sandwich ids
                synthetic = f"__synth_l2_for_{child}"
                available_prepared.add(synthetic)
                available_l2.add(synthetic)
                # Now place & deliver
                cost = place_and_deliver_cost(child_table)
                total_cost += cost
                # consume the synthetic prepared sandwich immediately
                available_prepared.discard(synthetic)
                available_l2.discard(synthetic)
            else:
                # prepare non-l2 sandwich
                if available_breads:
                    available_breads.pop()
                if available_contents:
                    available_contents.pop()
                synthetic = f"__synth_nonl2_for_{child}"
                available_prepared.add(synthetic)
                # place & deliver
                cost = place_and_deliver_cost(child_table)
                total_cost += cost
                available_prepared.discard(synthetic)

        # total_cost is our heuristic estimate
        # Ensure returned value is an int and finite
        if total_cost is None:
            return 0
        return int(total_cost)