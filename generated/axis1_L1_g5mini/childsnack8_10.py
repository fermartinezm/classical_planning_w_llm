from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the token list of a fact string '(pred arg1 arg2)' -> ['pred','arg1','arg2']"""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (biqhd61z).
    The domain models preparing sandwiches, placing them on trays and delivering
    them to children at particular tables. Goals are facts of the form
    (ivo59apj <child>) meaning the child has been served.

    This heuristic estimates the number of actions required to serve all
    children that are not yet served in the current state. It simulates a
    greedy, optimistic (delete-relaxed-like) plan in which already-existing
    resources (prepared sandwiches, raw components, trays, sandwiches-on-trays)
    are consumed when assigned to a child. The heuristic is intentionally
    optimistic about re-use where realistic, and otherwise counts concrete
    actions needed: preparing sandwiches, placing them on trays, moving trays,
    and the final serve action.

    Assumptions
    -----------
    - Goals are of the predicate ivo59apj applied to children.
    - Each child has a static mapping to a table via (eyker22y child table).
    - Children are of two types: those in n2ij1fah (require sandwiches with
      the l2thmd6e property) and those in dr25ifx3 (do not require l2thmd6e).
    - The planner state provides facts for:
        - sandwiches: zogo0zhl (raw), i9fq3vit (prepared), l2thmd6e (special prop)
        - trays: y0rxzzmc <tray> <table>
        - sandwich-on-tray: mpvvhjpt <sandw> <tray>
        - breads and contents: go9lczo2, e6jz04yd (special bread),
          fqungbps, odd6wyd1 (special content)
    - The actions available and their costs (we count each action as cost 1):
        - kdzjcqwj/h0xkysus: produce i9fq3vit (kdzjcqwj also produces l2thmd6e).
        - jhzk46zy: place prepared sandwich on a tray; requires the tray to be
          at the special table bupp7mhz.
        - whgyhnl0: move a tray from its current table to another table.
        - nfarwtet/xtn2vmez: serve a child from a sandwich on a tray at the
          child's table (nfarwtet requires l2thmd6e, xtn2vmez does not).
    - We treat resources (raw sandwiches, prepared sandwiches, breads,
      contents, trays) as consumable during the simulated assignment
      (to avoid double-counting), and update tray positions when we simulate
      moves. This produces a lower/optimistic (but informative) estimate.

    Heuristic Initialization
    ------------------------
    - Extract the set of goal facts from the task.
    - Extract static facts:
        - mapping child -> table from eyker22y
        - sets of children in n2ij1fah and dr25ifx3
    - No further preprocessing is required.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state, the heuristic performs the following steps:

    1. Quick check: if all goals are already true in the current state, return 0.

    2. Parse the current state into convenient data structures:
       - trays: dict tray -> table (from y0rxzzmc facts)
       - mp_on_tray: dict sandwich -> tray (from mpvvhjpt facts)
       - prepared: set of sandwiches with i9fq3vit
       - special: set of sandwiches with l2thmd6e
       - raw: set of sandwiches with zogo0zhl
       - breads: sets for go9lczo2 and e6jz04yd (special bread)
       - contents: sets for fqungbps and odd6wyd1 (special content)

    3. Identify children that still need serving (goals not present in state).
       Process them in a constrained-first order: children requiring the
       special l2thmd6e property are handled first (they are more constrained).

    4. For each child, greedily choose the least-cost way to produce the goal,
       updating resource structures to reflect consumed/updated resources:
       - Direct serve from sandwich already on a tray at the child's table:
           cost += 1 (nfarwtet/xtn2vmez). Consume that mpvvhjpt and the sandwich.
       - Move an existing tray that already holds a suitable sandwich to the
         child's table and serve:
           cost += 2 (whgyhnl0 + serve). Update tray position, consume mp and sandwich.
       - If a prepared sandwich (i9fq3vit) exists:
           * Ensure a tray is at bupp7mhz: if not, move one there (+1).
           * Use jhzk46zy to place sandwich on that tray (+1).
           * Move the tray to the child's table if needed (+1).
           * Serve (+1).
           Total: typically 3 if a tray already at bupp7mhz, or 4 otherwise.
           Update resource sets and tray positions accordingly.
       - If only raw components (zogo0zhl) are available:
           * Try to produce a special sandwich (kdzjcqwj) if child needs it and
             a special bread (e6jz04yd) and special content (odd6wyd1) exist:
               +1 to create (kdzjcqwj), then follow the "prepared" steps.
           * Otherwise create a normal prepared sandwich (h0xkysus) (+1),
             then follow "prepared" steps (but cannot be used for special-requiring child).
       - If none of the above resources exist to serve a child, the heuristic
         returns a large finite penalty (e.g. 1000) to indicate high cost /
         practical impossibility under current resources.

    5. Sum the chosen costs for all unserved children and return the sum.

    The heuristic is efficient (linear scans / greedy resource updates) and
    returns 0 for goal states. It is optimistic because it ignores many delete
    interactions and often assumes that a single simulated action suffices to
    make resources available; this tends to produce informative guidance for
    greedy best-first search while remaining fast to compute.
    """

    def __init__(self, task):
        # store goal facts (frozenset of strings)
        self.goals = task.goals
        # extract static facts relevant to mapping children to tables and types
        self.child_to_table = {}
        self.children_need_special = set()  # n2ij1fah
        self.children_other = set()  # dr25ifx3
        for fact in task.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "eyker22y" and len(parts) >= 3:
                child, table = parts[1], parts[2]
                self.child_to_table[child] = table
            elif pred == "n2ij1fah" and len(parts) >= 2:
                self.children_need_special.add(parts[1])
            elif pred == "dr25ifx3" and len(parts) >= 2:
                self.children_other.add(parts[1])

    def __call__(self, node):
        state = node.state

        # Quick goal check: if goals satisfied, return 0
        if self.goals <= state:
            return 0

        # Parse state into resources and positioned items
        trays = {}  # tray -> table
        mp_on_tray = {}  # sandwich -> tray
        prepared = set()  # i9fq3vit
        special = set()  # l2thmd6e
        raw = set()  # zogo0zhl
        breads = set()  # go9lczo2
        bread_special = set()  # e6jz04yd
        contents = set()  # fqungbps
        content_special = set()  # odd6wyd1
        served_children = set()  # ivo59apj already present

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "y0rxzzmc" and len(parts) >= 3:
                tray, table = parts[1], parts[2]
                trays[tray] = table
            elif pred == "mpvvhjpt" and len(parts) >= 3:
                sand, tray = parts[1], parts[2]
                mp_on_tray[sand] = tray
            elif pred == "i9fq3vit" and len(parts) >= 2:
                prepared.add(parts[1])
            elif pred == "l2thmd6e" and len(parts) >= 2:
                special.add(parts[1])
            elif pred == "zogo0zhl" and len(parts) >= 2:
                raw.add(parts[1])
            elif pred == "go9lczo2" and len(parts) >= 2:
                breads.add(parts[1])
            elif pred == "e6jz04yd" and len(parts) >= 2:
                bread_special.add(parts[1])
            elif pred == "fqungbps" and len(parts) >= 2:
                contents.add(parts[1])
            elif pred == "odd6wyd1" and len(parts) >= 2:
                content_special.add(parts[1])
            elif pred == "ivo59apj" and len(parts) >= 2:
                served_children.add(parts[1])

        # Identify which children still need serving (from goals)
        target_children = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "ivo59apj" and len(parts) >= 2:
                child = parts[1]
                if child not in served_children:
                    need_special = child in self.children_need_special
                    table = self.child_to_table.get(child)
                    target_children.append((need_special, child, table))

        # Process more constrained children first (those needing special sandwich)
        target_children.sort(reverse=True)  # True (need_special) first

        total_cost = 0

        # Helper to find a tray at a specific table
        def find_tray_at(table):
            for t, tab in trays.items():
                if tab == table:
                    return t
            return None

        # Helper to pick any tray (return id) or None
        def pick_any_tray():
            for t in trays:
                return t
            return None

        # Greedily serve each child
        for need_special, child, child_table in target_children:
            # If child's table unknown (static missing), assume extra cost but try
            if child_table is None:
                # pessimistic but finite: assume requires dedicated actions
                total_cost += 4
                continue

            # 1) Direct sandwich-on-tray at child's table that matches need
            served = False
            for sand, tray in list(mp_on_tray.items()):
                if trays.get(tray) == child_table:
                    if (not need_special) or (sand in special):
                        # can serve immediately
                        total_cost += 1  # nfarwtet/xtn2vmez
                        # consume the sandwich-on-tray and the sandwich itself
                        mp_on_tray.pop(sand, None)
                        prepared.discard(sand)
                        special.discard(sand)
                        served = True
                        break
            if served:
                continue

            # 2) Move a tray that already holds a suitable sandwich, then serve
            moved_and_served = False
            for sand, tray in list(mp_on_tray.items()):
                if (not need_special) or (sand in special):
                    # move tray to child's table, then serve
                    total_cost += 2  # whgyhnl0 + serve
                    trays[tray] = child_table  # update tray location
                    mp_on_tray.pop(sand, None)
                    prepared.discard(sand)
                    special.discard(sand)
                    moved_and_served = True
                    break
            if moved_and_served:
                continue

            # 3) Use an already prepared sandwich (i9fq3vit)
            chosen_sand = None
            for s in prepared:
                if (not need_special) or (s in special):
                    chosen_sand = s
                    break
            if chosen_sand is not None:
                # We need a tray at bupp7mhz for jhzk46zy
                BUpp = "bupp7mhz"
                tray_at_bupp = find_tray_at(BUpp)
                used_tray = None
                if tray_at_bupp is None:
                    # move any tray to bupp7mhz if exists
                    any_tray = pick_any_tray()
                    if any_tray is None:
                        # no tray exists at all, very bad; penalize heavily and continue
                        return 1000
                    # move chosen tray to bupp7mhz
                    total_cost += 1  # whgyhnl0 to move to bupp7mhz
                    trays[any_tray] = BUpp
                    tray_at_bupp = any_tray
                # place sandwich on that tray
                total_cost += 1  # jhzk46zy
                mp_on_tray[chosen_sand] = tray_at_bupp
                # now move to child's table if needed
                if trays[tray_at_bupp] != child_table:
                    total_cost += 1  # whgyhnl0 move to child_table
                    trays[tray_at_bupp] = child_table
                # serve
                total_cost += 1  # nfarwtet/xtn2vmez
                # consume sandwich and mpvvhjpt
                mp_on_tray.pop(chosen_sand, None)
                prepared.discard(chosen_sand)
                special.discard(chosen_sand)
                continue

            # 4) No prepared sandwich: try to create one from raw
            chosen_raw = None
            if need_special:
                # prefer kdzjcqwj which yields special: need raw + some bread in bread_special + content in content_special
                if raw and bread_special and content_special:
                    chosen_raw = next(iter(raw))
                    # consume one bread & content (optimistic)
                    # remove arbitrary items
                    b = next(iter(bread_special))
                    c = next(iter(content_special))
                    bread_special.discard(b)
                    breads.discard(b)
                    content_special.discard(c)
                    contents.discard(c)
                    raw.discard(chosen_raw)
                    # kdzjcqwj produces a prepared sandwich with special property
                    new_sand = chosen_raw  # we reuse the raw's name as sandwich id
                    prepared.add(new_sand)
                    special.add(new_sand)
                    # creating took 1 action
                    total_cost += 1  # kdzjcqwj
                    # now follow prepared sandwich steps: reuse code by falling through to placing on tray
                    # Try to find tray at bupp7mhz
                    BUpp = "bupp7mhz"
                    tray_at_bupp = find_tray_at(BUpp)
                    if tray_at_bupp is None:
                        any_tray = pick_any_tray()
                        if any_tray is None:
                            return 1000
                        total_cost += 1
                        trays[any_tray] = BUpp
                        tray_at_bupp = any_tray
                    total_cost += 1  # jhzk46zy
                    mp_on_tray[new_sand] = tray_at_bupp
                    if trays[tray_at_bupp] != child_table:
                        total_cost += 1  # move
                        trays[tray_at_bupp] = child_table
                    total_cost += 1  # serve
                    mp_on_tray.pop(new_sand, None)
                    prepared.discard(new_sand)
                    special.discard(new_sand)
                    continue
                else:
                    # cannot produce special sandwich from raw with available special bread/content
                    # try an optimistic fallback: if no chance to produce, penalize heavily
                    # but if there are raw + normal bread+content but not special flags, we still cannot serve this child,
                    # so give a high but finite cost to steer search away
                    if raw and breads and contents:
                        # create non-special sandwich (h0xkysus) won't help, so penalize more
                        total_cost += 20
                        # still consume resources to avoid infinite loop in counting
                        r = next(iter(raw))
                        raw.discard(r)
                        b = next(iter(breads))
                        breads.discard(b)
                        c = next(iter(contents))
                        contents.discard(c)
                        continue
                    else:
                        # no resources at all: large penalty
                        return 1000
            else:
                # child does not require special: we can create a normal sandwich if bread+content+raw available
                if raw and breads and contents:
                    chosen_raw = next(iter(raw))
                    raw.discard(chosen_raw)
                    # consume some bread & content (optimistic)
                    b = next(iter(breads))
                    breads.discard(b)
                    c = next(iter(contents))
                    contents.discard(c)
                    # produce prepared (without special)
                    new_sand = chosen_raw
                    prepared.add(new_sand)
                    # cost for creation
                    total_cost += 1  # h0xkysus
                    # now place on tray & move & serve as in prepared case
                    BUpp = "bupp7mhz"
                    tray_at_bupp = find_tray_at(BUpp)
                    if tray_at_bupp is None:
                        any_tray = pick_any_tray()
                        if any_tray is None:
                            return 1000
                        total_cost += 1
                        trays[any_tray] = BUpp
                        tray_at_bupp = any_tray
                    total_cost += 1  # jhzk46zy
                    mp_on_tray[new_sand] = tray_at_bupp
                    if trays[tray_at_bupp] != child_table:
                        total_cost += 1  # move
                        trays[tray_at_bupp] = child_table
                    total_cost += 1  # serve
                    mp_on_tray.pop(new_sand, None)
                    prepared.discard(new_sand)
                    special.discard(new_sand)
                    continue
                else:
                    # No raw+components; could try to create from prepared later, but for now penalize
                    return 1000

        return total_cost