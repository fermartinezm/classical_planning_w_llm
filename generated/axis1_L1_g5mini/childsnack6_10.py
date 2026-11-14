from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split the fact into tokens."""
    return fact.strip()[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the secret_domain (biqhd61z).
    The planning task models preparing sandwiches, placing them on trays,
    moving trays to children tables, and finally serving children. The goal
    facts are of the form (ivo59apj <child>) and are achieved by delivering a
    sandwich on a tray to the child's table and applying the serving operator.
    This heuristic estimates the number of actions required to achieve all
    goal child-served facts from the current state.

    Assumptions
    -----------
    - The goal set consists exclusively of child-served facts "(ivo59apj childX)".
    - There are two "serve" operators (nfarwtet / xtn2vmez) that require:
      * a sandwich on a tray (mpvvhjpt <sandw> <tray>)
      * the sandwich to have the l2thmd6e property (produced only by kdzjcqwj)
      * the tray to be at the child's table (y0rxzzmc <tray> <table>)
      * the child to have the appropriate child-type fact (n2ij1fah or dr25ifx3)
      and a static eyker22y mapping linking child -> table.
    - A sandwich must be prepared (operators kdzjcqwj or h0xkysus produce i9fq3vit,
      but only kdzjcqwj produces l2thmd6e which is required by the serve operators).
    - jhzk46zy places a prepared sandwich (i9fq3vit) onto a tray but requires
      the tray to be located at the specific constant bupp7mhz (the domain constant
      introduced in the domain). After jhzk46zy, the sandwich is on the tray
      (mpvvhjpt) and can be moved to the child's table using whgyhnl0.
    - whgyhnl0 moves a tray from its current table to another table in one action.
    - The heuristic treats each (unserved) child independently and sums per-child
      estimated costs. It greedily accounts for limited consumable resources
      (counts of bread, content, and raw sandwich items) so that one prepared
      sandwich cannot be assigned simultaneously to multiple children.
    - The heuristic is not guaranteed to be admissible (it may overestimate),
      but aims to be informative and fast to compute.

    Heuristic Initialization
    ------------------------
    The constructor extracts:
    - self.goals: the set of goal facts (task.goals).
    - child -> table mapping from static facts of the form (eyker22y child table).
    - child type classification sets: self.n2_children (require nfarwtet),
      self.dr_children (require xtn2vmez).
    - No other static structure is required, but this mapping is critical to
      determine where each child must be served.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given planner node/state:

    1) If all goal facts are already true in the state, return 0.

    2) Gather dynamic facts:
       - Which trays are at which tables (y0rxzzmc <tray> <table>).
       - Which sandwiches are prepared (i9fq3vit <sandw>).
       - Which sandwiches have the special property l2thmd6e <sandw>.
       - Which sandwiches are currently on trays (mpvvhjpt <sandw> <tray>).
       - Counts of raw consumables:
         * breads (go9lczo2 facts)
         * contents (fqungbps facts)
         * raw-sandwich items (zogo0zhl facts)
       These counts are used to ensure we do not double-assign the same limited
       consumable to multiple sandwich preparations.

    3) For each goal child that is not yet served, estimate the minimal number of
       actions to serve that particular child. The heuristic tries several
       scenarios (from cheapest to more expensive) and picks the minimum:

       Scenario A: There is already a sandwich on a tray with l2thmd6e and the tray
                   is located at the child's table. -> cost = 1 (serve operator).

       Scenario B: There is a sandwich on a tray with l2thmd6e, but the tray is
                   at a different table. -> cost = 2 (move tray to child + serve).

       Scenario C: There exists a prepared sandwich that already has l2thmd6e
                   but is not on a tray. To put it onto a tray we must use
                   jhzk46zy which requires a tray to be at the special location
                   'bupp7mhz'. If there is a tray at bupp7mhz currently:
                     - If child's table == bupp7mhz -> cost = 2 (jhzk46zy + serve).
                     - Else -> cost = 3 (jhzk46zy + move tray to child + serve).
                   If there is no tray at bupp7mhz but there is at least one tray
                   anywhere, we conservatively add +1 cost to represent moving a
                   tray to bupp7mhz before applying jhzk46zy.

       Scenario D: No prepared sandwich with l2thmd6e exists. If there are
                   consumable resources available (bread, content, raw sandwich),
                   we estimate creating a prepared sandwich via kdzjcqwj (cost 1),
                   then follow the logic of Scenario C for placing it on a tray
                   and serving. If resources are unavailable the heuristic still
                   returns a finite fallback estimate (a moderately large number)
                   to keep the value finite.

    4) The heuristic greedily reserves the consumable resources for children as
       it assigns Scenario D to avoid double-counting a single bread/content
       for multiple planned preparations.

    5) The final heuristic value is the sum of the per-child minimal costs.

    Notes on completeness and finiteness:
    - The heuristic is 0 exactly when all goal facts are present in the state.
    - If the problem is solvable from the given state, the heuristic returns a
      finite integer estimate; if there are insufficient resources, the heuristic
      still returns a finite fallback estimate (but the search will detect
      unsolvability).
    """

    def __init__(self, task):
        # Required by the planner framework: the set of goal facts.
        self.goals = task.goals
        # Extract static mapping child -> table (eyker22y child table)
        self.child_to_table = {}
        self.n2_children = set()
        self.dr_children = set()

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "eyker22y" and len(parts) >= 3:
                child = parts[1]
                table = parts[2]
                self.child_to_table[child] = table
            elif parts[0] == "n2ij1fah" and len(parts) >= 2:
                self.n2_children.add(parts[1])
            elif parts[0] == "dr25ifx3" and len(parts) >= 2:
                self.dr_children.add(parts[1])

        # Name of the special location required by jhzk46zy from the domain.
        # The domain declares a constant called 'bupp7mhz'; we use that literal.
        self.special_tray_location = "bupp7mhz"

    def __call__(self, node):
        state = node.state

        # If goal already satisfied -> heuristic 0
        if self.goals <= state:
            return 0

        # Parse dynamic facts
        # - tray locations: tray -> table
        tray_loc = {}  # tray -> table
        # - sandwiches that are prepared
        prepared = set()  # i9fq3vit <sandw>
        # - sandwiches with l2thmd6e property
        l2_sandw = set()
        # - mpvvhjpt facts: sandwich -> tray
        sand_on_tray = {}  # sand -> tray
        # - counts of resources
        bread_count = 0
        content_count = 0
        raw_sandw_count = 0

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "y0rxzzmc" and len(parts) >= 3:
                tray, table = parts[1], parts[2]
                tray_loc[tray] = table
            elif pred == "i9fq3vit" and len(parts) >= 2:
                prepared.add(parts[1])
            elif pred == "l2thmd6e" and len(parts) >= 2:
                l2_sandw.add(parts[1])
            elif pred == "mpvvhjpt" and len(parts) >= 3:
                sand, tray = parts[1], parts[2]
                sand_on_tray[sand] = tray
            elif pred == "go9lczo2":
                bread_count += 1
            elif pred == "fqungbps":
                content_count += 1
            elif pred == "zogo0zhl":
                raw_sandw_count += 1

        # Convenience: set of trays and whether any tray is at special location
        trays = set(tray_loc.keys())
        special_tray_present = any(loc == self.special_tray_location for loc in tray_loc.values())

        # Determine list of children to serve from goals (extract child names)
        children = []
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "ivo59apj" and len(parts) >= 2:
                children.append(parts[1])

        # We'll greedily assign available consumables when planning to prepare sandwiches.
        remaining_bread = bread_count
        remaining_content = content_count
        remaining_raw = raw_sandw_count

        total_cost = 0

        for child in children:
            # If child already served in current state, cost 0
            goal_fact = f"(ivo59apj {child})"
            if goal_fact in state:
                continue

            # Child's required table
            child_table = self.child_to_table.get(child, None)

            # First attempt: see if there exists mpvvhjpt(sand, tray) with sand having l2thmd6e
            best_cost_for_child = None

            for sand, tray in sand_on_tray.items():
                if sand not in l2_sandw:
                    continue
                tray_table = tray_loc.get(tray, None)
                if tray_table is None or child_table is None:
                    continue
                if tray_table == child_table:
                    cost = 1  # just serve
                else:
                    cost = 2  # move tray + serve
                if best_cost_for_child is None or cost < best_cost_for_child:
                    best_cost_for_child = cost
                    if cost == 1:
                        break  # can't do better than 1

            if best_cost_for_child == 1:
                total_cost += 1
                continue
            if best_cost_for_child == 2:
                total_cost += 2
                continue

            # Second attempt: any sandwich prepared and having l2thmd6e (but not on tray)
            prepared_l2 = [s for s in prepared if s in l2_sandw and s not in sand_on_tray]
            if prepared_l2:
                # we can use one of these prepared sandwiches
                # cost depends on availability of a tray at special location
                if special_tray_present:
                    if child_table == self.special_tray_location:
                        cost = 2  # put on tray (jhzk46zy) + serve
                    else:
                        cost = 3  # jhzk46zy + move tray + serve
                elif trays:
                    # need to move some tray to special location first (+1)
                    if child_table == self.special_tray_location:
                        cost = 3  # move tray to special + jhzk + serve
                    else:
                        cost = 4  # move to special + jhzk + move to child + serve
                else:
                    # no trays at all (unlikely), fallback: create tray movement overhead
                    if child_table == self.special_tray_location:
                        cost = 4
                    else:
                        cost = 5
                total_cost += cost
                continue

            # Third attempt: prepare a sandwich first (kdzjcqwj) then place on tray and serve
            # Check available consumables for preparation producing l2thmd6e
            if remaining_bread >= 1 and remaining_content >= 1 and remaining_raw >= 1:
                # consume resources for this planned sandwich
                remaining_bread -= 1
                remaining_content -= 1
                remaining_raw -= 1

                # cost to prepare = 1
                prep_cost = 1

                # then same logic as prepared_l2 case but with prep_cost added
                if special_tray_present:
                    if child_table == self.special_tray_location:
                        cost = prep_cost + 2  # prep + jhzk + serve
                    else:
                        cost = prep_cost + 3  # prep + jhzk + move + serve
                elif trays:
                    if child_table == self.special_tray_location:
                        cost = prep_cost + 3  # move tray to special + jhzk + serve
                    else:
                        cost = prep_cost + 4  # move to special + jhzk + move + serve
                else:
                    if child_table == self.special_tray_location:
                        cost = prep_cost + 4
                    else:
                        cost = prep_cost + 5
                total_cost += cost
                continue

            # If we reach here, we neither have an appropriate sandwich on a tray,
            # nor a prepared sandwich, nor enough consumables to prepare one.
            # Return a conservative finite estimate for this child.
            # Choose a fallback that is larger than normal sequences but finite.
            fallback_cost = 6
            total_cost += fallback_cost

        return total_cost