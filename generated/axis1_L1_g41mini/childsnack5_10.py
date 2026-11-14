from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact string, removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a pattern with possible wildcards.

    :param fact: PDDL fact string, e.g. "(go9lczo2 bread1)"
    :param args: sequence of patterns to match each token of fact
    :return: True if matches, False otherwise
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (biqhd61z).

    Summary:
    =========
    This heuristic estimates the number of actions still required to reach the goal,
    which is to have (ivo59apj child) true for every child in the goal.
    Informally, the domain models a scenario where sandwiches (uguffgcw) are prepared from bread and content,
    then placed on trays (eoraohfl), and finally delivered to children (bciid6ad).
    The heuristic counts how many children still need to be served by checking which children do not yet have
    the fact (ivo59apj child). For each such child, it estimates a cost based on the
    necessary sequence of actions to serve them using domain knowledge about required predicates and actions.

    Assumptions:
    ============
    - Each child must have the fact (ivo59apj child) true at goal.
    - Serving each child involves:
      -- having a sandwich prepared (the sandwich's predicates change via kdzjcqwj or h0xkysus).
      -- placing the sandwich on a tray linked to a child's table.
      -- performing the nfarwtet or xtn2vmez action to achieve (ivo59apj child).
    - Sandwich preparation actions consume certain bread and content; sandwiches start with the predicate (zogo0zhl).
    - Trays start with (y0rxzzmc tray bupp7mhz) representing their current placement.
    - The heuristic does not guarantee admissibility but aims to capture logical task progression.

    Heuristic Initialization:
    =========================
    - Extract the set of children from static facts marked with (n2ij1fah child) or (dr25ifx3 child).
      These represent all children that may need to be served.
    - Extract locations of trays and their current placement (from static or initial facts).
    - Store predicates and objects from static facts to help identify associations like (eyker22y child table),
      which relates children to tables, and (y0rxzzmc tray bupp7mhz), representing trays on tables.
    - Identify sandwiches (uguffgcw) and their states in the current state.
    - Preprocess static facts relevant to goals and serving children.

    Step-By-Step Thinking for Computing Heuristic:
    ==============================================
    1. Identify which children have not yet been served (missing (ivo59apj child)) — these must be counted.
    2. For each unserved child:
       a) Check whether the child is associated with a table (via (eyker22y child table)).
       b) Check if there is a tray assigned to that table (via (y0rxzzmc tray some_location), with the assumption trays must be placed correctly).
       c) To serve the child, a sandwich must have been prepared and placed on the tray.
          Preparing a sandwich requires an available bread and content in the right state with predicates such as (zogo0zhl sandwich).
       d) The sandwich will be “activated” through kdzjcqwj or h0xkysus actions; the heuristic counts a fixed cost for preparing these.
       e) After that, the sandwich is placed on tray and finally delivered by nfarwtet or xtn2vmez actions.
    3. The heuristic sums estimated minimal counts of these actions per remaining child:
       - 2 to 3 for sandwich preparation,
       - 1 for sandwich placement on tray,
       - 1 for final delivery,
       resulting in an approximate fixed cost per unserved child.
    4. The heuristic returns 0 if all children are served (goal reached).
    5. This estimate avoids overly expensive computations by assuming each unserved child requires roughly a fixed number of steps,
       thus guiding the greedy best-first search efficiently.

    """

    def __init__(self, task):
        """Extract relevant static facts and goal info from the task."""
        self.goals = task.goals
        self.static = task.static

        # Extract all children from static facts, combining the two relevant predicates:
        self.children = set()
        for fact in self.static:
            if match(fact, "n2ij1fah", "*"):
                _, child = get_parts(fact)[0], get_parts(fact)[1]
                self.children.add(child)
            elif match(fact, "dr25ifx3", "*"):
                _, child = get_parts(fact)[0], get_parts(fact)[1]
                self.children.add(child)

        # Map each child to its table via (eyker22y child table)
        self.child_to_table = dict()
        for fact in self.static:
            if match(fact, "eyker22y", "*", "*"):
                parts = get_parts(fact)
                child = parts[1]
                table = parts[2]
                self.child_to_table[child] = table

        # We know tray placement is indicated by (y0rxzzmc tray location)
        # From static and initial state, tray placement can change via whgyhnl0
        # but static has only facts that do not change, so only fixed facts here.
        # We'll use this mapping dynamically in the heuristic call.

        # We store the constant bupp7mhz from domain constants to identify tray base location
        # but since it is always 'bupp7mhz' for all tray placement predicates, we keep it fixed.
        self.bupp7mhz = "bupp7mhz"

    def __call__(self, node):
        """
        Estimate number of actions to reach goal from node.state.
        """
        state = node.state

        # If goal already reached, heuristic is 0
        if self.goals <= state:
            return 0

        # Identify which children are already served: have (ivo59apj child)
        served_children = set()
        for fact in state:
            if match(fact, "ivo59apj", "*") and get_parts(fact)[1] in self.children:
                served_children.add(get_parts(fact)[1])

        unserved_children = self.children - served_children

        if not unserved_children:
            return 0  # all served

        # Identify trays and their current table placement from state:
        # (y0rxzzmc trayX location)
        # We'll map tray -> location for all trays currently known.
        tray_to_location = dict()
        for fact in state:
            if match(fact, "y0rxzzmc", "*", "*"):
                parts = get_parts(fact)
                tray = parts[1]
                location = parts[2]
                tray_to_location[tray] = location

        # Identify which sandwiches are currently in which state
        # Sandwiches are type uguffgcw, but from the problem definition this is not explicit
        # So the heuristic works without exhaustively mapping sandwiches, by counting per child.

        # We will estimate cost per unserved child:
        cost = 0

        # Costs derived from reasoning about action sequences needed per child:
        # For each child:
        # - Sandwich preparation approx 2.5 actions (kdzjcqwj or h0xkysus + preconditions)
        # - Placing sandwich on tray approx 1 action (jhzk46zy)
        # - Delivering to child approx 1 action (nfarwtet or xtn2vmez)
        # Considering efficiency and overlap, we approximate with cost_per_child = 4 actions.

        # However, to reflect some subtleties:
        # If the child's table does not have a tray placed on it (tray mapped to child's table),
        # then an extra action for tray placement would be needed (action whgyhnl0).
        # This adds 1 action for tray placement if missing.

        for child in unserved_children:
            # Base cost per child
            child_cost = 4  # preparation + placement + delivery

            # Get child's table
            table = self.child_to_table.get(child)

            # Check if a tray is currently at that table by checking tray_to_location:
            # A tray is at child's table if there exists a tray with y0rxzzmc tray table
            tray_present = False
            if table is not None:
                for tray, loc in tray_to_location.items():
                    # loc should equal the child's table for tray to be present there
                    if loc == table:
                        tray_present = True
                        break

                if not tray_present:
                    # Tray not present at child's table => 1 additional action for placing tray
                    child_cost += 1

            cost += child_cost

        return cost
