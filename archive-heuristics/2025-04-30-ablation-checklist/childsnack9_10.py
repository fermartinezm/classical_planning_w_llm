from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts represented as strings
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully, though PDDL facts are structured.
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    This heuristic estimates the number of actions required to serve all children.
    It breaks down the process into stages: making sandwiches, putting them on
    trays, moving trays to children's locations, and serving the children.

    The heuristic counts the number of unserved children and estimates the
    minimum number of actions needed to get a suitable sandwich to each of them,
    considering the current state of sandwiches and trays.

    Heuristic components:
    1.  Number of sandwiches of each type (gluten-free/regular) that still need to be made.
    2.  Number of sandwiches of each type that are made but not yet on a tray.
    3.  Number of locations where unserved children are waiting but no tray is present.
    4.  Number of unserved children (representing the final 'serve' action for each).

    Assumptions and Simplifications:
    - Assumes sufficient bread/content and 'notexist' sandwich objects are available
      in total to make all needed sandwiches.
    - Assumes sufficient trays are available in total.
    - Estimates tray moves by counting locations needing a tray, not specific tray movements.
    - Does not account for potential bottlenecks like limited resources in the kitchen
      at any given moment, or trays needing to move *to* the kitchen first.
    - Each component is added linearly, ignoring potential parallelization or
      complex interactions between actions (e.g., one tray move enabling multiple serves).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal information and static facts.

        - Identify all children that need to be served (from goals).
        - Map children to their allergy status and waiting location (from static facts).
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Identify all children that are goals (i.e., need to be served)
        self.goal_children = set()
        for goal in self.goals:
             if match(goal, "served", "*"):
                  _, child = get_parts(goal)
                  self.goal_children.add(child)

        # Map children to their allergy status and waiting location from static facts
        self.child_info = {} # {child_name: {'allergy': 'gluten'/'not_gluten', 'location': 'place_name'}}

        for fact in self.static_facts:
            if match(fact, "waiting", "*", "*"):
                _, child, location = get_parts(fact)
                if child in self.goal_children: # Only track info for children we need to serve
                    if child not in self.child_info:
                         self.child_info[child] = {}
                    self.child_info[child]['location'] = location
            elif match(fact, "allergic_gluten", "*"):
                _, child = get_parts(fact)
                if child in self.goal_children:
                    if child not in self.child_info:
                         self.child_info[child] = {}
                    self.child_info[child]['allergy'] = 'gluten'
            elif match(fact, "not_allergic_gluten", "*"):
                _, child = get_parts(fact)
                if child in self.goal_children:
                    if child not in self.child_info:
                         self.child_info[child] = {}
                    self.child_info[child]['allergy'] = 'not_gluten'

        # Ensure all goal children have complete info (should be true for valid problems)
        # If not, the heuristic might be less accurate, but we proceed with available info.


    def __call__(self, node):
        """
        Compute the heuristic estimate for the given state.

        Estimates the remaining actions by summing up:
        - Sandwiches needing to be made (per type).
        - Sandwiches needing to be put on trays (per type).
        - Tray moves needed to reach locations with unserved children.
        - Serve actions needed (one per unserved child).
        """
        state = node.state

        # 1. Identify unserved children and their needs/locations
        unserved_children = set()
        u_gf = 0 # Unserved gluten-free children
        u_reg = 0 # Unserved regular children
        unserved_locations = set() # Locations where unserved children are waiting

        for child in self.goal_children:
            if f"(served {child})" not in state:
                unserved_children.add(child)
                info = self.child_info.get(child)
                if info: # Should exist based on __init__ logic
                    if info.get('allergy') == 'gluten':
                        u_gf += 1
                    else: # Assume not_gluten if info exists but allergy isn't 'gluten'
                        u_reg += 1
                    if info.get('location'):
                         unserved_locations.add(info['location'])

        # If no children are unserved, the goal is reached
        if not unserved_children:
            return 0

        # 2. Count existing sandwiches and their status (made, on tray, type)
        m_gf = 0 # Made GF (at_kitchen_sandwich or ontray)
        m_reg = 0 # Made Reg (at_kitchen_sandwich or ontray)
        ot_gf = 0 # Ontray GF
        ot_reg = 0 # Ontray Reg

        # Find all made sandwiches and classify them by type
        made_sandwiches = set()
        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                _, s = get_parts(fact)
                made_sandwiches.add(s)
            elif match(fact, "ontray", "*", "*"):
                _, s, t = get_parts(fact)
                made_sandwiches.add(s)

        for s in made_sandwiches:
             is_gf = f"(no_gluten_sandwich {s})" in state
             if is_gf:
                  m_gf += 1
             else:
                  m_reg += 1 # Sandwiches not marked no_gluten are treated as regular

        # Find all sandwiches currently on trays and classify them by type
        for fact in state:
             if match(fact, "ontray", "*", "*"):
                  _, s, t = get_parts(fact)
                  is_gf = f"(no_gluten_sandwich {s})" in state
                  if is_gf:
                       ot_gf += 1
                  else:
                       ot_reg += 1 # Sandwiches not marked no_gluten are treated as regular

        # 3. Count locations that need a tray move
        locations_with_trays = set()
        for fact in state:
            if match(fact, "at", "*", "*"):
                _, obj, loc = get_parts(fact)
                # In this domain, 'at' predicate is used for trays and kitchen constant.
                # We are interested in trays at places other than kitchen for delivery.
                # Assuming objects starting with 'tray' are trays.
                if obj.startswith('tray'):
                     locations_with_trays.add(loc)

        # Count locations with unserved children that do not currently have a tray
        locations_needing_tray_move = 0
        for loc in unserved_locations:
            # Children waiting at the kitchen is unlikely but possible.
            # A tray is needed at any location with unserved children, unless one is already there.
            if loc not in locations_with_trays:
                locations_needing_tray_move += 1

        # 4. Calculate heuristic components based on counts
        # Cost to make sandwiches: Need U_gf GF and U_reg Reg in total. M_gf/M_reg are already made.
        cost_make_gf = max(0, u_gf - m_gf)
        cost_make_reg = max(0, u_reg - m_reg)

        # Cost to put sandwiches on tray: Need U_gf GF and U_reg Reg on trays. OT_gf/OT_reg are already on trays.
        # This counts sandwiches that are either in the kitchen or need to be made, then put on a tray.
        cost_put_ontray_gf = max(0, u_gf - ot_gf)
        cost_put_ontray_reg = max(0, u_reg - ot_reg)

        # Cost to move trays: Estimate based on locations needing a tray.
        cost_move_tray = locations_needing_tray_move

        # Cost to serve: One action per unserved child.
        cost_serve = u_gf + u_reg

        # Total heuristic is the sum of estimated actions for each stage
        total_cost = cost_make_gf + cost_make_reg + cost_put_ontray_gf + cost_put_ontray_reg + cost_move_tray + cost_serve

        return total_cost

