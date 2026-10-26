from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to get parts of a PDDL fact string
def get_parts(fact):
    """Splits a PDDL fact string into its predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

# Helper function to match a fact against a pattern
def match(fact, *args):
    """Checks if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions required to reach a goal state
    (all children served) by summing up estimated costs for key steps:
    making sandwiches, putting them on trays, moving trays to children's
    locations, and serving the children.

    Summary:
    The heuristic counts the number of unserved children, the number of
    sandwiches that need to be made (considering gluten requirements and
    existing sandwiches), the number of sandwiches in the kitchen that
    need to be put on trays, and the number of tray movements needed
    to get trays to locations where unserved children are waiting. Each
    counted item corresponds to an estimated action cost of 1.

    Assumptions:
    - Assumes sufficient bread and content portions are available in the
      kitchen to make any required sandwiches.
    - Assumes trays are available to be moved if needed.
    - Assumes each relevant action (make, put_on_tray, move_tray, serve)
      has a cost of 1.
    - Does not explicitly model resource contention beyond counting available
      trays for moves.
    - It is non-admissible.

    Heuristic Initialization:
    The constructor extracts static information from the task definition,
    including which children are allergic/not allergic, where each child
    is waiting, the set of all children, the set of all trays, and the
    set of all places. This information is stored for efficient access
    during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the set of all children and the set of children who have
       already been served in the current state.
    2. Determine the set of unserved children and count the total number
       of unserved children (`N_unserved`). If this is 0, the goal is
       reached, and the heuristic is 0.
    3. Separate unserved children into allergic (`N_allergic_unserved`)
       and non-allergic (`N_non_allergic_unserved`).
    4. Count the number of sandwiches currently in the kitchen (`S_kitchen`)
       and on trays (`S_ontray`).
    5. Count the number of gluten-free sandwiches available (in kitchen or
       on trays) (`S_gf_available_total`) and non-gluten-free sandwiches
       (`S_nongf_available_total`).
    6. Calculate the minimum number of new gluten-free sandwiches that must
       be made (`M_gf`) to satisfy the needs of unserved allergic children,
       given the available gluten-free sandwiches.
    7. Calculate the minimum number of new 'any' sandwiches (can be non-GF
       or surplus GF) that must be made (`M_any`) to satisfy the needs of
       unserved non-allergic children, given the available non-GF sandwiches
       and any surplus GF sandwiches not needed by allergic children.
    8. The total number of sandwiches to make (`M_total_make`) is `M_gf + M_any`.
       This contributes `M_total_make` to the heuristic cost (for `make_sandwich` actions).
    9. Sandwiches that are currently in the kitchen (`S_kitchen`) and the
       sandwiches that are newly made (`M_total_make`) must be put on trays.
       This contributes `S_kitchen + M_total_make` to the heuristic cost
       (for `put_on_tray` actions).
    10. Identify the places where unserved children are waiting (`Places_with_unserved`).
    11. Identify the places that currently have a tray (`P_with_tray`).
    12. Determine the places that need a tray (`P_needing_tray = Places_with_unserved - P_with_tray`).
    13. Identify all trays and the trays that are currently located at a
        place where unserved children are waiting (`Trays_at_needed_place`).
    14. Determine the trays that are not currently at a needed place
        (`T_not_at_needed_place`).
    15. The number of tray movements needed is estimated as the minimum of
        the number of places needing a tray and the number of trays available
        to move (`min(|P_needing_tray|, |T_not_at_needed_place|)`). This
        contributes this minimum value to the heuristic cost (for `move_tray` actions).
    16. Each unserved child requires a `serve` action. This contributes
        `N_unserved` to the heuristic cost.
    17. The total heuristic value is the sum of the costs from steps 8, 9, 15, and 16.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.allergic_children = set()
        self.not_allergic_children = set()
        self.waiting_at = {}
        self.all_children = set()
        self.all_trays = set()
        self.all_places = set()

        # Extract static information
        for fact in static_facts:
            if match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.allergic_children.add(child)
                self.all_children.add(child)
            elif match(fact, "not_allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.not_allergic_children.add(child)
                self.all_children.add(child)
            elif match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1:3]
                self.waiting_at[child] = place
                self.all_children.add(child)
                self.all_places.add(place)
            # We don't strictly need no_gluten_bread/content in the heuristic calculation
            # as we assume ingredients are available if needed.

        # Infer trays and places from initial state 'at' facts
        # Assumes all trays are initially placed somewhere
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 tray, place = get_parts(fact)[1:3]
                 self.all_trays.add(tray)
                 self.all_places.add(place)

        # Ensure all children mentioned in goals are in all_children set
        for goal in self.goals:
             if match(goal, "served", "*"):
                 child = get_parts(goal)[1]
                 self.all_children.add(child)


    def __call__(self, node):
        state = node.state

        # Parse current state facts
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        at_kitchen_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        ontray_sandwiches_set = {get_parts(fact)[1] for fact in state if match(fact, "ontray", "*", "*")}
        # sandwich_on_tray = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "ontray", "*", "*")} # Not directly used in this heuristic
        tray_at_place = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*")}
        no_gluten_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}
        # notexist_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "notexist", "*")} # Not directly used in this heuristic

        # 1. Identify unserved children
        unserved_children = self.all_children - served_children
        n_unserved = len(unserved_children)

        # Goal reached
        if n_unserved == 0:
            return 0

        # 2. Count unserved children by allergy type
        n_allergic_unserved = len(unserved_children & self.allergic_children)
        n_non_allergic_unserved = len(unserved_children & self.not_allergic_children)

        # 3. Count available sandwiches by location and type
        s_kitchen = len(at_kitchen_sandwiches)
        s_ontray = len(ontray_sandwiches_set)
        # s_available_total = s_kitchen + s_ontray # Not directly used in this calculation method

        s_gf_available_total = len(no_gluten_sandwiches & (at_kitchen_sandwiches | ontray_sandwiches_set))
        s_nongf_available_total = len(((at_kitchen_sandwiches | ontray_sandwiches_set) - no_gluten_sandwiches))

        # 4. Calculate sandwiches to make (considering GF needs and available sandwiches)
        m_gf = max(0, n_allergic_unserved - s_gf_available_total)
        # Sandwiches needed for non-allergic children, considering surplus GF
        needed_any_after_gf_surplus = max(0, n_non_allergic_unserved - max(0, s_gf_available_total - n_allergic_unserved))
        m_any = max(0, needed_any_after_gf_surplus - s_nongf_available_total)
        m_total_make = m_gf + m_any

        # Cost component: make_sandwich actions
        cost_make = m_total_make

        # Cost component: put_on_tray actions
        # Sandwiches currently in kitchen + sandwiches that will be made in kitchen
        cost_put_on_tray = s_kitchen + m_total_make

        # Cost component: move_tray actions
        # Identify places needing trays
        places_with_unserved = {self.waiting_at[c] for c in unserved_children}
        places_with_tray = set(tray_at_place.values())
        places_needing_tray = places_with_unserved - places_with_tray

        # Identify trays available to move
        trays_at_needed_place = {t for t, p in tray_at_place.items() if p in places_with_unserved}
        trays_not_at_needed_place = self.all_trays - trays_at_needed_place

        cost_move_tray = min(len(places_needing_tray), len(trays_not_at_needed_place))

        # Cost component: serve_sandwich actions
        cost_serve = n_unserved

        # Total heuristic is the sum of estimated action costs
        total_cost = cost_make + cost_put_on_tray + cost_move_tray + cost_serve

        return total_cost
