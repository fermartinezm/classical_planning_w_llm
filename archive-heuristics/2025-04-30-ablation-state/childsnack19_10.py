from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(part == arg or arg == "*" for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting
    children. It counts the number of children not yet served, the number of
    sandwiches that still need to be made, the number of sandwiches that need
    to be put on trays, and the number of locations with waiting children that
    still need a tray delivered.

    # Assumptions
    - Each waiting child requires one sandwich.
    - Sandwiches must be made in the kitchen, put on a tray in the kitchen,
      the tray moved to the child's location, and then the sandwich served.
    - A tray can hold multiple sandwiches.
    - Resources (bread, content, notexist sandwich objects, trays) are
      sufficient for a solvable problem instance (i.e., we don't explicitly
      check resource counts beyond needing 'notexist' objects for making).
    - Moving a tray between any two locations costs 1 action.
    - The heuristic does not explicitly track gluten-free requirements beyond
      the initial count of children needing service. It assumes that if enough
      total sandwiches are made and delivered, the correct types will be handled
      by the planning process.

    # Heuristic Initialization
    - Extracts the set of goal predicates (all children served).
    - Extracts static facts (allergy information for children, although this
      specific heuristic implementation does not directly use the allergy info
      in its calculation counts, it's good practice to extract it).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic is calculated as the sum of estimated costs for four main stages:
    making sandwiches, putting sandwiches on trays, moving trays to children's
    locations, and serving children.

    1.  **Count Children Not Served:**
        - Identify the set of children who are goals (`(served ?c)`) but are not
          present in the current state's `(served ?c)` facts.
        - The number of such children (`N_not_served`) is a lower bound on the
          number of `serve_sandwich` actions required.

    2.  **Count Sandwiches to Put on Trays:**
        - To serve `N_not_served` children, we need `N_not_served` sandwiches
          to eventually be on trays.
        - Count the total number of sandwiches currently on trays anywhere
          (`N_ontray`).
        - The number of sandwiches that need to transition from `at_kitchen_sandwich`
          to `ontray` is `put_on_tray_count = max(0, N_not_served - N_ontray)`.
        - This contributes `put_on_tray_count` to the heuristic.

    3.  **Count Sandwiches to Make:**
        - These are the sandwiches counted in `put_on_tray_count` that are not
          *already* `at_kitchen_sandwich`.
        - Count the number of sandwiches currently `at_kitchen_sandwich`
          (`N_sandwich_kitchen`).
        - The number of sandwiches that need to transition from `notexist` to
          `at_kitchen_sandwich` is `make_count = max(0, cost_put_on_tray - N_sandwich_kitchen)`.
        - This contributes `make_count` to the heuristic.

    4.  **Count Trays to Move:**
        - Identify all locations where children are waiting (and not yet served).
        - Identify all locations where trays are currently present.
        - The number of locations with waiting children that do not currently
          have a tray is `move_tray_count`.
        - This contributes `move_tray_count` to the heuristic.

    The total heuristic value is the sum: `make_count + put_on_tray_count + move_tray_count + N_not_served`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals  # Set of goal facts (e.g., frozenset({'(served child1)', ...}))
        self.static_facts = task.static # Set of static facts (e.g., frozenset({'(allergic_gluten child4)', ...}))

        # Extract allergy information from static facts (for potential future use or understanding)
        self.is_allergic = {}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "allergic_gluten":
                    self.is_allergic[parts[1]] = True
                elif parts[0] == "not_allergic_gluten":
                     self.is_allergic[parts[1]] = False


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state # Current state as a frozenset of fact strings

        # --- Step 1: Count Children Not Served ---
        # Goal facts are like '(served child1)'
        served_children_goals = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}
        served_children_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # Children who are goals but not yet served
        children_not_served = served_children_goals - served_children_state
        N_not_served = len(children_not_served)

        # If all children are served, we are in a goal state, heuristic is 0.
        if N_not_served == 0:
            return 0

        # --- Extract relevant state information for other steps ---
        waiting_children_by_location = {} # {location: [child1, child2, ...]}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "waiting":
                child, location = parts[1], parts[2]
                # Only consider waiting children who are not yet served
                if child in children_not_served:
                    waiting_children_by_location.setdefault(location, []).append(child)

        # Count sandwiches on trays (anywhere)
        sandwiches_on_trays = {parts[1] for fact in state if match(fact, "ontray", "*", "*")}
        N_ontray = len(sandwiches_on_trays)

        # Count sandwiches at kitchen (not on trays)
        N_sandwich_kitchen = sum(1 for fact in state if match(fact, "at_kitchen_sandwich", "*"))

        # Count trays at locations
        trays_at_location = {parts[1]: parts[2] for fact in state if match(fact, "at", "*", "*") and parts[1].startswith("tray")}
        locations_with_trays = set(trays_at_location.values())

        # --- Heuristic Calculation Components ---

        # Component D: Serving children.
        # Number of children not yet served. Each needs one serve action.
        cost_serve = N_not_served

        # Component B: Putting sandwiches on trays.
        # Total sandwiches needed on trays eventually is N_not_served.
        # Number of sandwiches that need to transition from at_kitchen_sandwich to ontray.
        put_on_tray_count = max(0, N_not_served - N_ontray)
        cost_put_on_tray = put_on_tray_count

        # Component A: Making sandwiches.
        # These are the sandwiches that need to be put on trays (put_on_tray_count)
        # that are not already at the kitchen (N_sandwich_kitchen).
        make_count = max(0, cost_put_on_tray - N_sandwich_kitchen)
        cost_make = make_count

        # Component C: Moving trays.
        # Locations with waiting children that don't currently have a tray.
        locations_with_waiting = set(waiting_children_by_location.keys())
        move_tray_count = sum(1 for location in locations_with_waiting if location not in locations_with_trays)
        cost_move_tray = move_tray_count

        # --- Total Heuristic ---
        total_heuristic = cost_make + cost_put_on_tray + cost_move_tray + cost_serve

        return total_heuristic
