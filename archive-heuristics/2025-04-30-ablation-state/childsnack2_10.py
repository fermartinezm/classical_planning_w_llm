from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assumes fact is a string like '(predicate arg1 arg2)'
    # Added a basic check for expected format
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Return empty list or handle error as appropriate for context
         return []
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It sums up the estimated costs for the necessary actions: making sandwiches,
    putting sandwiches on trays, moving trays to children's locations, and serving
    the children. It accounts for the sequential nature of making and putting
    sandwiches on trays, and the shared cost of moving a tray to a location
    with multiple waiting children.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - Gluten-allergic children require gluten-free sandwiches. Non-allergic children
      can accept any sandwich.
    - The process for a sandwich is typically: made in kitchen -> put on tray
      in kitchen -> tray moved to child's location -> child served from tray.
    - Resource availability (bread, content, sandwich objects, trays) is assumed
      when estimating the cost of making sandwiches or using trays.
    - A single tray move can satisfy the requirement for all children waiting
      at the destination location.

    # Heuristic Initialization
    The heuristic extracts static information from the task:
    - The allergy status for each child.
    - The waiting place for each child.
    - The set of all children in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic estimates the remaining actions by summing
    costs for necessary action types based on the current state and the goal
    (serving all unserved children):

    1.  **Making Sandwiches (Cost: 1 per sandwich):**
        Count the number of gluten-free and regular sandwiches that still need
        to be made. This is calculated as the number of unserved children needing
        that type minus the number of that type already available (either in the
        kitchen or on a tray), clipped at zero. Each sandwich needing to be made
        requires a `make_sandwich` or `make_sandwich_no_gluten` action.

    2.  **Putting Sandwiches on Trays (Cost: 1 per sandwich):**
        Count the number of sandwiches currently `at_kitchen_sandwich` that are
        suitable for at least one unserved child. Each of these requires a
        `put_on_tray` action.

    3.  **Moving Trays (Cost: 1 per location):**
        Count the number of distinct locations where unserved children are
        waiting but where no tray is currently located. Each such location
        requires at least one `move_tray` action.

    4.  **Serving Children (Cost: 1 per child):**
        Count the number of children who have not yet been served. Each unserved
        child requires a `serve_sandwich` or `serve_sandwich_no_gluten` action.

    The heuristic combines these counts using the formula:
    H = 2 * (Sandwiches to Make) + (Sandwiches in Kitchen Needing Put) + (Places Needing Tray) + (Unserved Children)

    The factor of 2 for "Sandwiches to Make" accounts for both the `make_sandwich`
    action and the subsequent `put_on_tray` action that will be required for
    each newly made sandwich. Sandwiches already in the kitchen only require
    the `put_on_tray` action (cost 1).

    This formula ensures the heuristic is 0 if and only if all children are served,
    as all components become zero in that state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts."""
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are facts that do not change during planning.
        static_facts = task.static

        # Extract child allergy status and waiting places from static facts
        self.child_allergy = {}
        self.child_place = {}
        self.all_children = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'allergic_gluten':
                child = parts[1]
                self.child_allergy[child] = 'allergic_gluten'
                self.all_children.add(child)
            elif predicate == 'not_allergic_gluten':
                child = parts[1]
                self.child_allergy[child] = 'not_allergic_gluten'
                self.all_children.add(child)
            elif predicate == 'waiting':
                child = parts[1]
                place = parts[2]
                self.child_place[child] = place
                self.all_children.add(child)

    def is_suitable(self, sandwich, child, state):
        """Checks if a sandwich is suitable for a child based on allergy."""
        # Check if the sandwich is gluten-free
        is_gf_sandwich = (f'(no_gluten_sandwich {sandwich})' in state)

        # Check if the child is allergic (using pre-calculated static info)
        is_allergic_child = (self.child_allergy.get(child) == 'allergic_gluten')

        if is_allergic_child:
            # Allergic child requires a gluten-free sandwich
            return is_gf_sandwich
        else:
            # Non-allergic child can eat any sandwich
            return True

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children
        served_children = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'served'}
        unserved_children = self.all_children - served_children
        num_unserved_children = len(unserved_children)

        # If no children are unserved, the goal is reached
        if num_unserved_children == 0:
            return 0

        # 2. Identify available sandwiches and their status
        sandwiches_at_kitchen = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'at_kitchen_sandwich'}
        sandwiches_on_tray = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'ontray'}
        sandwiches_available = sandwiches_at_kitchen | sandwiches_on_tray

        # Identify gluten-free sandwiches among available ones
        gf_sandwiches_in_state = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'no_gluten_sandwich'}
        sandwiches_gf_avail = {s for s in sandwiches_available if s in gf_sandwiches_in_state}
        sandwiches_reg_avail = sandwiches_available - sandwiches_gf_avail # Assuming non-GF is regular

        # 3. Identify children needing specific sandwich types
        children_gf = {c for c in unserved_children if self.child_allergy.get(c) == 'allergic_gluten'}
        children_reg = {c for c in unserved_children if self.child_allergy.get(c) == 'not_allergic_gluten'}

        # 4. Calculate sandwiches to make (Component 1)
        num_unserved_gf = len(children_gf)
        num_unserved_reg = len(children_reg)
        num_avail_gf_sandwiches = len(sandwiches_gf_avail)
        num_avail_reg_sandwiches = len(sandwiches_reg_avail)

        sandwiches_to_make_gf = max(0, num_unserved_gf - num_avail_gf_sandwiches)
        sandwiches_to_make_reg = max(0, num_unserved_reg - num_avail_reg_sandwiches)
        sandwiches_to_make = sandwiches_to_make_gf + sandwiches_to_make_reg

        # 5. Calculate sandwiches in kitchen needing put_on_tray (Component 2)
        # Count sandwiches currently at_kitchen_sandwich that are suitable for *any* unserved child
        sandwiches_kitchen_needed = set()
        if unserved_children: # Optimization: only check if there are unserved children
            for s in sandwiches_at_kitchen:
                for c in unserved_children:
                    if self.is_suitable(s, c, state):
                        sandwiches_kitchen_needed.add(s)
                        break # This sandwich is needed by at least one child

        num_sandwiches_kitchen_needed = len(sandwiches_kitchen_needed)

        # 6. Calculate places needing tray (Component 3)
        # Assumes (at tray place) facts are the only ones using 'at' predicate for movable objects
        tray_locations = {get_parts(fact)[2] for fact in state if get_parts(fact)[0] == 'at'}
        places_with_unserved = {self.child_place[c] for c in unserved_children}
        places_needing_tray_set = places_with_unserved - tray_locations
        num_places_needing_tray = len(places_needing_tray_set)

        # 7. Calculate total heuristic value
        # H = 2 * (Make) + (Put) + (Move) + (Serve)
        heuristic_value = (2 * sandwiches_to_make) + num_sandwiches_kitchen_needed + num_places_needing_tray + num_unserved_children

        # The formula ensures the value is non-negative and 0 only at goal,
        # so no explicit check `if heuristic_value > 0 else 0` is strictly needed
        # if the logic is correct.
        return heuristic_value
