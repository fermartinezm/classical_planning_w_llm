from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Utility function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Utility function to check if a PDDL fact matches a given pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args has wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It breaks down the problem into stages: making sandwiches, putting them on trays,
    moving trays to children's locations, and serving the children. It counts the
    remaining actions needed in each stage, taking into account sandwiches already
    prepared or delivered, and shared actions like tray movements.

    # Assumptions
    - Resources (bread, content, notexist sandwich objects) are sufficient to make
      all required sandwiches eventually.
    - Trays are sufficient to carry needed sandwiches.
    - The 'kitchen' is the initial location for trays and where sandwiches are made.
    - A sandwich is standard unless explicitly marked with `no_gluten_sandwich`.

    # Heuristic Initialization
    - Identify which children are allergic to gluten from static facts.
    - Identify the set of children that need to be served from the goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic estimates the total cost as the sum of costs for:
    1.  **Making sandwiches:** Estimate actions to make sandwiches that are not yet made,
        needed to satisfy the demand from unserved children, considering gluten requirements.
    2.  **Putting sandwiches on trays:** Estimate actions to move sandwiches from the
        kitchen (where they are made) onto trays, for sandwiches that are made but not yet on trays,
        needed to satisfy demand not covered by sandwiches already on trays.
    3.  **Moving trays:** Estimate actions to move trays from their current location
        to places where unserved children are waiting and need sandwiches delivered,
        specifically counting a move if a location needs delivery and has no tray present.
    4.  **Serving children:** Count the number of unserved children, as each requires one
        serve action.

    Detailed Steps:
    - Identify all unserved children and their locations and allergy status.
    - If no children are unserved, the heuristic is 0.
    - Count the total number of standard and gluten-free sandwiches required (equal to the number of unserved non-allergic and allergic children, respectively).
    - Count standard and gluten-free sandwiches currently on trays.
    - Count standard and gluten-free sandwiches currently in the kitchen (made but not on trays).
    - Calculate the number of standard/GF sandwiches that still need to be made (total needed - on trays - in kitchen). Cost = (std_needed_make + gf_needed_make) * 1.
    - Calculate the number of standard/GF sandwiches in the kitchen that are needed on trays (min(in_kitchen, needed_total - on_trays)). Cost = (std_needed_put + gf_needed_put) * 1.
    - Identify places where unserved children are waiting (excluding kitchen).
    - For each such place P, determine if it needs delivery: Check if the number of unserved standard/GF children at P exceeds the number of standard/GF sandwiches already on trays *at P*.
    - Count the number of places that need delivery AND currently have no tray. This count is the tray move cost.
    - The serving cost is simply the total number of unserved children.
    - Sum all calculated costs.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Map child name to allergy status (True if allergic, False otherwise)
        self.child_allergy = {}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == "allergic_gluten":
                self.child_allergy[parts[1]] = True
            elif parts[0] == "not_allergic_gluten":
                self.child_allergy[parts[1]] = False

        # Identify children who need to be served based on goal facts
        self.children_to_serve = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "served":
                self.children_to_serve.add(parts[1])

        # Assume 'kitchen' is a constant place name
        # Find the kitchen constant from static facts or domain definition if needed,
        # but the domain file shows it as a constant 'kitchen'.
        self.kitchen_place = "kitchen"

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # 1. Identify Unserved Children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = self.children_to_serve - served_children

        if not unserved_children:
            return 0  # Goal state reached

        # 2. Categorize Unserved Children and their locations
        unserved_info = {} # child -> {'location': P, 'allergic': True/False}
        unserved_at_place_std = {} # place -> count
        unserved_at_place_gf = {}  # place -> count
        places_with_unserved = set()

        # Need to iterate through state to find waiting children
        waiting_children_in_state = {} # child -> place
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "waiting":
                 waiting_children_in_state[parts[1]] = parts[2]

        for child in unserved_children:
            if child in waiting_children_in_state:
                place = waiting_children_in_state[child]
                is_allergic = self.child_allergy.get(child, False) # Default to not allergic if info missing
                unserved_info[child] = {'location': place, 'allergic': is_allergic}
                places_with_unserved.add(place)

                if is_allergic:
                    unserved_at_place_gf[place] = unserved_at_place_gf.get(place, 0) + 1
                else:
                    unserved_at_place_std[place] = unserved_at_place_std.get(place, 0) + 1
            # Note: If a child is unserved but not waiting in the current state,
            # this heuristic doesn't account for getting them back to a waiting place.
            # Assuming children stay waiting until served.

        num_unserved_std = sum(1 for child, info in unserved_info.items() if not info['allergic'])
        num_unserved_gf = sum(1 for child, info in unserved_info.items() if info['allergic'])

        # 3. Count Available Sandwiches
        sandwiches_on_trays_list = [] # list of (sandwich_name, tray_name)
        sandwiches_in_kitchen_list = [] # list of sandwich_name
        gf_sandwiches_set = set() # set of sandwich_name that are GF
        tray_locations = {} # tray_name -> place_name

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "ontray":
                sandwiches_on_trays_list.append((parts[1], parts[2]))
            elif parts[0] == "at_kitchen_sandwich":
                sandwiches_in_kitchen_list.append(parts[1])
            elif parts[0] == "no_gluten_sandwich":
                gf_sandwiches_set.add(parts[1])
            elif parts[0] == "at":
                tray_locations[parts[1]] = parts[2]

        num_on_tray_std = sum(1 for s, t in sandwiches_on_trays_list if s not in gf_sandwiches_set)
        num_on_tray_gf = sum(1 for s, t in sandwiches_on_trays_list if s in gf_sandwiches_set)
        num_in_kitchen_std = sum(1 for s in sandwiches_in_kitchen_list if s not in gf_sandwiches_set)
        num_in_kitchen_gf = sum(1 for s in sandwiches_in_kitchen_list if s in gf_sandwiches_set)

        # 4. Calculate Sandwich Preparation Cost (Make + Put)
        # Sandwiches needed that are not yet made (neither on tray nor in kitchen)
        needed_make_std = max(0, num_unserved_std - num_on_tray_std - num_in_kitchen_std)
        needed_make_gf = max(0, num_unserved_gf - num_on_tray_gf - num_in_kitchen_gf)
        cost_make = needed_make_std + needed_make_gf # 1 action per make

        # Sandwiches in kitchen that are needed on trays (to meet demand not covered by trays)
        needed_put_std = min(num_in_kitchen_std, max(0, num_unserved_std - num_on_tray_std))
        needed_put_gf = min(num_in_kitchen_gf, max(0, num_unserved_gf - num_on_tray_gf))
        cost_put = needed_put_std + needed_put_gf # 1 action per put

        # 5. Calculate Tray Move Cost
        cost_move = 0
        sandwiches_on_trays_at_place = {} # place -> list of (sandwich_name, is_gf)

        for s, t in sandwiches_on_trays_list:
            if t in tray_locations:
                place = tray_locations[t]
                is_gf = s in gf_sandwiches_set
                sandwiches_on_trays_at_place.setdefault(place, []).append((s, is_gf))

        for place in places_with_unserved:
            if place == self.kitchen_place:
                 # Children waiting in the kitchen. Trays start here.
                 # A tray move is not needed just to get a tray *to* the kitchen.
                 continue

            # Count sandwiches on trays already at this place
            on_trays_at_P_std = sum(1 for s, is_gf in sandwiches_on_trays_at_place.get(place, []) if not is_gf)
            on_trays_at_P_gf = sum(1 for s, is_gf in sandwiches_on_trays_at_place.get(place, []) if is_gf)

            # Check if this place needs more sandwiches delivered by tray
            needs_delivery = (unserved_at_place_std.get(place, 0) > on_trays_at_P_std) or \
                             (unserved_at_place_gf.get(place, 0) > on_trays_at_P_gf)

            # Check if any tray is currently at this place
            has_tray = any(loc == place for loc in tray_locations.values())

            if needs_delivery and not has_tray:
                cost_move += 1 # Need to move a tray here

        # 6. Calculate Serving Cost
        cost_serve = num_unserved_std + num_unserved_gf # 1 action per serve

        # 7. Total Heuristic
        total_cost = cost_make + cost_put + cost_move + cost_serve

        return total_cost
