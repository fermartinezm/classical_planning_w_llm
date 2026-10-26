from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It counts the number of unserved children and the available suitable sandwiches
    in different stages of preparation/location (on tray at child's location,
    on tray elsewhere, in kitchen, makeable). It assigns the cheapest available
    sandwich source to each unserved child, prioritizing allergic children for
    gluten-free sandwiches.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - Actions have a cost of 1.
    - Ingredients and sandwich objects, once used for making, are consumed in the heuristic calculation.
    - Trays are assumed to be available and movable as needed, with a cost of 1
      for moving a tray to a child's location if it's not already there.
      (Simplified: the cost is associated with the sandwich's state, not explicit tray availability).
    - The heuristic does not consider the limited number of trays explicitly for
      concurrent actions or kitchen loading, only for delivery location.

    # Heuristic Initialization
    - Identify all children who are initially waiting and their allergy status.
    - Store which children are allergic vs non-allergic.
    - Store the waiting place for each child.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who are currently 'waiting' but not 'served'.
    2. Separate these unserved children into allergic and non-allergic groups.
    3. Count the available resources (sandwiches and ingredients) in the current state:
       - Count GF and regular sandwiches already on trays at any unserved child's waiting place.
       - Count GF and regular sandwiches on trays elsewhere (not at a child's waiting place).
       - Count GF and regular sandwiches in the kitchen.
       - Count available 'notexist' sandwich objects.
       - Count available bread, content, GF bread, and GF content in the kitchen.
    4. Calculate the maximum number of makeable GF and regular sandwiches based on available ingredients and 'notexist' objects.
    5. Initialize the total heuristic cost to 0.
    6. Greedily assign sandwich sources to unserved children, adding the estimated action cost for each child:
       - Prioritize allergic children (need GF). Assign them the cheapest available GF source first (on tray at location, then on tray elsewhere, then kitchen, then makeable GF). Add the corresponding cost (1, 2, 3, or 4).
       - Then, assign sandwich sources to non-allergic children (can take GF or regular). Assign them the cheapest available source first (remaining GF on tray at location, then regular on tray at location, then remaining GF on tray elsewhere, then regular on tray elsewhere, then remaining GF in kitchen, then regular in kitchen, then remaining makeable GF, then makeable regular). Add the corresponding cost (1, 2, 3, or 4).
    7. The total heuristic value is the sum of costs for all unserved children.
    8. If any children remain unserved after exhausting all potential sandwich sources, the state is likely unsolvable; return infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children.
        """
        self.goals = task.goals # Goal conditions
        # Initial state facts contain the initial waiting children and their properties
        initial_state = task.initial_state

        # Identify all children who are initially waiting and their properties
        self.initial_waiting_children = set()
        self.child_is_allergic = {}
        self.child_waiting_place = {}

        # Get initial waiting children and their places/allergies from initial state facts
        for fact in initial_state:
            parts = get_parts(fact)
            if parts[0] == 'waiting':
                child, place = parts[1], parts[2]
                self.initial_waiting_children.add(child)
                self.child_waiting_place[child] = place
            # Allergy status is also in initial state/static
            elif parts[0] == 'allergic_gluten':
                child = parts[1]
                self.child_is_allergic[child] = True
            elif parts[0] == 'not_allergic_gluten':
                child = parts[1]
                self.child_is_allergic[child] = False

        # Ensure all initially waiting children have an allergy status recorded
        # (should be guaranteed by valid PDDL, but add a fallback)
        for child in list(self.initial_waiting_children):
             if child not in self.child_is_allergic:
                 # Assume not allergic if status is missing
                 self.child_is_allergic[child] = False


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify unserved children
        unserved_allergic = []
        unserved_non_allergic = []
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        for child in self.initial_waiting_children:
            if child not in served_children:
                if self.child_is_allergic.get(child, False): # Default to False if status missing
                    unserved_allergic.append(child)
                else:
                    unserved_non_allergic.append(child)

        # If all initial waiting children are served, goal reached.
        if not unserved_allergic and not unserved_non_allergic:
            return 0

        # 2. Count available resources (sandwiches and ingredients)
        num_gf_ontray_at_child_loc = 0
        num_reg_ontray_at_child_loc = 0
        num_gf_ontray_elsewhere = 0
        num_reg_ontray_elsewhere = 0
        num_gf_kitchen = 0
        num_reg_kitchen = 0
        num_notexist = 0

        # Track tray locations
        tray_locations = {}
        # 'at' facts in this domain are only for trays and places
        for fact in state:
             parts = get_parts(fact)
             if len(parts) == 3 and parts[0] == 'at':
                  tray_locations[parts[1]] = parts[2]

        # Get the set of places where unserved children are waiting
        unserved_child_places = {self.child_waiting_place[child] for child in unserved_allergic + unserved_non_allergic}

        # Count sandwiches on trays
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                s_name, t_name = get_parts(fact)[1], get_parts(fact)[2]
                is_gf = "(no_gluten_sandwich {})".format(s_name) in state
                t_loc = tray_locations.get(t_name) # Get location of the tray

                if t_loc and t_loc in unserved_child_places:
                    if is_gf:
                        num_gf_ontray_at_child_loc += 1
                    else:
                        num_reg_ontray_at_child_loc += 1
                else:
                     # This includes trays in the kitchen or at places without unserved children
                     # 'elsewhere' includes kitchen for tray location purposes here.
                    if is_gf:
                        num_gf_ontray_elsewhere += 1
                    else:
                        num_reg_ontray_elsewhere += 1

        # Count sandwiches in kitchen
        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                s_name = get_parts(fact)[1]
                is_gf = "(no_gluten_sandwich {})".format(s_name) in state
                if is_gf:
                    num_gf_kitchen += 1
                else:
                    num_reg_kitchen += 1

        # Count notexist sandwich objects
        num_notexist = sum(1 for fact in state if match(fact, "notexist", "*"))

        # Count ingredients
        num_bread = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*"))
        num_content = sum(1 for fact in state if match(fact, "at_kitchen_content", "*"))
        num_gf_bread = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*") and match(fact, "no_gluten_bread", "*"))
        num_gf_content = sum(1 for fact in state if match(fact, "at_kitchen_content", "*") and match(fact, "no_gluten_content", "*"))

        # Calculate makeable sandwiches
        makeable_gf = min(num_gf_bread, num_gf_content, num_notexist)
        remaining_notexist = num_notexist - makeable_gf
        # Remaining bread/content available for regular sandwiches after using some for GF
        remaining_bread_for_reg = num_bread - makeable_gf
        remaining_content_for_reg = num_content - makeable_gf

        makeable_reg = min(remaining_bread_for_reg, remaining_content_for_reg, remaining_notexist)


        # 3. Assign sandwich sources and calculate cost
        h = 0

        # Serve allergic children first (need GF)
        needed_allergic = len(unserved_allergic)

        # Source 1: GF on tray at child loc (Cost 1: serve)
        use = min(needed_allergic, num_gf_ontray_at_child_loc)
        h += use * 1
        needed_allergic -= use
        num_gf_ontray_at_child_loc -= use

        # Source 2: GF on tray elsewhere (Cost 2: move tray + serve)
        use = min(needed_allergic, num_gf_ontray_elsewhere)
        h += use * 2
        needed_allergic -= use
        num_gf_ontray_elsewhere -= use

        # Source 3: GF in kitchen (Cost 3: put on tray + move tray + serve)
        use = min(needed_allergic, num_gf_kitchen)
        h += use * 3
        needed_allergic -= use
        num_gf_kitchen -= use

        # Source 4: Makeable GF (Cost 4: make + put on tray + move tray + serve)
        use = min(needed_allergic, makeable_gf)
        h += use * 4
        needed_allergic -= use
        makeable_gf -= use # Use up makeable GF count

        # Serve non-allergic children (can use remaining GF or regular)
        needed_non_allergic = len(unserved_non_allergic)

        # Source 1a: Remaining GF on tray at child loc (Cost 1: serve)
        use = min(needed_non_allergic, num_gf_ontray_at_child_loc)
        h += use * 1
        needed_non_allergic -= use
        num_gf_ontray_at_child_loc -= use

        # Source 1b: Regular on tray at child loc (Cost 1: serve)
        use = min(needed_non_allergic, num_reg_ontray_at_child_loc)
        h += use * 1
        needed_non_allergic -= use
        num_reg_ontray_at_child_loc -= use

        # Source 2a: Remaining GF on tray elsewhere (Cost 2: move tray + serve)
        use = min(needed_non_allergic, num_gf_ontray_elsewhere)
        h += use * 2
        needed_non_allergic -= use
        num_gf_ontray_elsewhere -= use

        # Source 2b: Regular on tray elsewhere (Cost 2: move tray + serve)
        use = min(needed_non_allergic, num_reg_ontray_elsewhere)
        h += use * 2
        needed_non_allergic -= use
        num_reg_ontray_elsewhere -= use

        # Source 3a: Remaining GF in kitchen (Cost 3: put on tray + move tray + serve)
        use = min(needed_non_allergic, num_gf_kitchen)
        h += use * 3
        needed_non_allergic -= use
        num_gf_kitchen -= use

        # Source 3b: Regular in kitchen (Cost 3: put on tray + move tray + serve)
        use = min(needed_non_allergic, num_reg_kitchen)
        h += use * 3
        needed_non_allergic -= use
        num_reg_kitchen -= use

        # Source 4a: Remaining makeable GF (Cost 4: make + put on tray + move tray + serve)
        use = min(needed_non_allergic, makeable_gf)
        h += use * 4
        needed_non_allergic -= use
        makeable_gf -= use # Use up makeable GF count

        # Source 4b: Makeable Regular (Cost 4: make + put on tray + move tray + serve)
        use = min(needed_non_allergic, makeable_reg)
        h += use * 4
        needed_non_allergic -= use
        makeable_reg -= use # Use up makeable regular count


        # If any children still need serving after exhausting all potential sandwich sources,
        # the state is likely unsolvable within this relaxed model.
        if needed_allergic > 0 or needed_non_allergic > 0:
             return float('inf')

        return h
