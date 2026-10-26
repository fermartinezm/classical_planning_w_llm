# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Helper functions (assuming they are available or defined elsewhere)
# If not available, uncomment and include them:
from fnmatch import fnmatch
from collections import defaultdict

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Assuming Heuristic base class is available in the environment like this:
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
#     def __call__(self, node):
#         raise NotImplementedError


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It counts the number of unserved children (representing the final 'serve' action
    for each), the number of sandwiches that need to be made and put on trays
    to satisfy the demand at different locations, and the number of tray movements
    needed to get trays with sandwiches to the locations where children are waiting.

    # Assumptions
    - Each child requires exactly one sandwich.
    - All children's waiting locations and allergy statuses are static.
    - Trays can hold multiple sandwiches. The heuristic does not consider tray capacity limits.
    - Sufficient bread, content, and sandwich objects exist in the initial state
      to make all necessary sandwiches (the heuristic does not check for resource
      exhaustion leading to unsolvability).
    - The 'kitchen' is a special place where sandwiches are made and trays start.

    # Heuristic Initialization
    - Extracts static information about children: their names, allergy status,
      and waiting locations.
    - Extracts static information about gluten-free bread and content types.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children from static facts and determine which are currently unserved
       by checking the state for the `(served ?c)` predicate.
    2. If all children are served, the heuristic value is 0.
    3. Initialize the total heuristic cost. Add the number of unserved children; this
       accounts for the final `serve` action needed for each.
    4. For each place where unserved children are waiting, count the number of
       gluten-free and regular sandwiches needed (`Needed_GF_P`, `Needed_Reg_P`).
    5. For each place, count the number of suitable sandwiches already available
       on trays currently located at that place (`Available_GF_on_trays_at_P`,
       `Available_Reg_on_trays_at_P`). Also, identify all places that currently have a tray.
       Sandwiches on trays *in the kitchen* are counted as kitchen availability, not place availability.
    6. For each place `P` (excluding 'kitchen'), calculate the number of sandwiches of each type that
       still need to be brought from the kitchen: `Sandwiches_to_bring_to_P_GF = max(0, Needed_GF_P - Available_GF_on_trays_at_P)`
       and `Sandwiches_to_bring_to_P_Reg = max(0, Needed_Reg_P - Available_Reg_on_trays_at_P)`.
    7. Sum these counts across all places (excluding 'kitchen') to get the total number of GF and Regular
       sandwiches that need to be brought from the kitchen (`Total_Sandwiches_to_bring_GF`,
       `Total_Sandwiches_to_bring_Reg`).
    8. Count the number of GF and Regular sandwiches that are currently in the kitchen
       (either `at_kitchen_sandwich` or `ontray` on a tray `at kitchen`).
    9. Calculate the number of sandwiches that still need to be made in the kitchen:
       `To_Make_GF = max(0, Total_Sandwiches_to_bring_GF - Available_GF_kitchen_made)`
       and `To_Make_Reg = max(0, Total_Sandwiches_to_bring_Reg - Available_Reg_kitchen_made)`.
    10. Add `To_Make_GF + To_Make_Reg` to the total cost (each represents one `make_sandwich` action).
    11. Add `Total_Sandwiches_to_bring_GF + Total_Sandwiches_to_bring_Reg` to the total cost
        (each represents one `put_on_tray` action for a sandwich that needs to be delivered from the kitchen).
    12. Identify places `P` (excluding 'kitchen') that need sandwiches brought to them
        (`Sandwiches_to_bring_to_P_GF + Sandwiches_to_bring_to_P_Reg > 0`) and do not
        currently have a tray located `at P`. Add 1 to the total cost for each such place
        (each represents one `move_tray` action from the kitchen to that place).
    13. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.goals = task.goals

        # Extract static information about children and their needs/locations
        self.children = set()
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.child_location = {}
        for fact in task.static:
            if match(fact, "allergic_gluten", "?c"):
                child = get_parts(fact)[1]
                self.children.add(child)
                self.allergic_children.add(child)
            elif match(fact, "not_allergic_gluten", "?c"):
                child = get_parts(fact)[1]
                self.children.add(child)
                self.not_allergic_children.add(child)
            elif match(fact, "waiting", "?c", "?p"):
                child, place = get_parts(fact)[1:]
                self.child_location[child] = place

        # Extract static information about gluten-free ingredients
        self.gf_bread_types = set()
        self.gf_content_types = set()
        for fact in task.static:
            if match(fact, "no_gluten_bread", "?b"):
                self.gf_bread_types.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_content", "?c"):
                self.gf_content_types.add(get_parts(fact)[1])

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # 1. Identify unserved children
        unserved_children = {c for c in self.children if f'(served {c})' not in state}

        # 2. If all children are served, goal reached
        if not unserved_children:
            return 0

        # 3. Initialize cost and add cost for serve actions
        total_cost = len(unserved_children)

        # Data structures to track needs and availability per place
        needed_gf_at_place = defaultdict(int)
        needed_reg_at_place = defaultdict(int)
        available_gf_on_trays_at_place = defaultdict(int)
        available_reg_on_trays_at_place = defaultdict(int)
        places_with_tray = set()

        # Map sandwiches to their gluten status and trays to their locations
        sandwich_is_gf = {}
        tray_location = {}

        for fact in state:
            if match(fact, "no_gluten_sandwich", "?s"):
                sandwich_is_gf[get_parts(fact)[1]] = True
            elif match(fact, "at", "?t", "?p"):
                tray_location[get_parts(fact)[1]] = get_parts(fact)[2]
                places_with_tray.add(get_parts(fact)[2])

        # 4. Count needed sandwiches per place
        for child in unserved_children:
            place = self.child_location[child]
            if child in self.allergic_children:
                needed_gf_at_place[place] += 1
            else:
                needed_reg_at_place[place] += 1

        # 5. Count available sandwiches on trays at places (excluding kitchen)
        for fact in state:
            if match(fact, "ontray", "?s", "?t"):
                sandwich = get_parts(fact)[1]
                tray = get_parts(fact)[2]
                place = tray_location.get(tray) # Get tray location from pre-calculated map
                if place and place != "kitchen": # Only count if tray is 'at' a place other than kitchen
                    if sandwich_is_gf.get(sandwich, False):
                        available_gf_on_trays_at_place[place] += 1
                    else:
                        available_reg_on_trays_at_place[place] += 1

        # 6. Calculate sandwiches to bring from kitchen for each place (excluding kitchen itself)
        sandwiches_to_bring_to_place_gf = defaultdict(int)
        sandwiches_to_bring_to_place_reg = defaultdict(int)
        places_needing_delivery = set()

        # Consider all places where children are waiting OR where trays currently are (excluding kitchen)
        # These are the potential destinations for deliveries
        all_relevant_places = set(needed_gf_at_place.keys()) | set(needed_reg_at_place.keys()) | places_with_tray
        all_relevant_places.discard("kitchen") # Kitchen is source, not destination for delivery

        for place in all_relevant_places:
            needed_gf = needed_gf_at_place[place]
            needed_reg = needed_reg_at_place[place]
            available_gf = available_gf_on_trays_at_place[place]
            available_reg = available_reg_on_trays_at_place[place]

            to_bring_gf = max(0, needed_gf - available_gf)
            to_bring_reg = max(0, needed_reg - available_reg)

            sandwiches_to_bring_to_place_gf[place] = to_bring_gf
            sandwiches_to_bring_to_place_reg[place] = to_bring_reg

            if to_bring_gf > 0 or to_bring_reg > 0:
                places_needing_delivery.add(place)

        # 7. Calculate total sandwiches to bring from kitchen
        total_to_bring_gf = sum(sandwiches_to_bring_to_place_gf.values())
        total_to_bring_reg = sum(sandwiches_to_bring_to_place_reg.values())

        # 8. Count available sandwiches/ingredients/objects in kitchen
        available_gf_kitchen_made = 0 # at_kitchen_sandwich GF or ontray GF on tray at kitchen
        available_reg_kitchen_made = 0 # at_kitchen_sandwich Reg or ontray Reg on tray at kitchen
        gf_bread_kitchen = 0
        gf_content_kitchen = 0
        reg_bread_kitchen = 0
        reg_content_kitchen = 0
        available_sandwich_objects = 0

        # Scan state for kitchen resources
        for fact in state:
            if match(fact, "at_kitchen_sandwich", "?s"):
                 sandwich = get_parts(fact)[1]
                 # Check if GF predicate exists for this sandwich in the state
                 if f'(no_gluten_sandwich {sandwich})' in state:
                     available_gf_kitchen_made += 1
                 else:
                     available_reg_kitchen_made += 1
            elif match(fact, "ontray", "?s", "?t"):
                 sandwich = get_parts(fact)[1]
                 tray = get_parts(fact)[2]
                 # Check if the tray is in the kitchen
                 if tray_location.get(tray) == "kitchen":
                     # Check if GF predicate exists for this sandwich in the state
                     if f'(no_gluten_sandwich {sandwich})' in state:
                         available_gf_kitchen_made += 1
                     else:
                         available_reg_kitchen_made += 1
            elif match(fact, "at_kitchen_bread", "?b"):
                bread_name = get_parts(fact)[1]
                if bread_name in self.gf_bread_types:
                    gf_bread_kitchen += 1
                else:
                    reg_bread_kitchen += 1
            elif match(fact, "at_kitchen_content", "?c"):
                content_name = get_parts(fact)[1]
                if content_name in self.gf_content_types:
                    gf_content_kitchen += 1
                else:
                    reg_content_kitchen += 1
            elif match(fact, "notexist", "?s"):
                available_sandwich_objects += 1

        # available_trays_at_kitchen is not directly used in this heuristic's cost calculation,
        # only places_with_tray matters for move cost. Ingredient counts are also not used
        # in cost, only implicitly assumed sufficient for 'to_make' calculation.

        # 9. Calculate sandwiches to make
        to_make_gf = max(0, total_to_bring_gf - available_gf_kitchen_made)
        to_make_reg = max(0, total_to_bring_reg - available_reg_kitchen_made)

        # Add cost for make actions
        total_cost += to_make_gf + to_make_reg

        # 10. Add cost for put on tray actions
        # Each sandwich that needs to be brought from the kitchen must be put on a tray.
        # This includes newly made sandwiches and existing kitchen sandwiches/ontray sandwiches
        # that are needed for delivery.
        total_cost += total_to_bring_gf + total_to_bring_reg

        # 11. Add cost for move tray actions
        places_needing_tray_move_count = 0
        for place in places_needing_delivery:
            # If the place needs delivery and does not currently have a tray
            if place not in places_with_tray:
                 places_needing_tray_move_count += 1

        total_cost += places_needing_tray_move_count

        # Return the estimated cost
        return total_cost
