from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Utility functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string case defensively
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
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

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It counts the number of sandwiches that need to be made, put on trays, moved to the
    correct locations, and finally served. It considers gluten allergies and available
    resources (ingredients, sandwich slots, existing sandwiches, trays at locations).

    # Assumptions
    - Each served child consumes one sandwich.
    - Trays can hold multiple sandwiches (implicitly, as tray movement cost is counted per location deficit, not per sandwich).
    - Ingredients are consumed when making sandwiches.
    - The problem is solvable if enough ingredients and sandwich slots exist. If not, a large heuristic value is returned.

    # Heuristic Initialization
    The heuristic extracts static information from the task:
    - Which children are allergic to gluten.
    - The waiting location for each child.
    - Which bread and content portions are gluten-free.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic calculates the total estimated cost by summing up the estimated costs for four main stages: making sandwiches, putting them on trays, moving trays to children's locations, and serving the children.

    1.  **Identify Unserved Children:** Determine which children are waiting (from static facts) but not yet served (from the current state). Count the total unserved children (`N_unserved`) and those needing gluten-free sandwiches (`N_gf`, `N_nongf`). If `N_unserved` is 0, the goal is reached, and the heuristic is 0.

    2.  **Count Available Resources:**
        -   Count available sandwiches currently in the kitchen (`K_gf`, `K_nongf`) or on any tray (`Ontray_gf`, `Ontray_nongf`), categorized by gluten status. Total available anywhere are `A_gf = K_gf + Ontray_gf` and `A_nongf = K_nongf + Ontray_nongf`.
        -   Count available ingredients in the kitchen (`I_gf_bread`, `I_any_bread`, `I_gf_content`, `I_any_content`).
        -   Count available sandwich slots (`S_notexist`) representing sandwiches that can be made.

    3.  **Estimate Sandwiches to Make (`Cost_make`):**
        -   Calculate how many GF sandwiches are needed that are not available anywhere: `Needed_to_make_gf = max(0, N_gf - A_gf)`.
        -   Calculate how many non-GF sandwiches are needed that are not available anywhere, considering that non-allergic children can consume surplus available GF sandwiches: `Needed_to_make_nongf = max(0, N_nongf - (A_nongf + max(0, A_gf - N_gf)))`.
        -   Total sandwiches needed to be made: `Total_needed_to_make = Needed_to_make_gf + Needed_to_make_nongf`.
        -   Calculate the maximum number of GF and non-GF sandwiches that *can* be made based on available ingredients and slots.
        -   If `Total_needed_to_make` exceeds the maximum possible, return a large value (indicating likely unsolvability).
        -   Otherwise, `Cost_make = Total_needed_to_make`.

    4.  **Estimate Sandwiches to Put on Trays (`Cost_put`):**
        -   Sandwiches that need to be put on trays are those just made (`Cost_make`) plus those currently `at_kitchen_sandwich` that are needed and not already on trays.
        -   Calculate how many GF and non-GF sandwiches are needed from the kitchen (`Needed_from_kitchen_gf`, `Needed_from_kitchen_nongf`), after accounting for those made and those already on trays.
        -   `Cost_put = Cost_make + Needed_from_kitchen_gf + Needed_from_kitchen_nongf`.

    5.  **Estimate Tray Movements (`Cost_move`):**
        -   Identify each location where unserved children are waiting.
        -   For each such location, count the number of sandwiches needed and the number of suitable sandwiches already available on trays *at that specific location*.
        -   If the number needed exceeds the number available at that location, a tray movement is required to bring sandwiches there.
        -   `Cost_move` is the count of such locations needing delivery.

    6.  **Estimate Serve Actions (`Cost_serve`):**
        -   Each unserved child requires one `serve` action.
        -   `Cost_serve = N_unserved`.

    7.  **Total Heuristic:** The total heuristic value is the sum of `Cost_make + Cost_put + Cost_move + Cost_serve`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Store static information
        self.child_allergy = {} # {child: True if allergic, False otherwise}
        self.child_location = {} # {child: place}
        self.all_children = set()
        self.is_gluten_free_item = {} # {item (bread/content): True if GF}

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            if parts[0] == "allergic_gluten":
                child = parts[1]
                self.child_allergy[child] = True
                self.all_children.add(child)
            elif parts[0] == "not_allergic_gluten":
                child = parts[1]
                self.child_allergy[child] = False
                self.all_children.add(child)
            elif parts[0] == "waiting":
                child, place = parts[1], parts[2]
                self.child_location[child] = place
            elif parts[0] == "no_gluten_bread":
                bread = parts[1]
                self.is_gluten_free_item[bread] = True
            elif parts[0] == "no_gluten_content":
                content = parts[1]
                self.is_gluten_free_item[content] = True

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify Unserved Children
        unserved_children = set()
        unserved_gf_children = set()
        unserved_nongf_children = set()

        served_children_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        for child in self.all_children:
            # A child is "waiting" if they are in the static child_location map
            if child in self.child_location and child not in served_children_in_state:
                 unserved_children.add(child)
                 if self.child_allergy.get(child, False): # Default to False if allergy status unknown
                     unserved_gf_children.add(child)
                 else:
                     unserved_nongf_children.add(child)

        N_unserved = len(unserved_children)
        N_gf = len(unserved_gf_children)
        N_nongf = len(unserved_nongf_children)

        if N_unserved == 0:
            return 0 # Goal reached

        # 2. Count Available Resources (Sandwiches, Ingredients, Slots)
        kitchen_gf_sandwiches = 0
        kitchen_nongf_sandwiches = 0
        sandwiches_on_trays = {} # {sandwich: tray}
        trays_at_location = {} # {tray: place}
        available_bread = {} # {bread: count}
        available_content = {} # {content: count}
        available_sandwich_slots = 0

        # Helper to check if a sandwich S is GF based on state
        def is_sandwich_gf(s, current_state):
             return f"(no_gluten_sandwich {s})" in current_state

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == "at_kitchen_sandwich":
                s = parts[1]
                if is_sandwich_gf(s, state):
                    kitchen_gf_sandwiches += 1
                else:
                    kitchen_nongf_sandwiches += 1
            elif parts[0] == "ontray":
                s, t = parts[1], parts[2]
                sandwiches_on_trays[s] = t
            elif parts[0] == "at":
                 t, p = parts[1], parts[2]
                 trays_at_location[t] = p
            elif parts[0] == "at_kitchen_bread":
                 b = parts[1]
                 available_bread[b] = available_bread.get(b, 0) + 1
            elif parts[0] == "at_kitchen_content":
                 c = parts[1]
                 available_content[c] = available_content.get(c, 0) + 1
            elif parts[0] == "notexist":
                 available_sandwich_slots += 1

        # Count sandwiches on trays based on the sandwiches_on_trays map
        Ontray_gf = 0
        Ontray_nongf = 0
        for s in sandwiches_on_trays:
             if is_sandwich_gf(s, state):
                  Ontray_gf += 1
             else:
                  Ontray_nongf += 1

        A_gf = kitchen_gf_sandwiches + Ontray_gf
        A_nongf = kitchen_nongf_sandwiches + Ontray_nongf

        # Count available ingredients by type (GF/Any)
        I_gf_bread = sum(count for item, count in available_bread.items() if self.is_gluten_free_item.get(item, False))
        I_any_bread = sum(available_bread.values())
        I_gf_content = sum(count for item, count in available_content.items() if self.is_gluten_free_item.get(item, False))
        I_any_content = sum(available_content.values())

        # 3. Estimate Sandwiches to Make (Cost_make)
        Needed_to_make_gf = max(0, N_gf - A_gf)
        Needed_to_make_nongf = max(0, N_nongf - (A_nongf + max(0, A_gf - N_gf))) # Non-allergic can use surplus GF
        Total_needed_to_make = Needed_to_make_gf + Needed_to_make_nongf

        # Calculate max possible sandwiches to make
        # Max GF: requires GF bread, GF content, and a slot
        Max_make_gf = min(I_gf_bread, I_gf_content, available_sandwich_slots)
        # Max non-GF: requires any bread, any content, and a slot, after using ingredients/slots for GF
        remaining_bread_for_any = I_any_bread + I_gf_bread - Max_make_gf
        remaining_content_for_any = I_any_content + I_gf_content - Max_make_gf
        remaining_slots_for_any = available_sandwich_slots - Max_make_gf
        Max_make_any = min(remaining_bread_for_any, remaining_content_for_any, remaining_slots_for_any)
        Max_make_total = Max_make_gf + Max_make_any

        if Total_needed_to_make > Max_make_total:
             # Unsolvable from this state based on ingredients/slots
             return 1000000 # Large number for infinity

        Cost_make = Total_needed_to_make

        # 4. Estimate Sandwiches to Put on Trays (Cost_put)
        # These are the ones just made, plus those at_kitchen_sandwich that are needed.
        # Sandwiches needed from kitchen are those required that aren't made and aren't already on trays.
        Needed_gf_not_ontray = max(0, N_gf - Ontray_gf)
        Needed_nongf_not_ontray = max(0, N_nongf - (Ontray_nongf + max(0, Ontray_gf - N_gf)))

        Needed_from_kitchen_gf = min(kitchen_gf_sandwiches, max(0, Needed_gf_not_ontray - Cost_make)) # Use Cost_make here as total made
        # Non-allergic can use surplus GF from kitchen first
        Needed_from_kitchen_nongf = min(kitchen_nongf_sandwiches, max(0, Needed_nongf_not_ontray - (Cost_make - Needed_to_make_gf) - max(0, Needed_from_kitchen_gf - (Needed_gf_not_ontray - Cost_make))))

        Cost_put = Cost_make + Needed_from_kitchen_gf + Needed_from_kitchen_nongf

        # 5. Estimate Tray Movements (Cost_move)
        Cost_move = 0
        locations_with_unserved = {self.child_location[c] for c in unserved_children}

        for place in locations_with_unserved:
            needed_at_P_gf = len([c for c in unserved_gf_children if self.child_location[c] == place])
            needed_at_P_nongf = len([c for c in unserved_nongf_children if self.child_location[c] == place])
            total_needed_at_P = needed_at_P_gf + needed_at_P_nongf

            # Count suitable sandwiches on trays *at this specific place P*
            available_at_P_gf = 0
            available_at_P_nongf = 0
            trays_at_this_place = {t for t, p in trays_at_location.items() if p == place}

            for s, t in sandwiches_on_trays.items():
                 if t in trays_at_this_place:
                      if is_sandwich_gf(s, state):
                           available_at_P_gf += 1
                      else:
                           available_at_P_nongf += 1

            total_available_at_P = available_at_P_gf + available_at_P_nongf

            if total_needed_at_P > total_available_at_P:
                 Cost_move += 1 # Need at least one tray movement to this location

        # 6. Estimate Serve Actions (Cost_serve)
        Cost_serve = N_unserved

        # Total Heuristic
        total_cost = Cost_make + Cost_put + Cost_move + Cost_serve

        return total_cost
