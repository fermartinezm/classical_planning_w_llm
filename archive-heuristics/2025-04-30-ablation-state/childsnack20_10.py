from fnmatch import fnmatch
from collections import defaultdict

# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for standalone testing
# Remove this section when integrating into the planner environment
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
    def __call__(self, node):
        pass # Abstract method
# End of dummy class section

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting
    children. It counts the necessary 'serve' actions and then estimates the
    minimum number of 'make', 'put_on_tray', and 'move_tray' actions needed
    to get suitable sandwiches onto trays at the children's locations. It
    prioritizes using sandwiches that are closer to being delivered (already
    on trays at the location) before considering those in the kitchen or
    needing to be made. Tray movement cost is estimated per location that
    needs a delivery.

    # Assumptions
    - All unserved children are waiting at some place specified by a `(waiting ?c ?p)` fact.
    - Ingredients (bread, content) only exist in the kitchen.
    - Sandwiches, once made, are initially in the kitchen.
    - Trays can hold enough sandwiches for the children at a location.
    - There are enough tray objects to fulfill simultaneous delivery needs
      to different locations (heuristic counts moves per location, not per tray object instance).
    - The problem instance is solvable with the given resources (ingredients,
      sandwich objects, trays). If not, the heuristic returns infinity.

    # Heuristic Initialization
    - Extract static facts: which children are allergic or not allergic to gluten.
      This information is needed to determine sandwich suitability. Also collects
      all child names mentioned in static facts and goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Count Unserved Children:** Iterate through all known child names. For each child `C`, check if `(served C)` is true in the current state. The number of children for whom `(served C)` is false is the number of unserved children (`num_unserved`). Each unserved child requires one 'serve' action. Add `num_unserved` to the total heuristic. If `num_unserved` is 0, the goal is reached, return 0.

    2.  **Identify Needs and Available Sandwiches at Locations:**
        - Determine which unserved children are allergic/non-allergic using the pre-calculated static facts.
        - For each place `P`, count the number of unserved allergic children waiting there (`needed_gf_at[P]`) and unserved non-allergic children waiting there (`needed_reg_at[P]`). Collect all unique place names encountered.
        - Identify all sandwiches currently on trays and their locations. For each place `P`, count the number of gluten-free sandwiches (`gf_on_tray_at[P]`) and regular sandwiches (`reg_on_tray_at[P]`) that are currently on trays located at `P`.

    3.  **Calculate Children Served by Existing Trays:** For each place `P`, calculate how many children waiting at `P` can be served by the sandwiches already on trays at `P`. A GF sandwich can serve an allergic or non-allergic child. A regular sandwich can only serve a non-allergic child. Prioritize serving allergic children with GF sandwiches. Sum this count over all places (`num_served_by_stage3`).

    4.  **Calculate Children Needing Delivery:** The number of children who still need a sandwich delivered to their location is `num_unserved - num_served_by_stage3`. Let this be `num_needing_earlier`. If `num_needing_earlier` is 0, the heuristic is just `num_unserved` (only serve actions needed).

    5.  **Count Suitable Sandwiches at Earlier Stages:**
        - A sandwich is "potentially useful" if it is suitable for *any* unserved child (GF is suitable if any unserved child exists; Regular is suitable if any unserved non-allergic child exists).
        - Count potentially useful sandwiches in the kitchen not on trays (`s_at_kitchen_sandwich`).
        - Count potentially useful sandwiches that can be made (`s_makeable`) based on available ingredients (bread, content) and `notexist` sandwich objects in the kitchen. This calculation considers resource limits and suitability.

    6.  **Count Tray Movements Needed:** Identify all distinct places `P` where children are waiting and the number of children needing delivery at `P` is greater than 0 (i.e., `needed_gf_at[P] + needed_reg_at[P] > served_at_P`). Each such place requires at least one tray movement to deliver sandwiches. Count the number of distinct places that still need a delivery (`num_places_needing_delivery`). Add `num_places_needing_delivery` to the total heuristic.

    7.  **Count Make and Put Actions Needed:** The `num_needing_earlier` children need sandwiches sourced from the kitchen (either existing not on tray, or made).
        - Sandwiches from kitchen stock (not on trays) used to cover needs require a 'put_on_tray' action. Count how many needs are covered this way (`covered_by_kitchen_stock`).
        - Sandwiches that must be made to cover remaining needs require a 'make' action and a 'put_on_tray' action. Count how many needs are covered this way (`covered_by_making`).
        - Add `covered_by_making` to the total heuristic (for 'make' actions).
        - Add `covered_by_kitchen_stock + covered_by_making` to the total heuristic (for 'put_on_tray' actions).

    8.  **Sum Costs:** The total heuristic is the sum of costs from steps 1, 6, and 7.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts about child allergies.
        """
        super().__init__(task)
        self.allergic_children = {
            get_parts(fact)[1] for fact in self.static if match(fact, "allergic_gluten", "*")
        }
        self.non_allergic_children = {
            get_parts(fact)[1] for fact in self.static if match(fact, "not_allergic_gluten", "*")
        }
        # Collect all child names from static facts and goal facts
        self.all_children = self.allergic_children | self.non_allergic_children
        for goal in self.goals:
             if match(goal, "served", "*"):
                 self.all_children.add(get_parts(goal)[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # --- Step 1: Count Unserved Children (Cost: 1 per serve action) ---
        unserved_children_names = {
            c for c in self.all_children if f"(served {c})" not in state
        }
        num_unserved = len(unserved_children_names)
        total_cost += num_unserved

        if num_unserved == 0:
            return 0 # Goal reached

        # Determine if any unserved children need GF or Reg sandwiches
        has_allergic_unserved = any(c in self.allergic_children for c in unserved_children_names)
        has_non_allergic_unserved = any(c in self.non_allergic_children for c in unserved_children_names)

        # --- Step 2 & 3: Identify Needs and Available Sandwiches at Locations ---
        needed_gf_at = defaultdict(int)
        needed_reg_at = defaultdict(int)
        gf_on_tray_at = defaultdict(int)
        reg_on_tray_at = defaultdict(int)
        all_places = set()

        # Find where unserved children are waiting
        for fact in state:
            if match(fact, "waiting", "*", "*"):
                _, child_name, place_name = get_parts(fact)
                all_places.add(place_name)
                if child_name in unserved_children_names:
                    if child_name in self.allergic_children:
                        needed_gf_at[place_name] += 1
                    elif child_name in self.non_allergic_children:
                        needed_reg_at[place_name] += 1

        # Find sandwiches on trays and their locations
        sandwiches_on_trays = {} # Map sandwich -> tray
        tray_locations = {} # Map tray -> place
        sandwich_is_gf = {} # Map sandwich -> bool

        for fact in state:
            if match(fact, "ontray", "*", "*"):
                _, s_name, t_name = get_parts(fact)
                sandwiches_on_trays[s_name] = t_name
            elif match(fact, "at", "*", "*"):
                 _, obj_name, p_name = get_parts(fact)
                 # Assume 'at' applies to trays and kitchen constant
                 if obj_name.startswith("tray"): # Simple check based on naming convention
                     tray_locations[obj_name] = p_name
                     all_places.add(p_name)
                 elif obj_name == "kitchen":
                     all_places.add(obj_name) # Add kitchen as a place

            elif match(fact, "no_gluten_sandwich", "*"):
                 _, s_name = get_parts(fact)
                 sandwich_is_gf[s_name] = True

        # Count sandwiches on trays at each place
        for s_name, t_name in sandwiches_on_trays.items():
            if t_name in tray_locations:
                p_name = tray_locations[t_name]
                if sandwich_is_gf.get(s_name, False):
                    gf_on_tray_at[p_name] += 1
                else:
                    reg_on_tray_at[p_name] += 1

        # Calculate children served by current trays at their location
        num_served_by_stage3 = 0
        for place_name in all_places:
            gf_needed = needed_gf_at.get(place_name, 0)
            reg_needed = needed_reg_at.get(place_name, 0)
            gf_available = gf_on_tray_at.get(place_name, 0)
            reg_available = reg_on_tray_at.get(place_name, 0)

            served_at_place = min(gf_needed, gf_available)
            remaining_gf_available = gf_available - served_at_place
            served_at_place += min(reg_needed, reg_available + remaining_gf_available)
            num_served_by_stage3 += served_at_place

        # --- Step 4: Calculate Children Needing Delivery ---
        num_needing_earlier = num_unserved - num_served_by_stage3

        if num_needing_earlier == 0:
             # All children can be served by sandwiches already at their location
             # Total cost is just the serve actions (already added)
             return total_cost

        # --- Step 5: Count Suitable Sandwiches at Earlier Stages ---

        # Helper to check if a sandwich type is potentially useful for any unserved child
        def is_potentially_useful(is_gf):
             if is_gf:
                 # GF is useful if any unserved child exists (allergic or non-allergic)
                 return has_allergic_unserved or has_non_allergic_unserved
             else:
                 # Reg is useful only if any unserved non-allergic child exists
                 return has_non_allergic_unserved

        # Count sandwiches in kitchen (not on trays)
        s_at_kitchen_sandwich = 0
        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                _, s_name = get_parts(fact)
                if s_name not in sandwiches_on_trays: # Not already on a tray
                    is_gf = sandwich_is_gf.get(s_name, False)
                    if is_potentially_useful(is_gf):
                         s_at_kitchen_sandwich += 1

        # Count makeable sandwiches
        num_gf_bread = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*") and match(fact, "no_gluten_bread", get_parts(fact)[1]))
        num_reg_bread = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*") and not match(fact, "no_gluten_bread", get_parts(fact)[1]))
        num_gf_content = sum(1 for fact in state if match(fact, "at_kitchen_content", "*") and match(fact, "no_gluten_content", get_parts(fact)[1]))
        num_reg_content = sum(1 for fact in state if match(fact, "at_kitchen_content", "*") and not match(fact, "no_gluten_content", get_parts(fact)[1]))
        num_notexist = sum(1 for fact in state if match(fact, "notexist", "*"))

        makeable_gf = min(num_gf_bread, num_gf_content, num_notexist)
        # Remaining resources after making GF
        rem_gf_bread = num_gf_bread - makeable_gf
        rem_gf_content = num_gf_content - makeable_gf
        rem_notexist = num_notexist - makeable_gf

        makeable_reg = min(num_reg_bread + max(0, rem_gf_bread), num_reg_content + max(0, rem_gf_content), max(0, rem_notexist))

        s_makeable = 0
        if has_non_allergic_unserved:
            # If non-allergic unserved exist, any sandwich (GF or Reg) is useful
            s_makeable = makeable_gf + makeable_reg
        elif has_allergic_unserved:
            # If only allergic unserved exist, only GF sandwiches are useful
            s_makeable = makeable_gf
        # else: s_makeable remains 0 (no unserved children, should have returned 0 earlier)


        # --- Step 6: Count Tray Movements Needed ---
        # Count distinct places where children need delivery
        places_needing_delivery = set()
        for place_name in all_places:
             gf_needed = needed_gf_at.get(place_name, 0)
             reg_needed = needed_reg_at.get(place_name, 0)
             gf_available = gf_on_tray_at.get(place_name, 0)
             reg_available = reg_on_tray_at.get(place_name, 0)

             served_at_place = min(gf_needed, gf_available)
             remaining_gf_available = gf_available - served_at_place
             served_at_place += min(reg_needed, reg_available + remaining_gf_available)

             if gf_needed + reg_needed > served_at_place:
                 places_needing_delivery.add(place_name)

        num_places_needing_delivery = len(places_needing_delivery)
        total_cost += num_places_needing_delivery # Cost for moving trays to these places


        # --- Step 7: Count Make and Put Actions Needed ---
        # We need to source num_needing_earlier sandwiches from kitchen stock or by making them.
        # Prioritize kitchen stock (not on tray), then making.

        # Sandwiches from kitchen (not on tray) used
        covered_by_kitchen_stock = min(num_needing_earlier, s_at_kitchen_sandwich)
        rem_needs = num_needing_earlier - covered_by_kitchen_stock

        # Sandwiches made that are used
        covered_by_making = min(rem_needs, s_makeable)
        rem_needs -= covered_by_making

        # If rem_needs > 0, it means we need more sandwiches than can be made or are in kitchen stock.
        # This suggests the problem might be unsolvable with current resources.
        # For a non-admissible heuristic, we can return a large number.
        # Let's return infinity if we can't source enough sandwiches.
        if rem_needs > 0:
             return float('inf')

        make_actions = covered_by_making
        put_actions = covered_by_kitchen_stock + covered_by_making # Both need put_on_tray

        total_cost += make_actions # Cost for make actions
        total_cost += put_actions # Cost for put_on_tray actions

        return total_cost
