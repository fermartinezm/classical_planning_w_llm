from fnmatch import fnmatch
# Assuming heuristic_base.py is available in the execution environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not running in the planner environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # print("Warning: Could not import Heuristic base class. Using a dummy class.")
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic not implemented")


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is treated as a string and handle potential whitespace
    fact_str = str(fact).strip()
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
         # Return empty list or handle error for malformed facts
         return []
    # Split by whitespace, filtering out empty strings from multiple spaces
    return [part for part in fact_str[1:-1].split() if part]

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if not parts or len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting
    children by summing up the estimated minimum number of 'make', 'put', 'move',
    and 'serve' actions needed based on the current state. It counts the total
    requirements across all unserved children and available resources.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - Allergic children require gluten-free sandwiches; non-allergic children can
      accept any sandwich (but the heuristic counts distinct needs for GF and Reg).
    - Ingredients (bread, content) and 'notexist' sandwich objects are assumed
      sufficient if *any* of the required type are present in the initial state
      (or static facts) to make the needed sandwiches. The heuristic does not
      explicitly check for ingredient counts beyond the initial state's types.
    - Trays are assumed to have sufficient capacity to hold multiple sandwiches
      needed for a single location.
    - Trays are assumed to be available in the kitchen when needed for 'put_on_tray'.
    - The cost of moving a tray is 1, regardless of distance.

    # Heuristic Initialization
    - Extract the set of children that need to be served from the goal conditions.
    - Extract static facts, specifically allergy information for children.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is calculated as the sum of estimated minimum actions for
    four main stages for all unserved children:

    1.  **Serving:** Each unserved child requires one 'serve' action. Count the
        number of children present in the initial goal set who are not currently
        marked as 'served' in the state. Add this count to the heuristic.

    2.  **Tray Movement:** For each location where unserved children are waiting,
        if there is no tray currently at that location, at least one 'move_tray'
        action is needed to bring a tray there. Count the number of such locations
        and add this count to the heuristic.

    3.  **Putting on Tray:** Count the total number of suitable sandwiches needed
        for all unserved children (gluten-free for allergic, regular for non-allergic).
        Subtract the number of suitable sandwiches already present on *any* tray
        (categorized by type: GF/Reg). The remainder must be put on trays. Add this
        number to the heuristic (representing 'put_on_tray' actions).

    4.  **Making Sandwiches:** Count the number of sandwiches that need to be put
        on trays (from step 3). Subtract the number of suitable sandwiches already
        available in the kitchen (categorized by type: GF/Reg). The remainder must
        be newly made. Add this number to the heuristic (representing 'make_sandwich'
        actions).

    The total heuristic is the sum of the counts from steps 1, 2, 3, and 4.
    If the set of unserved children is empty, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        # Set of children that must be served in the goal state.
        self.goal_children = {
            get_parts(goal)[1]
            for goal in task.goals
            if match(goal, "served", "*")
        }

        # Map children to their allergy status using static facts.
        self.child_allergy = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts and len(parts) == 2:
                predicate, child_name = parts[0], parts[1]
                if predicate == "allergic_gluten":
                    self.child_allergy[child_name] = "allergic"
                elif predicate == "not_allergic_gluten":
                    self.child_allergy[child_name] = "not_allergic"

        # Store static facts for quick lookup if needed later (e.g., initial ingredient availability)
        self.static_facts = task.static


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # 1. Identify unserved children
        served_children = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "served", "*")
        }
        unserved_children = self.goal_children - served_children

        # If all goal children are served, the heuristic is 0.
        if not unserved_children:
            return 0

        heuristic_cost = 0

        # Cost for serving each unserved child
        # Each unserved child needs 1 'serve' action.
        heuristic_cost += len(unserved_children)

        # Map unserved children to their current waiting locations
        unserved_children_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and len(parts) == 3 and parts[0] == "waiting":
                child_name, place_name = parts[1], parts[2]
                if child_name in unserved_children:
                    unserved_children_locations[child_name] = place_name

        # 2. Identify locations needing trays
        # Locations where unserved children are waiting.
        locations_with_unserved = set(unserved_children_locations.values())

        # Locations where trays are currently present.
        trays_at_locations = {
            get_parts(fact)[2]
            for fact in state
            if match(fact, "at", "*", "*") and get_parts(fact)[1].startswith("tray") # Ensure it's a tray
        }

        # Locations with unserved children that do not currently have a tray.
        locations_needing_tray = {
            loc for loc in locations_with_unserved if loc not in trays_at_locations
        }
        # Minimum 1 'move_tray' action needed for each such location.
        heuristic_cost += len(locations_needing_tray)

        # 3. Count sandwiches needed on trays
        # Total number of GF and Regular sandwiches required for all unserved children.
        total_gf_needed = sum(1 for child in unserved_children if self.child_allergy.get(child) == "allergic")
        total_reg_needed = sum(1 for child in unserved_children if self.child_allergy.get(child) == "not_allergic")

        # Count suitable sandwiches already present on *any* tray.
        ontray_sandwiches = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "ontray", "*", "*")
        }
        ontray_gf_sandwiches = {
            s for s in ontray_sandwiches if f"(no_gluten_sandwich {s})" in state
        }
        ontray_reg_sandwiches = {
             s for s in ontray_sandwiches if f"(no_gluten_sandwich {s})" not in state
        }

        avail_ontray_gf = len(ontray_gf_sandwiches)
        avail_ontray_reg = len(ontray_reg_sandwiches)

        # Number of GF sandwiches that are needed but not yet on trays.
        need_put_on_tray_gf = max(0, total_gf_needed - avail_ontray_gf)
        # Number of Regular sandwiches that are needed but not yet on trays.
        need_put_on_tray_reg = max(0, total_reg_needed - avail_ontray_reg)

        # Minimum 1 'put_on_tray' action needed for each sandwich that must be moved from kitchen to tray.
        heuristic_cost += need_put_on_tray_gf + need_put_on_tray_reg

        # 4. Count sandwiches to make
        # Count suitable sandwiches already available in the kitchen.
        kitchen_sandwiches = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "at_kitchen_sandwich", "*")
        }
        kitchen_gf_sandwiches = {
            s for s in kitchen_sandwiches if f"(no_gluten_sandwich {s})" in state
        }
        kitchen_reg_sandwiches = {
            s for s in kitchen_sandwiches if f"(no_gluten_sandwich {s})" not in state
        }

        avail_kitchen_gf = len(kitchen_gf_sandwiches)
        avail_kitchen_reg = len(kitchen_reg_sandwiches)

        # Number of GF sandwiches that are needed on trays (need_put_on_tray_gf)
        # but are not yet available in the kitchen. These must be made.
        need_make_gf = max(0, need_put_on_tray_gf - avail_kitchen_gf)
        # Number of Regular sandwiches that are needed on trays (need_put_on_tray_reg)
        # but are not yet available in the kitchen. These must be made.
        need_make_reg = max(0, need_put_on_tray_reg - avail_kitchen_reg)

        # Minimum 1 'make_sandwich' action needed for each sandwich that must be created.
        heuristic_cost += need_make_gf + need_make_reg

        return heuristic_cost
