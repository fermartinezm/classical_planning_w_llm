from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle cases like '(predicate)' or '(predicate arg)'
    if fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    return [] # Should not happen with valid PDDL facts

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
    This heuristic estimates the number of actions required to serve all waiting
    children. It sums the estimated costs for making necessary sandwiches,
    putting them on trays, moving trays to children's locations, and finally
    serving the children.

    # Assumptions
    - Each waiting child needs exactly one sandwich.
    - Gluten-allergic children require gluten-free sandwiches.
    - Non-allergic children can accept any sandwich (regular or gluten-free).
    - Ingredients (bread, content) and 'notexist' sandwich objects are available
      in the kitchen if needed to make sandwiches.
    - Trays have unlimited capacity for sandwiches.
    - Any tray can be used.
    - Action costs are uniform (cost 1).

    # Heuristic Initialization
    - Identify the set of children who are initially waiting. These are the
      children who must be served to reach the goal.
    - Store the allergy status (allergic_gluten or not_allergic_gluten) for
      all children mentioned in the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic is calculated by summing up the estimated costs for four main stages:

    1.  **Cost to Serve:**
        - Count the number of children who were initially waiting but are not
          yet marked as 'served' in the current state. This is the number of
          'serve' actions required. Add this count to the total heuristic.

    2.  **Cost to Make Sandwiches:**
        - Determine the number of gluten-free and regular sandwiches needed
          for the children identified in step 1, based on their allergy status.
        - Count the number of gluten-free and regular sandwiches already
          available (either 'at_kitchen_sandwich' or 'ontray').
        - Calculate the deficit of gluten-free sandwiches needed. This is the
          minimum number of 'make_sandwich_no_gluten' actions required.
        - Calculate the deficit of regular sandwiches needed, considering that
          any surplus gluten-free sandwiches can satisfy regular needs. This is
          the minimum number of 'make_sandwich' actions required.
        - Sum the counts of gluten-free and regular sandwiches to make. Add this
          sum to the total heuristic.

    3.  **Cost to Put Sandwiches on Trays:**
        - All sandwiches that need to be served must eventually be 'ontray'.
        - Count the number of sandwiches currently 'ontray'.
        - The number of sandwiches that still need to be put on trays is the
          total number of children to serve minus the number of sandwiches
          already on trays (minimum 0).
        - This count represents the minimum number of 'put_on_tray' actions needed.
          Add this count to the total heuristic.

    4.  **Cost to Move Trays:**
        - Identify the distinct locations where the children identified in step 1
          are currently waiting. These locations will need a tray.
        - Count the number of trays currently located at these waiting places.
        - The minimum number of tray movements required to get trays to the
          necessary waiting locations is the number of distinct waiting locations
          minus the number of trays already at those locations (minimum 0). Add
          this count to the total heuristic.
        - Additionally, if sandwiches need to be put on trays (cost from step 3 > 0)
          and there is no tray currently at the 'kitchen', one extra tray movement
          is needed to bring a tray to the kitchen. Add 1 to the heuristic in this case.

    The total heuristic value is the sum of the costs from steps 1, 2, 3, and 4.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal children and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Identify children who are initially waiting (these are the goal children)
        self.initial_waiting_children = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "waiting", "*", "*")
        }

        # Store allergy status from static facts
        self.allergic_children_static = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "allergic_gluten", "*")
        }
        self.not_allergic_children_static = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "not_allergic_gluten", "*")
        }

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # 1. Cost to Serve
        # Identify children who were initially waiting but are not yet served
        current_served_children = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "served", "*")
        }
        children_to_serve = {
            c for c in self.initial_waiting_children
            if c not in current_served_children
        }
        N_children_to_serve = len(children_to_serve)

        # If all goal children are served, the heuristic is 0
        if N_children_to_serve == 0:
            return 0

        total_cost += N_children_to_serve # Each child needs one serve action

        # 2. Cost to Make Sandwiches
        # Count needed sandwiches by type
        gf_needed = sum(1 for c in children_to_serve if c in self.allergic_children_static)
        reg_needed = N_children_to_serve - gf_needed # Non-allergic children need regular

        # Count available sandwiches by type (anywhere - kitchen or on tray)
        all_sandwiches_facts = {
            fact for fact in state
            if match(fact, "at_kitchen_sandwich", "*") or match(fact, "ontray", "*", "*")
        }
        # Need to extract sandwich name correctly from facts like '(ontray sandw1 tray1)'
        available_sandwich_names = {
             get_parts(fact)[1] for fact in all_sandwiches_facts
        }

        gf_available_facts = {
            fact for fact in state # Check all state facts for no_gluten_sandwich
            if match(fact, "no_gluten_sandwich", "*") and get_parts(fact)[1] in available_sandwich_names
        }
        gf_available = len(gf_available_facts)
        reg_available = len(available_sandwich_names) - gf_available

        # Calculate sandwiches to make
        gf_to_make = max(0, gf_needed - gf_available)
        # Remaining GF sandwiches can satisfy regular needs
        remaining_gf_avail = max(0, gf_available - gf_needed)
        reg_to_make = max(0, reg_needed - (reg_available + remaining_gf_avail))

        total_cost += gf_to_make + reg_to_make # Cost to make deficit sandwiches

        # 3. Cost to Put Sandwiches on Trays
        # We need N_children_to_serve sandwiches to be on trays eventually
        ontray_facts = {fact for fact in state if match(fact, "ontray", "*", "*")}
        N_ontray = len(ontray_facts)
        # Number of sandwiches that still need to be put on trays
        needed_to_put_on_tray = max(0, N_children_to_serve - N_ontray)
        total_cost += needed_to_put_on_tray # Cost for put_on_tray actions

        # 4. Cost to Move Trays
        # Identify distinct locations of children who need serving and are currently waiting
        current_waiting_facts = {
             fact for fact in state
             if match(fact, "waiting", "*", "*") and get_parts(fact)[1] in children_to_serve
        }
        waiting_locations_for_needed_children = {
            get_parts(fact)[2] for fact in current_waiting_facts
        }
        N_waiting_locations = len(waiting_locations_for_needed_children)

        # Count trays currently at these waiting locations
        trays_at_places_facts = {fact for fact in state if match(fact, "at", "*", "*")}
        N_trays_at_waiting_locations = sum(
            1 for fact in trays_at_places_facts
            if get_parts(fact)[2] in waiting_locations_for_needed_children
        )

        # Cost to move trays to waiting locations
        total_cost += max(0, N_waiting_locations - N_trays_at_waiting_locations)

        # Cost to get a tray to the kitchen if needed for put_on_tray
        needs_put_on_tray_at_kitchen = needed_to_put_on_tray > 0
        has_tray_at_kitchen = any(match(fact, "at", "*", "kitchen") for fact in trays_at_places_facts)
        if needs_put_on_tray_at_kitchen and not has_tray_at_kitchen:
             total_cost += 1 # Add cost for one move action to bring a tray to kitchen

        return total_cost
