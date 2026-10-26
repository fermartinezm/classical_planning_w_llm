from fnmatch import fnmatch

# Define a dummy Heuristic base class if not running in the specific environment
# This allows the code to be tested or analyzed outside the planner environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic base class not found")
        def __str__(self):
            return self.__class__.__name__


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and remove surrounding parentheses
    if isinstance(fact, str) and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    # Return empty list for malformed or non-fact strings
    return []


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the minimum number of major steps required to serve
    each unserved child, summing the estimates for all unserved children.
    The steps considered are: make sandwich, put on tray, move tray, and serve.

    # Assumptions
    - The heuristic is additive, meaning it sums the estimated costs for each
      unserved child independently, ignoring potential resource contention
      (like multiple children needing the same limited sandwich type or tray).
    - It assumes that if a 'notexist' sandwich slot is available and suitable
      ingredients are in the kitchen, a sandwich can be made (cost 1).
    - It assumes trays can be moved to any location (cost 1).
    - It assumes a tray is available at the kitchen or can be moved there
      when needed for putting a sandwich on it.

    # Heuristic Initialization
    The heuristic extracts static information from the task:
    - The waiting place for each child (`waiting ?c ?p`).
    - The allergy status of each child (`allergic_gluten ?c`).
    - The set of children that need to be served according to the goal (`served ?c`).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic iterates through each child that needs to be served according
    to the goal but is not yet marked as `served` in the current state.
    For each unserved child `C` waiting at place `P`, it estimates the minimum
    number of actions required to serve them, based on the current state:

    1.  **Cost 0 (Served):** If `(served C)` is true in the current state, the cost for this child is 0.

    2.  **Cost 1 (Ready to Serve):** If State 1 is not met, but there exists a suitable sandwich `S` (gluten-free if `C` is allergic, any otherwise) on a tray `T` (`(ontray S T)`) and that tray `T` is already at the child's waiting place `P` (`(at T P)`), the cost is 1 (the `serve_sandwich` action).

    3.  **Cost 2 (Tray Needs Moving):** If State 1 is not met, but there exists a suitable sandwich `S` on a tray `T` (`(ontray S T)`) and that tray `T` is at a different place `P'` (`(at T P')` where `P' != P`), the cost is 2 (1 for `move_tray` + 1 for `serve_sandwich`).

    4.  **Cost 3 (Sandwich in Kitchen):** If States 1 and 2 are not met, but there exists a suitable sandwich `S` in the kitchen (`(at_kitchen_sandwich S)`), the cost is 3 (1 for `put_on_tray` + 1 for `move_tray` + 1 for `serve_sandwich`). This assumes a tray is available in the kitchen or can be moved there.

    5.  **Cost 4 (Sandwich Needs Making):** If States 1, 2, and 3 are not met (no suitable sandwich exists in the state, either on a tray or in the kitchen), the cost is 4 (1 for `make_sandwich` + 1 for `put_on_tray` + 1 for `move_tray` + 1 for `serve_sandwich`). This step is only considered possible if there is at least one `(notexist S)` fact in the state and suitable ingredients (bread and content, potentially gluten-free) are available in the kitchen.

    6.  **Cost 1000 (Unservable):** If none of the above states are met (e.g., no `notexist` sandwich slots or no suitable ingredients available to make a sandwich), the child is considered unservable in the current state, and a high cost (1000) is assigned to strongly penalize this state.

    The total heuristic value is the sum of the costs calculated for each unserved child.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Waiting place for each child.
        - Allergy status for each child.
        - Children that are goals (need to be served).
        """
        # Store child waiting places: {child_name: place_name}
        self.child_waiting_place = {}
        # Store allergic children: {child_name}
        self.allergic_children = set()
        # Store children that need to be served: {child_name}
        self.goal_children = set()

        # Process static facts
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == 'waiting':
                # Fact is (waiting child place)
                if len(parts) == 3:
                    self.child_waiting_place[parts[1]] = parts[2]
            elif parts[0] == 'allergic_gluten':
                # Fact is (allergic_gluten child)
                if len(parts) == 2:
                    self.allergic_children.add(parts[1])
            # Note: no_gluten_bread/content are sometimes in static in examples,
            # but the domain defines them as predicates that can change (e.g. make_sandwich removes ingredients).
            # We will read their state from the current state in __call__.

        # Process goal facts
        for goal in task.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip malformed goals

            if parts[0] == 'served':
                # Goal is (served child)
                if len(parts) == 2:
                    self.goal_children.add(parts[1])

        # The kitchen constant is defined in the domain, but not explicitly in static facts.
        # We know it exists and is a place.
        self.kitchen_place = 'kitchen' # Assuming the constant name is 'kitchen'


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to serve all goal children.
        """
        state = node.state  # Current world state (frozenset of strings)

        # --- Extract relevant information from the current state ---
        served_children = set()
        kitchen_sandwiches = set()
        ontray_sandwiches = {} # {sandwich: tray}
        tray_locations = {} # {tray: place}
        no_gluten_sandwiches_in_state = set()
        notexist_sandwiches = set()
        kitchen_bread = set()
        kitchen_content = set()
        no_gluten_bread_in_state = set()
        no_gluten_content_in_state = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'served':
                if len(parts) == 2: served_children.add(parts[1])
            elif predicate == 'at_kitchen_sandwich':
                if len(parts) == 2: kitchen_sandwiches.add(parts[1])
            elif predicate == 'ontray':
                if len(parts) == 3: ontray_sandwiches[parts[1]] = parts[2]
            elif predicate == 'at':
                 # Check if the first argument is a tray
                 if len(parts) == 3 and parts[1].startswith('tray'): # Assuming trays are named starting with 'tray'
                     tray_locations[parts[1]] = parts[2]
            elif predicate == 'no_gluten_sandwich':
                if len(parts) == 2: no_gluten_sandwiches_in_state.add(parts[1])
            elif predicate == 'notexist':
                if len(parts) == 2: notexist_sandwiches.add(parts[1])
            elif predicate == 'at_kitchen_bread':
                 if len(parts) == 2: kitchen_bread.add(parts[1])
            elif predicate == 'at_kitchen_content':
                 if len(parts) == 2: kitchen_content.add(parts[1])
            elif predicate == 'no_gluten_bread':
                 if len(parts) == 2: no_gluten_bread_in_state.add(parts[1])
            elif predicate == 'no_gluten_content':
                 if len(parts) == 2: no_gluten_content_in_state.add(parts[1])


        total_cost = 0

        # Iterate through all children that are goals (need to be served)
        for child in self.goal_children:
            # If child is already served, no cost
            if child in served_children:
                continue

            # Child is not served, calculate cost
            # Find where the child is waiting (from static info)
            waiting_place = self.child_waiting_place.get(child)
            if waiting_place is None:
                 # This child is a goal but doesn't have a waiting place?
                 # Should not happen in valid problems, but handle defensively.
                 # Treat as unservable for now.
                 total_cost += 1000
                 continue

            # Determine if the child needs a gluten-free sandwich
            is_allergic = child in self.allergic_children

            child_cost = 1000 # Default high cost if unservable

            # --- Check states from lowest cost to highest ---

            # State 1: Suitable sandwich on tray at waiting_place (Cost 1: Serve)
            found_state_1 = False
            for s, t in ontray_sandwiches.items():
                if t in tray_locations and tray_locations[t] == waiting_place:
                    # Check if sandwich 's' is suitable for child 'child'
                    s_is_gluten_free = s in no_gluten_sandwiches_in_state
                    if (not is_allergic) or s_is_gluten_free:
                        child_cost = 1
                        found_state_1 = True
                        break # Found the best case for this child

            if found_state_1:
                total_cost += child_cost
                continue # Move to the next child

            # State 2: Suitable sandwich on tray not at waiting_place (Cost 2: Move + Serve)
            found_state_2 = False
            for s, t in ontray_sandwiches.items():
                 if t in tray_locations and tray_locations[t] != waiting_place:
                    # Check if sandwich 's' is suitable for child 'child'
                    s_is_gluten_free = s in no_gluten_sandwiches_in_state
                    if (not is_allergic) or s_is_gluten_free:
                        child_cost = 2
                        found_state_2 = True
                        break # Found the next best case

            if found_state_2:
                total_cost += child_cost
                continue # Move to the next child

            # State 3: Suitable sandwich in kitchen (Cost 3: Put + Move + Serve)
            found_state_3 = False
            for s in kitchen_sandwiches:
                 # Check if sandwich 's' is suitable for child 'child'
                 s_is_gluten_free = s in no_gluten_sandwiches_in_state
                 if (not is_allergic) or s_is_gluten_free:
                     child_cost = 3
                     found_state_3 = True
                     break # Found the next best case

            if found_state_3:
                total_cost += child_cost
                continue # Move to the next child

            # State 4: No suitable sandwich exists, but can one be made?
            # (Cost 4: Make + Put + Move + Serve)
            can_make_suitable_sandwich = False
            if len(notexist_sandwiches) > 0: # Need an available sandwich slot
                # Check if suitable ingredients are available in the kitchen
                available_gf_bread_count = len(kitchen_bread.intersection(no_gluten_bread_in_state))
                available_gf_content_count = len(kitchen_content.intersection(no_gluten_content_in_state))
                # available_reg_bread_count = len(kitchen_bread) - available_gf_bread_count # Not strictly needed, just need > 0 total
                # available_reg_content_count = len(kitchen_content) - available_gf_content_count # Not strictly needed, just need > 0 total

                if is_allergic:
                    # Need GF sandwich: requires GF bread and GF content
                    if available_gf_bread_count > 0 and available_gf_content_count > 0:
                        can_make_suitable_sandwich = True
                else:
                    # Need any sandwich: requires any bread and any content
                    if len(kitchen_bread) > 0 and len(kitchen_content) > 0:
                         can_make_suitable_sandwich = True

            if can_make_suitable_sandwich:
                 child_cost = 4

            # If child_cost is still 1000, it means no suitable sandwich exists
            # and one cannot be made with available resources.
            total_cost += child_cost

        return total_cost
