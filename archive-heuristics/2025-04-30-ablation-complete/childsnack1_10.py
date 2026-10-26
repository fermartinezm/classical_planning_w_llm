from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has the expected format before processing
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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
    This heuristic estimates the number of actions required to serve all unserved
    children. It calculates the cost for each unserved child individually based
    on the current state of their required sandwich and the location of trays.
    The cost for a child is estimated by counting the number of major steps
    (make, put on tray, move tray, serve) that are still needed for a suitable
    sandwich to reach the child's location and be served.

    # Assumptions
    - Each child requires exactly one sandwich of the correct type (gluten-free
      for allergic children, regular otherwise).
    - Ingredients (bread, content) and sandwich objects are assumed to be
      available in the kitchen when needed to make a sandwich, even if the
      state representation doesn't explicitly track counts beyond existence.
      This simplifies the heuristic and makes it non-admissible.
    - Trays are assumed to be available when needed for putting sandwiches on
      or moving.
    - The heuristic sums the costs for each unserved child independently,
      ignoring potential resource contention (e.g., multiple children needing
      the same sandwich type, limited trays).

    # Heuristic Initialization
    - Extracts static information about child allergies and waiting locations.
    - Identifies the set of children that need to be served based on the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1. Identify all children who are in the goal state but are not yet marked as `served` in the current state. These are the unserved children.
    2. If there are no unserved children, the state is a goal state, and the heuristic is 0.
    3. Initialize the total heuristic cost to 0.
    4. For each unserved child:
        a. Determine the type of sandwich required (gluten-free or regular) based on their allergy status (extracted during initialization).
        b. Initialize the cost for this child to 0.
        c. Add 1 to the child's cost for the final `serve` action.
        d. Check if there is *any* suitable sandwich (matching the required type) currently on *any* tray.
            i. If yes: Check if any of these trays carrying a suitable sandwich is already located at the child's waiting place.
                - If yes: No additional cost for tray location or sandwich placement is added for this child.
                - If no: Add 1 to the child's cost (for the `move_tray` action needed to bring a suitable tray to the child).
            ii. If no suitable sandwich is on *any* tray: Add 1 to the child's cost (for the `put_on_tray` action).
                - Check if there is *any* suitable sandwich currently in the kitchen.
                    - If yes: No additional cost for making the sandwich is added for this child.
                    - If no: Add 1 to the child's cost (for the `make_sandwich` action). (Note: This step assumes ingredients and sandwich objects are available).
        e. Add the calculated child's cost to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.

        @param task: The planning task object containing initial state, goals, operators, and static facts.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract static information
        self.child_allergies = {} # Map child name to boolean (True if allergic)
        self.child_waiting_places = {} # Map child name to place name
        # Static gluten-free ingredients are not directly used in this heuristic's cost calculation,
        # but are useful for understanding the domain. We store them anyway.
        self.static_no_gluten_bread = set() # Set of gluten-free bread names
        self.static_no_gluten_content = set() # Set of gluten-free content names


        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "allergic_gluten" and len(parts) == 2:
                self.child_allergies[parts[1]] = True
            elif predicate == "not_allergic_gluten" and len(parts) == 2:
                self.child_allergies[parts[1]] = False
            elif predicate == "waiting" and len(parts) == 3:
                self.child_waiting_places[parts[1]] = parts[2]
            elif predicate == "no_gluten_bread" and len(parts) == 2:
                self.static_no_gluten_bread.add(parts[1])
            elif predicate == "no_gluten_content" and len(parts) == 2:
                self.static_no_gluten_content.add(parts[1])

        # Identify children that need serving from goals
        # Assuming goals are of the form (served childX)
        self.goal_children = {c for goal in self.goals for pred, c in [get_parts(goal)] if pred == 'served' and len(get_parts(goal)) == 2}


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        @param node: The search node containing the current state.
        @return: The estimated number of actions to reach a goal state.
        """
        state = node.state

        # Identify children already served in the current state
        served_children_in_state = {c for fact in state for pred, c in [get_parts(fact)] if pred == 'served' and len(get_parts(fact)) == 2}

        # Identify children who still need to be served
        unserved_children = self.goal_children - served_children_in_state

        # If all goal children are served, the heuristic is 0
        if not unserved_children:
            return 0

        # Parse relevant facts from the current state into efficient data structures
        ontray_facts = set() # Stores (sandwich, tray) tuples
        tray_locations = {} # Map tray -> place
        kitchen_sandwiches = set() # Set of sandwiches in kitchen
        gluten_free_sandwiches_in_state = set() # Set of sandwiches currently marked as gluten-free

        # Note: We don't strictly need available_bread/content/objects for this heuristic's
        # cost calculation logic due to the simplifying assumption, but parsing them
        # doesn't hurt and could be useful for debugging or future heuristic versions.
        # available_bread_kitchen = set()
        # available_content_kitchen = set()
        # available_sandwich_objects = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "ontray" and len(parts) == 3:
                ontray_facts.add((parts[1], parts[2]))
            elif predicate == "at" and len(parts) == 3 and parts[1].startswith("tray"):
                 tray_locations[parts[1]] = parts[2]
            elif predicate == "at_kitchen_sandwich" and len(parts) == 2:
                kitchen_sandwiches.add(parts[1])
            elif predicate == "no_gluten_sandwich" and len(parts) == 2:
                gluten_free_sandwiches_in_state.add(parts[1])
            # elif predicate == "at_kitchen_bread" and len(parts) == 2:
            #      available_bread_kitchen.add(parts[1])
            # elif predicate == "at_kitchen_content" and len(parts) == 2:
            #      available_content_kitchen.add(parts[1])
            # elif predicate == "notexist" and len(parts) == 2:
            #      available_sandwich_objects.add(parts[1])


        total_heuristic = 0

        # Calculate cost for each unserved child
        for child in unserved_children:
            child_cost = 0
            # Get waiting place from static info (it doesn't change)
            child_place = self.child_waiting_places.get(child)
            if child_place is None:
                 # This child is a goal child but has no waiting location in static facts.
                 # This indicates a potentially malformed problem instance or a child
                 # that doesn't need serving via the standard process (e.g., already served
                 # in initial state but not marked in goals, or some other mechanism).
                 # For this heuristic, we assume all goal children have a waiting place.
                 # If not found, we skip this child or assign a high cost. Skipping is safer
                 # if the problem definition guarantees waiting facts for goal children.
                 # Let's assume valid problems where goal children have waiting facts.
                 # If child_place is None, it means the child is in goals but not in static waiting facts.
                 # This heuristic is based on serving waiting children, so we skip.
                 continue


            # Determine required sandwich type
            needs_gluten_free = self.child_allergies.get(child, False) # Default to not allergic if allergy status unknown

            # Cost for the final 'serve' action
            child_cost += 1

            # Check if a suitable sandwich is on a tray at the child's location
            suitable_ontray_at_location = False
            suitable_ontray_anywhere = False

            # Iterate through sandwiches on trays
            for s, t in ontray_facts:
                # Check if the sandwich type matches the child's requirement
                is_gluten_free_sandwich = s in gluten_free_sandwiches_in_state
                if is_gluten_free_sandwich == needs_gluten_free:
                    # Found a suitable sandwich on a tray
                    suitable_ontray_anywhere = True
                    # Check if the tray is at the child's location
                    if tray_locations.get(t) == child_place:
                        suitable_ontray_at_location = True
                        break # Found a suitable sandwich on a tray at the correct location

            if not suitable_ontray_at_location:
                # If no suitable sandwich is on a tray at the location,
                # we need either to move a tray or put a sandwich on a tray first.
                if suitable_ontray_anywhere:
                    # A suitable sandwich is on a tray, but not at the location.
                    # Need to move the tray.
                    child_cost += 1 # Cost for 'move_tray'
                else:
                    # No suitable sandwich is on any tray.
                    # Need to put a sandwich on a tray.
                    child_cost += 1 # Cost for 'put_on_tray'

                    # Check if a suitable sandwich is in the kitchen
                    suitable_kitchen_sandwich_exists = False
                    # Iterate through sandwiches in the kitchen
                    for s in kitchen_sandwiches:
                        # Check if the sandwich type matches the child's requirement
                        is_gluten_free_sandwich = s in gluten_free_sandwiches_in_state
                        if is_gluten_free_sandwich == needs_gluten_free:
                            suitable_kitchen_sandwich_exists = True
                            break # Found a suitable sandwich in the kitchen

                    if not suitable_kitchen_sandwich_exists:
                        # No suitable sandwich is in the kitchen.
                        # Need to make a sandwich.
                        child_cost += 1 # Cost for 'make_sandwich'
                        # Note: This heuristic does NOT check if ingredients/object exist.
                        # It assumes they are available if needed.

            total_heuristic += child_cost

        return total_heuristic
