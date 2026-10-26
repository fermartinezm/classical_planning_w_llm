from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


# Helper functions (as seen in example heuristics)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully, though PDDL states are structured.
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    A domain-dependent heuristic function for the PDDL domain childsnacks.

    # Summary
    This heuristic estimates the total number of actions required to serve all unserved children.
    It calculates the minimum steps needed for each unserved child independently, based on the
    current location and type of the best available suitable sandwich. The total heuristic
    value is the sum of these independent costs.

    # Assumptions
    - Each unserved child requires exactly one suitable sandwich.
    - A suitable sandwich is a gluten-free sandwich for allergic children and any existing sandwich for non-allergic children.
    - The cost calculation for each child is performed independently of other children, ignoring potential resource conflicts (like competition for trays, ingredients, or 'notexist' sandwich objects). This relaxation makes the heuristic non-admissible but faster to compute and potentially better guiding for greedy search.
    - It is always possible to make a new sandwich if needed, provided a 'notexist' sandwich object is available. The heuristic assumes 'notexist' objects and necessary ingredients (bread and content, including gluten-free variants if required) are sufficient whenever a sandwich needs to be made. This is a simplification based on the domain structure where ingredient-gathering actions are absent.
    - Moving a tray between any two places takes a fixed cost of 1 action.
    - Putting a sandwich on a tray takes a fixed cost of 1 action, assuming the sandwich is in the kitchen and a tray is available in the kitchen.
    - Serving a sandwich takes a fixed cost of 1 action, assuming the sandwich is on a tray, the tray is at the child's waiting location, and the sandwich type is suitable for the child's allergy status.

    # Heuristic Initialization
    During initialization, the heuristic extracts static information from the planning task:
    - Identifies all children involved in the problem.
    - Determines which children are allergic to gluten.
    - Records the waiting location for each child.
    This static information is stored for efficient access during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:
    1. Initialize the total heuristic cost to 0.
    2. Identify the set of children who have already been served in the current state.
    3. Identify all existing sandwich objects in the current state (those not marked with the `notexist` predicate).
    4. Determine which of the existing sandwiches are gluten-free.
    5. Record the current location of each existing sandwich (either in the kitchen or on a specific tray).
    6. Record the current location of each tray.
    7. Iterate through every child identified during initialization:
       a. If the child is already in the set of served children, add 0 to the total cost and proceed to the next child.
       b. If the child is not served, determine their waiting location (from static information) and whether they require a gluten-free sandwich (from static information).
       c. Search among the existing sandwiches to find one that is "suitable" for this child (gluten-free if required, any if not required).
       d. Determine the "best status" among all suitable sandwiches found, representing the minimum number of steps needed to get *a* suitable sandwich into a position where it can be served to this child:
          - "ready": A suitable sandwich is found on a tray that is currently located at the child's waiting place. This is the ideal status.
          - "on_tray_elsewhere": A suitable sandwich is found on a tray, but that tray is located at a different place than the child's waiting location.
          - "in_kitchen": A suitable sandwich is found in the kitchen (not yet on any tray).
          - "needs_making": No suitable sandwich is found anywhere in the current state.
       e. Assign a cost to the total heuristic based on the determined best status for this child:
          - If "ready": Add 1 (representing the `serve` action).
          - If "on_tray_elsewhere": Add 2 (representing a `move_tray` action and a `serve` action).
          - If "in_kitchen": Add 3 (representing a `put_on_tray` action, a `move_tray` action, and a `serve` action).
          - If "needs_making": Add 4 (representing a `make_sandwich` action, a `put_on_tray` action, a `move_tray` action, and a `serve` action). This assumes a new sandwich can be made.
    8. The final `total_cost` accumulated is the heuristic value for the given state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children.
        """
        self.static_facts = task.static

        # Extract static info: allergic children and waiting places
        self.allergic_children = set()
        self.waiting_places = {} # child -> place
        self.all_children = set() # Collect all children mentioned in static facts

        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts
            predicate = parts[0]

            if predicate == "allergic_gluten":
                if len(parts) > 1:
                    self.allergic_children.add(parts[1])
                    self.all_children.add(parts[1])
            elif predicate == "not_allergic_gluten":
                 if len(parts) > 1:
                    self.all_children.add(parts[1])
            elif predicate == "waiting":
                if len(parts) > 2:
                    child, place = parts[1], parts[2]
                    self.waiting_places[child] = place
                    self.all_children.add(child)

        # Note: Assuming all children that need serving are listed in static 'waiting' facts.
        # The goal state is typically `(served ?c)` for all children ?c defined in the problem.
        # The set self.all_children should ideally come from the problem's object list,
        # but extracting from static 'allergic_gluten', 'not_allergic_gluten', 'waiting'
        # predicates covers all children mentioned in the static definition of the problem.


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to serve all unserved children.
        """
        state = node.state
        total_cost = 0

        # Identify served children in the current state
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # Identify existing sandwiches and their properties/locations
        existing_sandwiches = set()
        gluten_free_sandwiches = set()
        sandwich_location = {} # sandwich -> 'kitchen' or tray_name

        # Find all objects mentioned as sandwiches in relevant predicates
        all_mentioned_sandwiches = set()
        for fact in state:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             if predicate in ["at_kitchen_sandwich", "ontray", "no_gluten_sandwich", "notexist"]:
                 if len(parts) > 1:
                     all_mentioned_sandwiches.add(parts[1])

        # Determine which mentioned sandwiches actually exist (are not marked as notexist)
        notexist_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "notexist", "*")}
        existing_sandwiches = all_mentioned_sandwiches - notexist_sandwiches

        # Determine which existing sandwiches are gluten-free
        for s in existing_sandwiches:
             if f"(no_gluten_sandwich {s})" in state:
                 gluten_free_sandwiches.add(s)

        # Find current locations of existing sandwiches
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == "at_kitchen_sandwich":
                if len(parts) > 1:
                    sandwich_location[parts[1]] = 'kitchen'
            elif predicate == "ontray":
                 if len(parts) > 2:
                    sandwich_location[parts[1]] = parts[2] # Store tray name

        # Find current locations of trays
        tray_location = {} # tray -> place
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            # Assuming objects starting with 'tray' are trays
            if predicate == "at" and len(parts) > 2 and parts[1].startswith("tray"):
                 tray_location[parts[1]] = parts[2]

        # Iterate through all children identified in __init__
        for child in self.all_children:
            if child in served_children:
                continue # This child is already served, cost is 0 for them

            # Child is unserved, calculate cost for this child
            child_place = self.waiting_places.get(child)
            # If a child is in all_children but not waiting, they cannot be served by current actions.
            # Assuming valid problems mean all goal children would be waiting.
            if child_place is None:
                 # This case implies an issue with the problem definition or my assumption.
                 # In a standard setup, all goal children would be waiting.
                 # If they aren't waiting, they can't be served by the 'serve' actions.
                 # A heuristic could return infinity or a very high cost if a goal child isn't waiting,
                 # but there's no action to make them wait. Let's assume they are always waiting
                 # at the static location if not served.
                 continue # Skip if waiting place is not found (shouldn't happen in valid problems)


            gluten_free_required = child in self.allergic_children

            # Determine the best status of a suitable sandwich for this child
            best_status = "needs_making" # Default: assume sandwich needs to be made

            # Check existing sandwiches for suitability and location
            for s in existing_sandwiches:
                is_suitable = False
                if gluten_free_required:
                    # Suitable only if it's a gluten-free sandwich
                    if s in gluten_free_sandwiches:
                        is_suitable = True
                else:
                    # Not allergic, any existing sandwich is suitable
                    is_suitable = True

                if is_suitable:
                    s_loc = sandwich_location.get(s) # Get location of sandwich s

                    if s_loc and s_loc.startswith("tray"): # Sandwich is on a tray
                        tray_name = s_loc
                        current_tray_place = tray_location.get(tray_name)

                        if current_tray_place == child_place:
                            best_status = "ready" # Found a suitable sandwich on a tray at the child's location
                            break # Found the best possible status for this child, no need to check other sandwiches

                        # If not ready, check if it's on a tray elsewhere.
                        # Update status only if it's currently worse (needs_making or in_kitchen).
                        if best_status == "needs_making" or best_status == "in_kitchen":
                             best_status = "on_tray_elsewhere"

                    elif s_loc == 'kitchen': # Sandwich is in the kitchen
                         # Update status only if it's currently worse (needs_making).
                         if best_status == "needs_making":
                             best_status = "in_kitchen"

            # Add cost for this unserved child based on the best status found
            if best_status == "ready":
                total_cost += 1 # Cost: serve
            elif best_status == "on_tray_elsewhere":
                total_cost += 2 # Cost: move_tray, serve
            elif best_status == "in_kitchen":
                total_cost += 3 # Cost: put_on_tray, move_tray, serve
            elif best_status == "needs_making":
                # Cost: make_sandwich, put_on_tray, move_tray, serve
                # This assumes making is possible (notexist object, ingredients available).
                total_cost += 4

        return total_cost
