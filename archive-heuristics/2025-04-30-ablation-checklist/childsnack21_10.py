from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts represented as strings
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
    This heuristic estimates the number of actions required to serve all
    unserved children. It does this by estimating the steps needed for each
    unserved child independently, based on the current state of suitable
    sandwiches and trays.

    # Assumptions
    - Each unserved child needs one suitable sandwich.
    - A suitable sandwich for an allergic child must be gluten-free.
    - Any sandwich is suitable for a non-allergic child.
    - The heuristic estimates steps based on the *stage* of readiness for
      each child's sandwich:
      1. Sandwich on tray at child's location (needs 1 action: serve).
      2. Sandwich on tray elsewhere (needs 2 actions: move_tray, serve).
      3. Sandwich made, in kitchen (needs 3 actions: put_on_tray, move_tray, serve).
      4. Sandwich not yet made (needs 4 actions: make, put_on_tray, move_tray, serve).
    - This heuristic ignores resource constraints like the availability of
      bread, content, sandwich objects, or trays for making/putting sandwiches.
      It assumes these steps are always possible if the sandwich isn't in a
      later stage. This makes it non-admissible but efficiently computable.
    - The cost of moving a tray is always 1, regardless of distance (as per domain).

    # Heuristic Initialization
    - Extracts which children are allergic or not from static facts.
    - Extracts the waiting place for each child from static facts.
    - Identifies all children that need to be served from the goal state.

    # Step-By-Step Thinking for Computing Heuristic (__call__)
    1. Initialize total heuristic cost `h = 0`.
    2. Identify all children that are in the goal state but not yet served
       in the current state.
    3. For each unserved child:
       a. Determine if the child is allergic to gluten.
       b. Get the child's waiting place.
       c. Check the current state to see if a *suitable* sandwich is available
          in the most advanced stage:
          - Is there a suitable sandwich on a tray *at the child's waiting place*?
            (Requires `(ontray ?s ?t)` and `(at ?t ?p)` where `?s` is suitable).
            If yes, add 1 (for `serve`) to `h` and move to the next child.
          - If not, is there a suitable sandwich on *any* tray *anywhere*?
            (Requires `(ontray ?s ?t)` where `?s` is suitable).
            If yes, add 2 (for `move_tray` and `serve`) to `h` and move to the next child.
          - If not, is there a suitable sandwich *made* and `at_kitchen_sandwich`?
            (Requires `(at_kitchen_sandwich ?s)` where `?s` is suitable).
            If yes, add 3 (for `put_on_tray`, `move_tray`, and `serve`) to `h` and move to the next child.
          - If not, the sandwich needs to be made. Add 4 (for `make`, `put_on_tray`, `move_tray`, and `serve`) to `h` and move to the next child.
    4. Return the total cost `h`.

    # Suitability Check
    A sandwich `s` is suitable for child `c` if:
    - Child `c` is allergic AND `(no_gluten_sandwich s)` is true in the state.
    - OR Child `c` is NOT allergic.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children
        and their requirements/locations.
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Facts that are not affected by actions

        # Map children to their allergy status
        self.is_allergic = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "allergic_gluten":
                self.is_allergic[parts[1]] = True
            elif parts[0] == "not_allergic_gluten":
                self.is_allergic[parts[1]] = False

        # Map children to their waiting place
        self.waiting_place = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "waiting":
                self.waiting_place[parts[1]] = parts[2]

        # Identify all children that need to be served (from the goal)
        self.goal_children = {
            get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == "served"
        }

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to serve all unserved children.
        """
        state = node.state  # Current world state
        h = 0  # Initialize heuristic cost

        # Iterate through all children that need to be served
        for child in self.goal_children:
            # If the child is already served, they don't contribute to the heuristic
            if f"(served {child})" in state:
                continue

            # Child is not served, estimate steps needed
            child_place = self.waiting_place[child]
            needs_gluten_free = self.is_allergic.get(child, False) # Default to False if allergy status unknown

            # --- Check Stage 4: Ready to Serve? (Sandwich on tray at child's place) ---
            found_ready_sandwich = False
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == "ontray":
                    sandwich, tray = parts[1], parts[2]
                    # Check if the sandwich is suitable
                    is_no_gluten = f"(no_gluten_sandwich {sandwich})" in state
                    is_suitable = (needs_gluten_free and is_no_gluten) or (not needs_gluten_free)

                    if is_suitable:
                        # Check if the tray is at the child's location
                        if f"(at {tray} {child_place})" in state:
                            h += 1 # Cost for 'serve' action
                            found_ready_sandwich = True
                            break # Found a suitable sandwich ready to serve this child

            if found_ready_sandwich:
                continue # Move to the next unserved child

            # --- Check Stage 3: Sandwich on Tray Elsewhere? ---
            found_on_tray_elsewhere = False
            for fact in state:
                 parts = get_parts(fact)
                 if parts[0] == "ontray":
                    sandwich, tray = parts[1], parts[2]
                    # Check if the sandwich is suitable
                    is_no_gluten = f"(no_gluten_sandwich {sandwich})" in state
                    is_suitable = (needs_gluten_free and is_no_gluten) or (not needs_gluten_free)

                    if is_suitable:
                         # Check if the tray is NOT at the child's location (implies it's elsewhere)
                         # We don't need to find the exact location, just know it's not at child_place
                         # A tray is always at *some* location if it exists and is not carried (no carry predicate here)
                         # So, if it's on a tray, and not found at child_place above, it's elsewhere.
                         # However, let's be explicit and check for `(at tray place)` where place != child_place
                         for at_fact in state:
                             at_parts = get_parts(at_fact)
                             if at_parts[0] == "at" and at_parts[1] == tray and at_parts[2] != child_place:
                                 h += 2 # Cost for 'move_tray' + 'serve'
                                 found_on_tray_elsewhere = True
                                 break # Found a suitable sandwich on a tray elsewhere
                         if found_on_tray_elsewhere:
                             break # Break outer loop over ontray facts

            if found_on_tray_elsewhere:
                continue # Move to the next unserved child

            # --- Check Stage 2: Sandwich Made, In Kitchen? ---
            found_in_kitchen = False
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == "at_kitchen_sandwich":
                    sandwich = parts[1]
                    # Check if the sandwich is suitable
                    is_no_gluten = f"(no_gluten_sandwich {sandwich})" in state
                    is_suitable = (needs_gluten_free and is_no_gluten) or (not needs_gluten_free)

                    if is_suitable:
                        h += 3 # Cost for 'put_on_tray' + 'move_tray' + 'serve'
                        found_in_kitchen = True
                        break # Found a suitable sandwich in the kitchen

            if found_in_kitchen:
                continue # Move to the next unserved child

            # --- Stage 1: Sandwich Not Yet Made ---
            # If none of the above stages are met, the sandwich needs to be made.
            h += 4 # Cost for 'make' + 'put_on_tray' + 'move_tray' + 'serve'

        return h

