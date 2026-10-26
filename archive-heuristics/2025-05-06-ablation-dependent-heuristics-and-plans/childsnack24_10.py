from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj room)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts in the fact is at least the number of arguments in the pattern
    if len(parts) < len(args):
        return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It sums up the estimated costs for four main stages for the unserved children:
    1. Serving the child.
    2. Making a suitable sandwich.
    3. Putting a sandwich on a tray.
    4. Moving a tray to the child's location.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - Sandwiches need to be made, put on a tray, and the tray moved to the child's location before serving.
    - Resources (bread, content, notexist sandwich objects, trays) are assumed sufficient if they exist in the initial problem definition, although the heuristic does count the *need* to make sandwiches based on available ones.
    - A single tray move can potentially serve multiple children waiting at the same location.
    - A single make/put_on_tray action provides one sandwich or puts one sandwich on a tray.

    # Heuristic Initialization
    - Extracts static information about child allergies and gluten-free ingredients.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of four components, representing the estimated number of actions needed for different stages of the process for all unserved children:

    1.  **Unserved Children Count:** The number of children who are not yet served. Each requires a final 'serve' action.
        Cost = Number of children `C` such that `(served C)` is false.

    2.  **Sandwiches to Make Count:** The number of *new* sandwiches that need to be made to satisfy the demand from unserved children, considering available sandwiches (both in the kitchen and on trays). This distinguishes between gluten-free and regular sandwich needs.
        - Count unserved allergic children (need GF).
        - Count unserved non-allergic children (need Any).
        - Count available GF sandwiches (on tray or in kitchen).
        - Count available non-GF sandwiches (on tray or in kitchen).
        - Calculate how many GF sandwiches are still needed after using available ones.
        - Calculate how many Any sandwiches are still needed after using available non-GF and leftover available GF ones.
        - The sum is the number of 'make_sandwich' actions required.
        Cost = Number of suitable sandwiches required by unserved children that are not currently available (either in the kitchen or on a tray).

    3.  **Sandwiches Needing Put-on-Tray Count:** The number of sandwiches that need to be moved from the kitchen onto a tray. This is estimated by the total number of sandwiches needed by unserved children minus those already on trays.
        Cost = Maximum of 0 and (Number of unserved children - Number of sandwiches already on trays).

    4.  **Trays Needing Move Count:** The number of distinct locations where unserved children are waiting but where no tray is currently located. At least one tray needs to be moved to each such location.
        - Identify all places where unserved children are waiting.
        - Identify all places where trays are located.
        - Count the waiting places that do not have a tray.
        Cost = Number of places `P` such that some unserved child is `waiting` at `P`, but no tray is `at P`.

    The total heuristic value is the sum of these four costs. If all children are served, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.
        """
        self.goals = task.goals

        # Extract static information
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()

        for fact in task.static:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "allergic_gluten":
                self.allergic_children.add(parts[1])
            elif predicate == "not_allergic_gluten":
                self.not_allergic_children.add(parts[1])
            elif predicate == "no_gluten_bread":
                self.no_gluten_breads.add(parts[1])
            elif predicate == "no_gluten_content":
                self.no_gluten_contents.add(parts[1])

        # Get the set of all children mentioned in the goals
        self.goal_children = set(get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*"))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # 1. Count unserved children
        unserved_children = {c for c in self.goal_children if f"(served {c})" not in state}
        N_unserved = len(unserved_children)

        if N_unserved == 0:
            return 0 # Goal state reached

        total_cost += N_unserved # Cost for 'serve' actions

        # 2. Count sandwiches needing 'make'
        needed_gf = sum(1 for c in unserved_children if c in self.allergic_children)
        needed_any = sum(1 for c in unserved_children if c in self.not_allergic_children)

        # Count available sandwiches by type and location
        avail_gf_ontray = 0
        avail_nongf_ontray = 0
        avail_gf_kitchen = 0
        avail_nongf_kitchen = 0

        for fact in state:
            if match(fact, "ontray", "*", "*"):
                sandwich = get_parts(fact)[1]
                if f"(no_gluten_sandwich {sandwich})" in state:
                    avail_gf_ontray += 1
                else:
                    avail_nongf_ontray += 1
            elif match(fact, "at_kitchen_sandwich", "*"):
                 sandwich = get_parts(fact)[1]
                 if f"(no_gluten_sandwich {sandwich})" in state:
                     avail_gf_kitchen += 1
                 else:
                     avail_nongf_kitchen += 1

        total_avail_gf = avail_gf_ontray + avail_gf_kitchen
        total_avail_nongf = avail_nongf_ontray + avail_nongf_kitchen

        # Calculate how many of each type need to be made
        n_make_gf = max(0, needed_gf - total_avail_gf)

        # Remaining 'any' need can be satisfied by remaining available GF or available non-GF
        remaining_any_needed = needed_any
        remaining_avail_gf = max(0, total_avail_gf - needed_gf) # GF not used by allergic
        remaining_any_needed = max(0, remaining_any_needed - remaining_avail_gf) # Use leftover GF for non-allergic
        remaining_any_needed = max(0, remaining_any_needed - total_avail_nongf) # Use non-GF for non-allergic

        n_make_any = remaining_any_needed

        n_make = n_make_gf + n_make_any
        total_cost += n_make # Cost for 'make_sandwich' actions

        # 3. Count sandwiches needing 'put_on_tray'
        # Number of sandwiches that need to end up on a tray is N_unserved.
        # Subtract those already on trays.
        n_ontray = avail_gf_ontray + avail_nongf_ontray
        n_need_put_on_tray = max(0, N_unserved - n_ontray)
        total_cost += n_need_put_on_tray # Cost for 'put_on_tray' actions

        # 4. Count trays needing 'move_tray'
        waiting_places_needed = set()
        for fact in state:
            if match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1], get_parts(fact)[2]
                if child in unserved_children:
                    waiting_places_needed.add(place)

        tray_locations = set()
        for fact in state:
             if match(fact, "at", "*", "*"):
                 obj, place = get_parts(fact)[1], get_parts(fact)[2]
                 if obj.startswith("tray"):
                     tray_locations.add(place)

        n_places_need_tray = sum(1 for p in waiting_places_needed if p not in tray_locations)
        total_cost += n_places_need_tray # Cost for 'move_tray' actions

        return total_cost
