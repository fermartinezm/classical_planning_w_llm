from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved children.
    It sums the number of children not yet served and the number of sandwiches that still need to be created from ingredients to meet the demand of unserved children.

    # Assumptions
    - Ingredients (bread, content) are sufficient in the kitchen if they exist in the initial state to make any required sandwich type.
    - The costs of putting sandwiches on trays and moving trays are not explicitly counted, assuming they are less significant bottlenecks or are implicitly handled by ensuring sandwiches exist.
    - Each unserved child requires one suitable sandwich.

    # Heuristic Initialization
    - Identify all children and their allergy status (allergic_gluten or not_allergic_gluten) from static facts.
    - Identify the waiting place for each child from static facts (although the place is not directly used in this simplified heuristic formula, it's good practice to extract relevant static info).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all children who have not yet been served (`(served ?c)` is not in the state).
    2. Count the total number of unserved children (`N_unserved`). This contributes `N_unserved` to the heuristic (one `serve` action per child is the minimum final step).
    3. Categorize the unserved children by their allergy status (gluten-allergic vs. not-allergic) using the static information. Count the number of unserved GF children (`N_unserved_gf`) and unserved regular children (`N_unserved_regular`).
    4. Count the number of available sandwiches in the current state, categorized by type (gluten-free vs. regular). A sandwich is available if it exists (i.e., is not `notexist`), regardless of whether it's in the kitchen or on a tray. Check for the presence of the `(no_gluten_sandwich ?s)` fact to determine the type.
    5. Calculate the number of GF sandwiches that still need to be *made*: `max(0, N_unserved_gf - N_gf_available)`.
    6. Calculate the number of regular sandwiches that still need to be *made*: `max(0, N_unserved_regular - N_regular_available)`.
    7. Sum the counts from steps 5 and 6 to get the total number of sandwiches that need to be made (`sandwiches_to_make`). This contributes `sandwiches_to_make` to the heuristic (one `make_sandwich` action per sandwich).
    8. The total heuristic value is the sum of the costs from steps 2 and 7: `N_unserved + sandwiches_to_make`.
    9. If the state is a goal state (all children served), `N_unserved` will be 0. Since no children need serving, `N_unserved_gf` and `N_unserved_regular` will also be 0, resulting in `sandwiches_to_make = 0`. Thus, the heuristic is 0 if and only if the goal is reached.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children.
        """
        self.goals = task.goals # Store goals to check for goal state (optional, but good practice)
        static_facts = task.static

        # Extract static information: child allergies and waiting locations
        self.child_allergy = {} # {child_name: 'gluten' or 'not_gluten'}
        self.child_waiting_place = {} # {child_name: place_name}

        # Assuming all children are mentioned in either allergy or waiting facts in static
        all_children_set = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                child = parts[1]
                self.child_allergy[child] = 'gluten'
                all_children_set.add(child)
            elif parts[0] == 'not_allergic_gluten':
                 child = parts[1]
                 self.child_allergy[child] = 'not_gluten'
                 all_children_set.add(child)
            elif parts[0] == 'waiting':
                 child = parts[1]
                 place = parts[2]
                 self.child_waiting_place[child] = place
                 all_children_set.add(child)

        self.all_children = frozenset(all_children_set)


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # 1. Check if goal is reached (heuristic is 0)
        # This check is implicitly handled by the calculation, but explicit check is faster
        if self.goals <= state:
             return 0

        # 2. Identify unserved children and count
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = self.all_children - served_children
        num_unserved = len(unserved_children)

        # If no unserved children, goal is reached (redundant due to check above, but safe)
        if num_unserved == 0:
             return 0

        # 3. Categorize unserved children by allergy
        num_unserved_gf = 0
        num_unserved_regular = 0

        for child in unserved_children:
            # Use .get for safety, although static facts should cover all children
            allergy = self.child_allergy.get(child)
            if allergy == 'gluten':
                num_unserved_gf += 1
            elif allergy == 'not_gluten':
                num_unserved_regular += 1
            # Children not found in static facts would be ignored, assuming they don't exist or aren't goals.

        # 4. Count available sandwiches by type
        existing_sandwiches = set()
        sandwich_is_gf = set()

        for fact in state:
            parts = get_parts(fact)
            # Sandwiches exist if they are in the kitchen or on a tray
            if parts[0] in ["at_kitchen_sandwich", "ontray"]:
                existing_sandwiches.add(parts[1])
            # GF status is a separate fact about the sandwich object
            elif parts[0] == "no_gluten_sandwich":
                 sandwich_is_gf.add(parts[1])

        # Count available GF and regular sandwiches among those that exist
        available_gf_sandwiches = len(existing_sandwiches.intersection(sandwich_is_gf))
        available_regular_sandwiches = len(existing_sandwiches - sandwich_is_gf) # Sandwiches that exist but are not GF

        # 5. Calculate sandwiches to make
        sandwiches_to_make_gf = max(0, num_unserved_gf - available_gf_sandwiches)
        sandwiches_to_make_regular = max(0, num_unserved_regular - available_regular_sandwiches)
        sandwiches_to_make = sandwiches_to_make_gf + sandwiches_to_make_regular

        # 6. Total heuristic value
        heuristic_value = num_unserved + sandwiches_to_make

        return heuristic_value
