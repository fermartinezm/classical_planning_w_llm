from fnmatch import fnmatch
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static

        def __call__(self, node):
            raise NotImplementedError


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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all unserved
    children. It counts the number of children needing serving, the number of
    locations needing a tray, the number of sandwiches needing to be made, and
    the number of sandwiches needing to be put on a tray.

    # Assumptions
    - Each action (make_sandwich, put_on_tray, move_tray, serve_sandwich) costs 1.
    - Resources like bread, content, and trays are available when needed for the count.
    - A single tray move can potentially serve multiple children at the destination.
    - A single put_on_tray action moves one sandwich onto a tray.
    - Non-allergic children can eat any type of sandwich (gluten-free or regular).
    - Allergic children can only eat gluten-free sandwiches.

    # Heuristic Initialization
    - Extracts all children, places where children wait, trays, and sandwiches
      from the initial state and static facts.
    - Stores static facts like waiting locations and allergy information.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of estimated costs for four main components:

    1.  **Serving Cost:** Count the number of children who are not yet served. Each unserved child requires a final 'serve' action. Add this count to the heuristic.

    2.  **Tray Movement Cost:** Identify all locations where unserved children are waiting. For each such location, check if there is currently any tray present. If a location has unserved children but no tray, a tray must be moved there. Count the number of such locations and add this count to the heuristic.

    3.  **Sandwich Creation Cost:** Determine the total number of gluten-free (GF) and non-gluten-free (non-GF) sandwiches required to serve all unserved children (allergic children need GF, non-allergic can take any). Count the number of GF and non-GF sandwiches already available globally (either at the kitchen or on any tray). Calculate how many additional GF and non-GF sandwiches need to be made. Account for the fact that excess available GF sandwiches can satisfy non-GF needs. Add the total number of sandwiches to be made to the heuristic.

    4.  **Put on Tray Cost:** Determine the total number of sandwiches that need to end up on a tray to be delivered (this is the total number of sandwiches required). Count the number of sandwiches that are already on trays. The difference represents the number of sandwiches that must be put on a tray (either from existing kitchen stock or after being made). Add this count to the heuristic.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting object names and static facts."""
        super().__init__(task)

        # Extract all relevant objects from initial state and static facts
        self.all_children = {get_parts(fact)[1] for fact in self.static if match(fact, "waiting", "*", "*")}
        self.all_places = {get_parts(fact)[2] for fact in self.static if match(fact, "waiting", "*", "*")}
        # Get all tray objects mentioned in the initial state
        self.all_trays = {get_parts(fact)[1] for fact in task.initial_state if match(fact, "at", "*", "*")}
        # Get all sandwich objects mentioned in the initial state (usually as notexist)
        self.all_sandwiches = {get_parts(fact)[1] for fact in task.initial_state if match(fact, "notexist", "*")}
        # Also include any sandwiches that might somehow exist initially (e.g., in a different problem definition)
        self.all_sandwiches.update({get_parts(fact)[1] for fact in task.initial_state if match(fact, "at_kitchen_sandwich", "*")})
        self.all_sandwiches.update({get_parts(fact)[1] for fact in task.initial_state if match(fact, "ontray", "*", "*")})


        # Store static information for quick lookup
        self.child_waiting_location = {}
        self.child_needs_gf = {}
        for fact in self.static:
            if match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1], get_parts(fact)[2]
                self.child_waiting_location[child] = place
            elif match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.child_needs_gf[child] = True
            elif match(fact, "not_allergic_gluten", "*"):
                 child = get_parts(fact)[1]
                 self.child_needs_gf[child] = False # Explicitly store False

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # 1. Identify unserved children and their needs/locations
        unserved_children_info = [] # List of (child, location, needs_gf)
        for child in self.all_children:
            if '(served ' + child + ')' not in state:
                location = self.child_waiting_location.get(child) # Get location from static info
                needs_gf = self.child_needs_gf.get(child, False) # Get allergy info from static info, default to False
                if location: # Only consider children who are waiting somewhere
                     unserved_children_info.append((child, location, needs_gf))

        # If no unserved children, goal reached
        if not unserved_children_info:
            return 0

        # 2. Cost Component 1: Serving
        # Each unserved child needs a serve action
        h += len(unserved_children_info)

        # 3. Cost Component 2: Tray Movement
        # Count locations with unserved children that don't have a tray
        locations_needing_tray = set()
        trays_at_location = {get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*")} # Locations where trays currently are
        for child, location, needs_gf in unserved_children_info:
            if location not in trays_at_location:
                 locations_needing_tray.add(location)
        h += len(locations_needing_tray)

        # 4. Cost Component 3 & 4: Sandwich Preparation (Make and Put on Tray)

        # Count how many GF/non-GF sandwiches are needed in total by unserved children.
        needed_gf_total = sum(1 for child, loc, needs_gf in unserved_children_info if needs_gf)
        needed_non_gf_total = sum(1 for child, loc, needs_gf in unserved_children_info if not needs_gf)

        # Count available suitable sandwiches globally (kitchen + on tray).
        available_gf_global = sum(1 for s in self.all_sandwiches if '(no_gluten_sandwich ' + s + ')' in state and ('(at_kitchen_sandwich ' + s + ')' in state or any('(ontray ' + s + ' ' + t + ')' in state for t in self.all_trays)))
        available_non_gf_global = sum(1 for s in self.all_sandwiches if '(no_gluten_sandwich ' + s + ')' not in state and ('(at_kitchen_sandwich ' + s + ')' in state or any('(ontray ' + s + ' ' + t + ')' in state for t in self.all_trays)))

        # Cost to Make:
        # Need needed_gf_total GF. Have available_gf_global. Make max(0, needed_gf_total - available_gf_global).
        num_make_gf = max(0, needed_gf_total - available_gf_global)
        # Excess GF can cover non-GF needs. Excess GF = max(0, available_gf_global - needed_gf_total).
        # Need needed_non_gf_total non-GF. Have available_non_gf_global. Make max(0, needed_non_gf_total - available_non_gf_global - Excess GF).
        excess_gf_global = max(0, available_gf_global - needed_gf_total)
        num_make_non_gf = max(0, needed_non_gf_total - available_non_gf_global - excess_gf_global)
        h += num_make_gf + num_make_non_gf

        # Cost to Put on Tray:
        # Count sandwiches already on trays.
        ontray_gf = sum(1 for s in self.all_sandwiches if '(no_gluten_sandwich ' + s + ')' in state and any('(ontray ' + s + ' ' + t + ')' in state for t in self.all_trays)))
        ontray_non_gf = sum(1 for s in self.all_sandwiches if '(no_gluten_sandwich ' + s + ')' not in state and any('(ontray ' + s + ' ' + t + ')' in state for t in self.all_trays)))

        # Number of sandwiches that need to be put on a tray (either from kitchen stock or after being made).
        # This is the total number of sandwiches needed minus those already on trays.
        # Account for excess on-tray GF covering non-GF needs is implicitly handled by counting total needed vs total on tray.
        total_sandwiches_needed = needed_gf_total + needed_non_gf_total
        total_ontray_sandwiches = ontray_gf + ontray_non_gf
        # The number of sandwiches that *must* pass through the put_on_tray stage is the total needed minus those already past that stage.
        num_put_on_tray_needed = max(0, total_sandwiches_needed - total_ontray_sandwiches)
        h += num_put_on_tray_needed

        return h
