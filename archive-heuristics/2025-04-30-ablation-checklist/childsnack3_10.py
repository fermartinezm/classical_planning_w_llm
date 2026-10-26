import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and empty facts
    fact = fact.strip()
    if not fact:
        return []
    # Remove outer parentheses
    if fact.startswith('(') and fact.endswith(')'):
        fact = fact[1:-1]
    # Split by whitespace
    return fact.split()

def match(fact, *args):
    """
    Check if a PDDL fact string matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    waiting children. It counts the number of unserved children and adds
    estimates for the necessary make_sandwich, put_on_tray, and move_tray
    actions based on the current state of sandwiches and trays relative
    to the children's needs and locations.

    # Assumptions
    - All children listed in the goal must be served.
    - Bread and content portions are sufficient if available in the kitchen.
    - notexist sandwich objects are sufficient if available.
    - Trays can be used to serve multiple children sequentially or at the same location.
    - Gluten-free sandwiches can serve non-allergic children, but regular
      sandwiches cannot serve allergic children.

    # Heuristic Initialization
    - Extracts the list of all children who need to be served from the goal.
    - Extracts static information about children's allergy status and
      waiting locations from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who need to be served (from the goal).
    2. Identify which of these children are not yet served in the current state.
    3. Categorize unserved children by allergy status (allergic/non-allergic).
    4. Count existing sandwiches (in kitchen or on trays), distinguishing
       between gluten-free and regular.
    5. Count sandwiches that are already on trays, distinguishing GF/regular.
    6. Identify the locations where unserved children are waiting.
    7. Identify the locations where trays are currently present.
    8. Calculate the heuristic as the sum of several components:
       a.  Number of unserved children (lower bound on serve actions).
       b.  Number of suitable sandwiches that still need to be made
           (based on unserved children vs existing sandwiches).
       c.  Number of sandwiches that still need to be put on trays
           (based on unserved children vs sandwiches already on trays).
       d.  Number of waiting locations that do not currently have a tray
           (lower bound on move_tray actions to get trays to needed locations).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal children and static facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Identify all children who need to be served from the goal facts
        self.children_to_serve = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "served":
                self.children_to_serve.add(parts[1])

        # Map children to their allergy status and waiting place from static facts
        self.child_info = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == "allergic_gluten":
                child_name = parts[1]
                if child_name in self.children_to_serve:
                    if child_name not in self.child_info:
                        self.child_info[child_name] = {}
                    self.child_info[child_name]['allergic'] = True
            elif parts[0] == "not_allergic_gluten":
                 child_name = parts[1]
                 if child_name in self.children_to_serve:
                    if child_name not in self.child_info:
                        self.child_info[child_name] = {}
                    self.child_info[child_name]['allergic'] = False
            elif parts[0] == "waiting":
                child_name = parts[1]
                place_name = parts[2]
                if child_name in self.children_to_serve:
                    if child_name not in self.child_info:
                        self.child_info[child_name] = {}
                    self.child_info[child_name]['waiting_place'] = place_name

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # 1. Identify unserved children and categorize by allergy
        unserved_children = set()
        allergic_unserved = 0
        non_allergic_unserved = 0
        waiting_places = set() # Places where unserved children are waiting

        for child in self.children_to_serve:
            if f"(served {child})" not in state:
                unserved_children.add(child)
                info = self.child_info.get(child, {}) # Get info, handle cases where child might be in goal but not static waiting (unlikely in valid problems)
                if info.get('allergic', False): # Default to False if info missing
                    allergic_unserved += 1
                else:
                    non_allergic_unserved += 1
                if 'waiting_place' in info:
                     waiting_places.add(info['waiting_place'])


        num_unserved = len(unserved_children)

        # If no children are unserved, the goal is reached.
        if num_unserved == 0:
            return 0

        # 2. Count existing sandwiches and those on trays, distinguishing GF/regular
        existing_sandwiches = set()
        gf_sandwiches = set()
        sandwiches_ontray = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == "at_kitchen_sandwich":
                sandwich_name = parts[1]
                existing_sandwiches.add(sandwich_name)
            elif parts[0] == "ontray":
                sandwich_name = parts[1]
                tray_name = parts[2]
                existing_sandwiches.add(sandwich_name)
                sandwiches_ontray.add(sandwich_name)
            elif parts[0] == "no_gluten_sandwich":
                 sandwich_name = parts[1]
                 # This fact just marks a sandwich as GF, it doesn't mean it exists yet
                 # We only care about existing GF sandwiches
                 if sandwich_name in existing_sandwiches:
                     gf_sandwiches.add(sandwich_name)

        num_existing_sandwiches = len(existing_sandwiches)
        num_gf_existing = len(gf_sandwiches)
        num_reg_existing = num_existing_sandwiches - num_gf_existing # Assume non-GF is regular

        num_sandwiches_ontray = len(sandwiches_ontray)
        # num_gf_ontray = len([s for s in sandwiches_ontray if s in gf_sandwiches]) # Could refine if needed, but total ontray is enough for this heuristic part
        # num_reg_ontray = num_sandwiches_ontray - num_gf_ontray

        # 3. Count tray locations
        tray_locations = set()
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                # Check if the first argument is a tray type object (heuristic needs access to types,
                # but we can infer from common naming or check against known tray objects if available.
                # A simpler approach is to assume anything with "(at ?obj ?place)" where ?obj is not
                # a child or sandwich is likely a tray or similar movable object.
                # Given the domain, only trays move.
                obj_name = parts[1]
                place_name = parts[2]
                # A more robust check would use task.objects, but let's assume 'tray' in name or similar
                # Or just check if it's not kitchen_bread/content/sandwich/child/notexist
                # The simplest is to assume anything 'at' a place other than kitchen_... is a tray
                # Or, better, iterate through task.objects to find trays. Let's skip that for simplicity
                # and rely on the structure of the domain/instance.
                # The fact format is (at tray_name place_name)
                if obj_name.startswith('tray'): # Simple check based on naming convention
                     tray_locations.add(place_name)


        # 4. Calculate heuristic components

        # Component 1: Serve actions
        # Each unserved child needs one serve action.
        h_serve = num_unserved

        # Component 2: Make sandwich actions
        # Need enough GF sandwiches for allergic children, and enough total
        # sandwiches for all unserved children.
        # Number of GF sandwiches that still need to be made:
        needed_gf_made = max(0, allergic_unserved - num_gf_existing)
        # Number of regular sandwiches that still need to be made:
        # Non-allergic children can take regular or GF. We prioritize using existing GF for allergic.
        # Total sandwiches needed = num_unserved.
        # Total existing = num_existing_sandwiches.
        # Total needing to be made = max(0, num_unserved - num_existing_sandwiches).
        # This total must be split into GF and Regular.
        # We need needed_gf_made GF ones. The rest can be regular.
        # The number of regular ones needed is max(0, num_unserved - num_existing_sandwiches - needed_gf_made)
        # This is equivalent to max(0, non_allergic_unserved - num_reg_existing) if we assume GF are used for allergic first.
        needed_reg_made = max(0, non_allergic_unserved - num_reg_existing)

        h_make = needed_gf_made + needed_reg_made

        # Component 3: Put on tray actions
        # Each unserved child needs a sandwich on a tray.
        # Number of sandwiches that still need to be put on trays:
        h_put_on_tray = max(0, num_unserved - num_sandwiches_ontray)

        # Component 4: Move tray actions
        # Each waiting place needs a tray. Count places with unserved children
        # that do not currently have a tray.
        places_needing_tray_moved = waiting_places - tray_locations
        h_move_tray = len(places_needing_tray_moved)

        # Total heuristic is the sum of these components.
        # This is non-admissible as actions contribute to multiple components
        # (e.g., make -> increases existing, put_on_tray -> increases ontray).
        # But it provides a reasonable estimate of steps in different stages.
        total_heuristic = h_serve + h_make + h_put_on_tray + h_move_tray

        return total_heuristic

