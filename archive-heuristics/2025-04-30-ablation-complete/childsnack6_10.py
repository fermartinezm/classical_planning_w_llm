from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
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
    This heuristic estimates the number of actions needed to serve all unserved
    children. It breaks down the process for each child into stages:
    making the sandwich, putting it on a tray, moving the tray to the child's
    location, and finally serving the child. It counts the total number of
    children who need a sandwich to pass through each of these stages,
    accounting for sandwiches already available at later stages.

    # Assumptions
    - Each unserved child requires exactly one sandwich.
    - Gluten-allergic children require gluten-free sandwiches.
    - Non-allergic children can receive any sandwich.
    - The sequence of preparation and delivery steps is generally:
      Make Sandwich -> Put on Tray -> Move Tray -> Serve.
    - Actions have a cost of 1.
    - The heuristic sums the estimated steps needed across all unserved children,
      accounting for shared resources (sandwiches at different stages) in an
      aggregated way, which makes it non-admissible but potentially more
      informative than a simple goal count.

    # Heuristic Initialization
    - Extracts static information about children: which are allergic, their
      waiting locations, and the set of all children.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic is calculated by summing up the estimated costs for groups
    of unserved children based on the "readiness" of their required sandwich.

    1.  **Base Cost (Serve Action):** Initialize the heuristic value with the
        total number of unserved children. Each unserved child requires at
        least one 'serve' action.

    2.  **Cost for Delivery/Placement:** Identify children who do *not* have
        a suitable sandwich already on a tray located at their waiting place.
        Each such child needs a sandwich delivered to their location (which
        involves getting a sandwich onto a tray and moving the tray). Add the
        count of these children to the heuristic. This count is the number of
        unserved children minus the number of children who *can* be served by
        sandwiches already on trays at their location.

    3.  **Cost for Putting on Tray:** Identify children who need delivery (from
        step 2) but cannot be covered by suitable sandwiches already existing
        on *any* tray (regardless of location). These children need a sandwich
        to be put on a tray first. Add the count of these children to the
        heuristic. This count is the number of children needing delivery minus
        the number of available suitable sandwiches on trays elsewhere.

    4.  **Cost for Making Sandwich:** Identify children who need a sandwich
        put on a tray (from step 3) but cannot be covered by suitable
        sandwiches already existing in the kitchen. These children need a
        sandwich to be made first. Add the count of these children to the
        heuristic. This count is the number of children needing put_on_tray
        minus the number of available suitable sandwiches in the kitchen.

    The total heuristic value is the sum of costs from steps 1, 2, 3, and 4.
    This method counts how many children's sandwich needs require progressing
    through each stage of preparation/delivery.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children.
        """
        super().__init__(task) # Call the base class constructor

        self.allergic_children = set()
        self.waiting_locations = {}
        self.all_children = set()

        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == 'allergic_gluten':
                child = parts[1]
                self.allergic_children.add(child)
                self.all_children.add(child)
            elif predicate == 'not_allergic_gluten':
                 # Assuming any child mentioned in allergic/not_allergic is a child object
                 self.all_children.add(parts[1])
            elif predicate == 'waiting':
                child, place = parts[1], parts[2]
                self.waiting_locations[child] = place

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # --- Parse State Facts ---
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        tray_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and len(get_parts(fact)) == 3} # Ensure correct arity
        sandwich_on_tray = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "ontray", "*", "*")}
        kitchen_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        gluten_free_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}
        # We don't strictly need kitchen_bread, kitchen_content, notexist_sandwiches for this heuristic logic,
        # as we only count up to the 'make_sandwich' step deficit, not ingredient availability.

        # --- Identify Unserved Children ---
        unserved_children = self.all_children - served_children
        unserved_gf_children = unserved_children.intersection(self.allergic_children)
        unserved_reg_children = unserved_children - unserved_gf_children

        # If no children are unserved, the goal is reached.
        if not unserved_children:
            return 0

        # --- Heuristic Calculation ---
        h = len(unserved_children) # Step 1: Cost for serve action

        # Step 2: Count children needing delivery/placement at location
        # Group unserved children by location to efficiently check trays at that location
        unserved_by_loc = {}
        for child in unserved_children:
            loc = self.waiting_locations.get(child) # Use .get for safety
            if loc is None: continue # Should not happen in valid problems
            if loc not in unserved_by_loc:
                unserved_by_loc[loc] = {'gf': [], 'reg': []}
            if child in self.allergic_children:
                unserved_by_loc[loc]['gf'].append(child)
            else:
                unserved_by_loc[loc]['reg'].append(child)

        # Calculate how many children at each location can be served by existing sandwiches there
        served_by_ready_gf = 0
        served_by_ready_reg = 0

        for loc, children_at_loc in unserved_by_loc.items():
            # Find trays at this location
            trays_at_loc = {t for t, p in tray_locations.items() if p == loc}
            # Find sandwiches on these trays
            sandwiches_on_trays_at_loc = {s for s, t in sandwich_on_tray.items() if t in trays_at_loc}

            # Count suitable sandwiches at this location
            available_gf_at_loc = {s for s in sandwiches_on_trays_at_loc if s in gluten_free_sandwiches}
            available_reg_at_loc = {s for s in sandwiches_on_trays_at_loc if s not in gluten_free_sandwiches} # Assuming non-GF is regular

            # Each sandwich can serve one child. Count how many children can be covered.
            served_by_ready_gf += min(len(children_at_loc['gf']), len(available_gf_at_loc))
            served_by_ready_reg += min(len(children_at_loc['reg']), len(available_reg_at_loc))

        # Children needing delivery are those not covered by ready slots
        needs_delivery_gf = max(0, len(unserved_gf_children) - served_by_ready_gf)
        needs_delivery_reg = max(0, len(unserved_reg_children) - served_by_ready_reg)
        h += needs_delivery_gf + needs_delivery_reg # Cost: 1 per child needing delivery

        # Step 3: Count children needing put_on_tray
        # These are children needing delivery minus those who can be covered by *any* on-tray sandwich.

        # Count total suitable sandwiches on *any* tray
        ontray_total_gf = sum(1 for s in sandwich_on_tray if s in gluten_free_sandwiches)
        ontray_total_reg = sum(1 for s in sandwich_on_tray if s not in gluten_free_sandwiches)

        # Available on-tray sandwiches that are *not* already at the correct location
        # This is the pool of sandwiches that can cover the 'needs_delivery' deficit.
        # The number of children needing put_on_tray is the deficit minus the number of children that can be served by on-tray-elsewhere sandwiches.
        # The number of children that can be served by on-tray-elsewhere sandwiches is limited by the number of such sandwiches available.

        # Number of suitable on-tray sandwiches available *elsewhere*
        available_ontray_elsewhere_gf = max(0, ontray_total_gf - served_by_ready_gf)
        available_ontray_elsewhere_reg = max(0, ontray_total_reg - served_by_ready_reg)

        # Children needing put_on_tray are those needing delivery minus those coverable by on-tray-elsewhere sandwiches
        # The number of children coverable by on-tray-elsewhere sandwiches is limited by the number of such sandwiches.
        needs_put_on_tray_gf = max(0, needs_delivery_gf - available_ontray_elsewhere_gf)
        needs_put_on_tray_reg = max(0, needs_delivery_reg - available_ontray_elsewhere_reg)
        h += needs_put_on_tray_gf + needs_put_on_tray_reg # Cost: 1 per child needing put_on_tray

        # Step 4: Count children needing make_sandwich
        # These are children needing put_on_tray minus those who can be covered by kitchen sandwiches.

        # Count total suitable sandwiches in the kitchen
        kitchen_total_gf = sum(1 for s in kitchen_sandwiches if s in gluten_free_sandwiches)
        kitchen_total_reg = sum(1 for s in kitchen_sandwiches if s not in gluten_free_sandwiches)

        # Children needing make_sandwich are those needing put_on_tray minus those covered by kitchen sandwiches
        needs_make_gf = max(0, needs_put_on_tray_gf - kitchen_total_gf)
        needs_make_reg = max(0, needs_put_on_tray_reg - kitchen_total_reg)
        h += needs_make_gf + needs_make_reg # Cost: 1 per child needing make_sandwich

        return h
