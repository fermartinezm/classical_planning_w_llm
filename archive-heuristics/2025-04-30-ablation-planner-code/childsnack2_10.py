from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Note: The 'match' helper function from examples is not strictly needed
# for this heuristic's logic, as we are directly checking predicate names
# and arguments. Keeping get_parts is sufficient.

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children
    who are currently waiting and not yet served. It considers the steps of
    making suitable sandwiches, getting them onto trays, moving trays to the
    children's locations, and finally serving the children.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - A 'suitable' sandwich matches the child's allergy status (gluten-free for allergic children, regular for others).
    - Trays can hold multiple sandwiches (heuristic counts 'put_on_tray' per sandwich needing delivery, not limited by tray capacity).
    - Ingredients and 'notexist' sandwich objects are available if needed for making sandwiches (creation cost is based on need vs existence, not resource availability).
    - 'move_tray' cost is counted once per location that needs delivery and doesn't currently have a tray.
    - 'put_on_tray' cost is counted once per sandwich that needs to be put on a tray for delivery to a location.

    # Heuristic Initialization
    The heuristic extracts static information from the task:
    - The set of children that need to be served (from the goal).
    - The allergy status (allergic_gluten or not_allergic_gluten) for each child.
    - The waiting location for each child.
    - The set of all possible places in the domain (kitchen, waiting locations, initial tray locations).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:

    1.  **Identify Unserved Children:** Determine which children from the goal list are not yet marked as 'served' in the current state. Group these unserved children by their waiting location and allergy status.

    2.  **Cost for Serving:** Each unserved child requires a final 'serve' action. Add the total number of unserved children to the heuristic cost.

    3.  **Cost for Delivery (Put on Tray + Move Tray):**
        - For each group of unserved children at a specific location with a specific allergy status:
            - Count how many suitable sandwiches (matching the allergy status) are *already* on trays that are currently *at* that location.
            - The number of children in this group who still need a suitable sandwich delivered to their location is the group size minus the count of suitable sandwiches already present at the location. Let this be `needing_delivery_to_loc`.
            - Add `needing_delivery_to_loc` to a running total for 'put_on_tray' actions needed across all groups. (Each sandwich needing delivery requires a 'put_on_tray' action).
            - If `needing_delivery_to_loc` is greater than 0 (meaning this location/allergy group needs delivery) AND there are no trays currently located at this specific location, add 1 to a running total for 'move_tray' actions needed. (This estimates the cost of getting a tray to the location).
        - Add the total 'put_on_tray' cost and the total 'move_tray' cost to the heuristic.

    4.  **Cost for Creation (Make Sandwich):**
        - Count the total number of gluten-free sandwiches needed across all unserved allergic children.
        - Count the total number of regular sandwiches needed across all unserved non-allergic children.
        - Count the total number of gluten-free sandwiches that currently exist in the state (either in the kitchen or on a tray).
        - Count the total number of regular sandwiches that currently exist in the state.
        - For each type (gluten-free and regular), if the number needed is greater than the number existing, add the difference to a running total for 'make_sandwich' actions.
        - Add the total 'make_sandwich' cost to the heuristic.

    5.  **Total Heuristic Value:** The sum of costs from steps 2, 3, and 4 is the heuristic value for the current state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        # Extract children that need to be served from the goal
        self.goal_children = {get_parts(g)[1] for g in task.goals if get_parts(g)[0] == 'served'}

        # Extract static facts about children and places
        self.child_allergy = {}  # {child: 'allergic_gluten' or 'not_allergic_gluten'}
        self.child_location = {} # {child: place}
        self.all_places = {'kitchen'} # Start with the constant kitchen

        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate in ['allergic_gluten', 'not_allergic_gluten']:
                self.child_allergy[parts[1]] = predicate
            elif predicate == 'waiting':
                self.child_location[parts[1]] = parts[2]
                self.all_places.add(parts[2]) # Add child's waiting place

        # Add initial tray locations to known places
        for fact in task.init:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == 'at' and parts[1].startswith('tray'):
                 self.all_places.add(parts[2])


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach a goal state.
        """
        state = node.state

        # --- Step 1: Identify Unserved Children and Group ---
        unserved_children = [c for c in self.goal_children if f'(served {c})' not in state]

        # If all goal children are served, the heuristic is 0.
        if not unserved_children:
            return 0

        unserved_by_loc_allergy = {} # {(place, allergy_status): [child1, child2, ...]}
        for child in unserved_children:
            loc = self.child_location.get(child)
            allergy = self.child_allergy.get(child)
            if loc and allergy: # Ensure we have location and allergy info
                key = (loc, allergy)
                if key not in unserved_by_loc_allergy:
                    unserved_by_loc_allergy[key] = []
                unserved_by_loc_allergy[key].append(child)
            # Note: Children without waiting location or allergy info are ignored by this heuristic.
            # This is a limitation, but typical for domain-dependent heuristics.

        # --- Step 2: Cost for Serving ---
        total_cost = len(unserved_children) # 1 action per unserved child

        # --- Extract Current State Information ---
        trays_at_place = {p: [] for p in self.all_places}
        sandwiches_on_tray = {} # {tray: [sandwich1, sandwich2, ...]}
        sandwich_is_gf = {} # {sandwich: True/False}
        existing_sandwiches = set() # All sandwiches that exist (not notexist)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'at' and parts[1].startswith('tray'):
                place = parts[2]
                tray = parts[1]
                if place in trays_at_place: # Only track trays at known places
                     trays_at_place[place].append(tray)
            elif predicate == 'ontray':
                s, t = parts[1], parts[2]
                if t not in sandwiches_on_tray:
                    sandwiches_on_tray[t] = []
                sandwiches_on_tray[t].append(s)
                existing_sandwiches.add(s)
            elif predicate == 'at_kitchen_sandwich':
                 existing_sandwiches.add(parts[1])
            elif predicate == 'no_gluten_sandwich':
                sandwich_is_gf[parts[1]] = True

        # Assume sandwiches not marked as gluten-free are regular
        for s in existing_sandwiches:
            if s not in sandwich_is_gf:
                sandwich_is_gf[s] = False # It's a regular sandwich

        # --- Step 3: Cost for Delivery (Put on Tray + Move Tray) ---
        locations_needing_delivery = set()
        total_put_on_tray_needed = 0

        for (loc, allergy), children in unserved_by_loc_allergy.items():
            needed_count = len(children)
            suitable_sandwiches_at_loc_on_trays = 0

            # Count suitable sandwiches already on trays at this location
            if loc in trays_at_place:
                for tray in trays_at_place[loc]:
                    if tray in sandwiches_on_tray:
                        for s in sandwiches_on_tray[tray]:
                            s_is_gf = sandwich_is_gf.get(s, False) # Default to False if status unknown
                            is_suitable = (allergy == 'allergic_gluten' and s_is_gf) or (allergy == 'not_allergic_gluten' and not s_is_gf)
                            if is_suitable:
                                suitable_sandwiches_at_loc_on_trays += 1

            # Number of sandwiches for this group that still need delivery to this location
            needing_delivery_to_loc = max(0, needed_count - suitable_sandwiches_at_loc_on_trays)

            if needing_delivery_to_loc > 0:
                locations_needing_delivery.add(loc)
                total_put_on_tray_needed += needing_delivery_to_loc

        # Cost for moving trays to locations that need delivery
        for loc in locations_needing_delivery:
            # Check if any tray is already at this location
            trays_already_at_loc = len(trays_at_place.get(loc, []))
            if trays_already_at_loc == 0:
                total_cost += 1 # move_tray action

        # Cost for putting sandwiches on trays
        total_cost += total_put_on_tray_needed # put_on_tray actions

        # --- Step 4: Cost for Creation (Make Sandwich) ---
        needed_gf_total = sum(len(children) for (loc, allergy), children in unserved_by_loc_allergy.items() if allergy == 'allergic_gluten')
        needed_reg_total = sum(len(children) for (loc, allergy), children in unserved_by_loc_allergy.items() if allergy == 'not_allergic_gluten')

        existing_gf_total = sum(1 for s, is_gf in sandwich_is_gf.items() if is_gf)
        existing_reg_total = sum(1 for s, is_gf in sandwich_is_gf.items() if not is_gf)

        to_make_gf = max(0, needed_gf_total - existing_gf_total)
        to_make_reg = max(0, needed_reg_total - existing_reg_total)

        total_cost += to_make_gf + to_make_reg # make_sandwich actions

        # --- Step 5: Total Heuristic Value ---
        return total_cost
