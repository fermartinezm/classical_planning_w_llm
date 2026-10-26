from heuristics.heuristic_base import Heuristic
from collections import defaultdict

def get_parts(fact):
    """Helper function to split a PDDL fact string into its parts."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    This heuristic estimates the number of actions required to reach the goal state
    (all children served) by summing up the estimated costs for different stages
    of the process for all unserved children. The stages are: making sandwiches,
    putting them on trays, moving trays with needed sandwiches to the correct locations,
    and finally serving the children.

    Assumptions:
    - This heuristic is designed for greedy best-first search and does not need
      to be admissible. Its goal is to minimize expanded nodes by providing a
      good estimate of the remaining cost.
    - Actions are assumed to have a unit cost of 1.
    - The heuristic uses a simplified model for resource management (sandwiches,
      trays), primarily counting the number of items/tasks that need to reach
      a certain state, rather than strictly modeling the minimum number of
      action applications considering resource constraints like tray capacity
      or shared tray movements. For example, the cost for moving trays counts
      the number of sandwiches needing delivery, not the minimum number of tray moves.

    Heuristic Initialization:
    The heuristic constructor extracts and stores static information from the
    planning task, which does not change during search. This includes:
    - Mapping each child to their allergy status (allergic or not_allergic).
    - Mapping each child to their waiting location.
    - Identifying all children in the problem.
    - Identifying gluten-free bread and content portions (though these are not
      directly used in the current heuristic calculation, they are typically
      static facts one might extract).

    Step-By-Step Thinking for Computing Heuristic:
    For a given state, the heuristic calculates the sum of four cost components:

    1.  Cost_serve: The number of children who have not yet been served. Each
        unserved child requires a final 'serve_sandwich' action. This component
        is a direct count of unsatisfied goal predicates.

    2.  Cost_make: The number of sandwiches that still need to be created to
        satisfy the needs of all unserved children. This is calculated as the
        total number of sandwiches needed (equal to the number of unserved
        children) minus the number of sandwiches that already exist (either
        in the kitchen or on trays).

    3.  Cost_put_on_tray: The number of sandwiches that need to be moved from
        the kitchen onto a tray. This is estimated by taking the total number
        of sandwiches needed on trays (equal to the number of unserved children)
        and subtracting those already on trays. The deficit must be fulfilled
        by sandwiches currently in the kitchen or those that will be made.
        The cost is the minimum of the available supply (kitchen stock + those
        to be made) and the deficit, representing the number of sandwiches
        that must pass through the 'put_on_tray' stage.

    4.  Cost_move_tray: The number of sandwiches that need to arrive at the
        correct location (the waiting place of the child who will eat it) on
        a tray. For each location where unserved children are waiting, the
        heuristic calculates how many suitable sandwiches are needed at that
        location but are not currently present there on a tray. This count
        considers allergy requirements and the fact that gluten-free sandwiches
        can satisfy non-allergic needs. The total Cost_move_tray is the sum
        of these deficits across all locations with unserved children. This
        component estimates the "delivery effort" needed, counting items needing
        transport rather than minimum tray movements.

    The total heuristic value is the sum of these four costs. A state is a goal
    state if and only if all children are served, in which case all cost
    components are zero, resulting in a heuristic value of 0.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by extracting static information from the task.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract static information about children (allergy, location)
        self.child_allergy = {}
        self.child_location = {}
        self.all_children = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                child = parts[1]
                self.child_allergy[child] = 'allergic'
                self.all_children.add(child)
            elif parts[0] == 'not_allergic_gluten':
                child = parts[1]
                self.child_allergy[child] = 'not_allergic'
                self.all_children.add(child)
            elif parts[0] == 'waiting':
                child, place = parts[1], parts[2]
                self.child_location[child] = place
                self.all_children.add(child)

        # Extract static information about gluten-free ingredients (for potential future use or completeness)
        # Note: These are not directly used in the current heuristic calculation logic,
        # as the focus is on the state of sandwiches and children.
        self.gf_bread = set()
        self.gf_content = set()
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == 'no_gluten_bread':
                 self.gf_bread.add(parts[1])
             elif parts[0] == 'no_gluten_content':
                 self.gf_content.add(parts[1])


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer estimate of the remaining cost to reach the goal.
        """
        state = node.state

        # Check if goal is reached (heuristic is 0 iff goal is reached)
        # This check is also done by the search algorithm, but good practice for heuristic.
        if self.goals <= state:
            return 0

        # --- Heuristic Calculation ---

        # 1. Identify unserved children and their needs/locations
        unserved_children = {c for c in self.all_children if '(served ' + c + ')' not in state}
        num_unserved = len(unserved_children)

        # If no children are unserved, the goal should be met.
        # This check is redundant if self.goals <= state is checked first,
        # but included for clarity on the heuristic's logic base.
        if num_unserved == 0:
             return 0

        # Total number of sandwiches needed is equal to the number of unserved children
        needed_any = num_unserved

        # Count unserved children per location and allergy type
        unserved_at_loc = defaultdict(lambda: {'gf': 0, 'reg': 0})
        for child in unserved_children:
            loc = self.child_location.get(child)
            if loc: # Ensure location is known from static facts
                if self.child_allergy.get(child) == 'allergic':
                    unserved_at_loc[loc]['gf'] += 1
                else:
                    unserved_at_loc[loc]['reg'] += 1

        # 2. Count available sandwiches by type and location/status
        avail_gf_ontray_at_loc = defaultdict(int)
        avail_reg_ontray_at_loc = defaultdict(int)
        avail_gf_kitchen = 0
        avail_reg_kitchen = 0
        avail_existing_total = 0

        # Map trays to their current location
        tray_location = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1].startswith('tray'):
                tray, loc = parts[1], parts[2]
                tray_location[tray] = loc

        # Determine which sandwiches are GF based on the current state
        is_gf_sandwich = {}
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'no_gluten_sandwich':
                 is_gf_sandwich[parts[1]] = True

        # Count sandwiches on trays and in kitchen
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'ontray':
                s, t = parts[1], parts[2]
                avail_existing_total += 1
                sandwich_is_gf = is_gf_sandwich.get(s, False) # Default to False if no no_gluten_sandwich fact
                if t in tray_location:
                    if sandwich_is_gf:
                         avail_gf_ontray_at_loc[tray_location[t]] += 1
                    else:
                         avail_reg_ontray_at_loc[tray_location[t]] += 1

            elif parts[0] == 'at_kitchen_sandwich':
                s = parts[1]
                avail_existing_total += 1
                sandwich_is_gf = is_gf_sandwich.get(s, False)
                if sandwich_is_gf:
                    avail_gf_kitchen += 1
                else:
                    avail_reg_kitchen += 1

        # 3. Calculate Cost Components

        # Cost 1: Make sandwiches
        # Number of sandwiches that need to be made.
        cost_make = max(0, needed_any - avail_existing_total)

        # Cost 2: Put sandwiches on trays
        # Number of sandwiches that need to be moved from kitchen (or made) onto a tray.
        needed_on_tray = needed_any
        avail_on_tray = sum(avail_gf_ontray_at_loc.values()) + sum(avail_reg_ontray_at_loc.values())
        needed_from_kitchen_or_make = max(0, needed_on_tray - avail_on_tray)
        avail_in_kitchen = avail_gf_kitchen + avail_reg_kitchen
        # The sandwiches that need to get onto trays must come from the current kitchen stock
        # or from the sandwiches that will be made.
        cost_put_on_tray = min(avail_in_kitchen + cost_make, needed_from_kitchen_or_make)


        # Cost 3: Move trays
        # Number of sandwiches that are needed at each location but are not currently there on a tray.
        # This counts items needing delivery, not minimum tray moves.
        cost_move_tray = 0
        for loc in unserved_at_loc:
            needed_gf_at_loc = unserved_at_loc[loc]['gf']
            needed_reg_at_loc = unserved_at_loc[loc]['reg']
            avail_gf_at_loc = avail_gf_ontray_at_loc[loc]
            avail_reg_at_loc = avail_reg_ontray_at_loc[loc]

            # How many GF sandwiches are needed at this location that are not available locally?
            delivered_gf_at_loc = min(needed_gf_at_loc, avail_gf_at_loc)
            rem_needed_gf_at_loc = needed_gf_at_loc - delivered_gf_at_loc
            rem_avail_gf_at_loc = avail_gf_at_loc - delivered_gf_at_loc # Excess GF at loc

            # How many Reg sandwiches are needed at this location that are not available locally (Reg or excess GF)?
            delivered_reg_at_loc = min(needed_reg_at_loc, avail_reg_at_loc)
            rem_needed_reg_at_loc = needed_reg_at_loc - delivered_reg_at_loc
            # rem_avail_reg_at_loc = avail_reg_at_loc - delivered_reg_at_loc # Excess Reg at loc (not needed for calculation below)

            # Use excess GF at location for remaining needed Reg at location
            delivered_gf_for_reg_at_loc = min(rem_needed_reg_at_loc, rem_avail_gf_at_loc)
            rem_needed_reg_at_loc -= delivered_gf_for_reg_at_loc

            # Sandwiches needed at this location that are NOT currently on trays at this location
            needs_delivery_to_loc = rem_needed_gf_at_loc + rem_needed_reg_at_loc
            cost_move_tray += needs_delivery_to_loc # Add 1 for each sandwich needing delivery

        # Cost 4: Serve children
        # Each unserved child needs one serve action.
        cost_serve = num_unserved

        # Total heuristic is the sum of costs for each stage.
        total_cost = cost_make + cost_put_on_tray + cost_move_tray + cost_serve

        return total_cost
