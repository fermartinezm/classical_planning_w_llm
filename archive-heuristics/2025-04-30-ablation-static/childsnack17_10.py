from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle cases like '(at tray1 kitchen)' -> ['at', 'tray1', 'kitchen']
    # Handle cases like '(served child1)' -> ['served', 'child1']
    # Assumes fact is a string starting with '(' and ending with ')'
    if not fact or fact[0] != '(' or fact[-1] != ')':
        # Handle unexpected format, though valid PDDL facts should match
        return []
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    unserved children. It calculates a cost for each unserved child based
    on the "stage" of readiness of a suitable sandwich and tray at their
    location, and sums these costs. The stages represent the necessary steps:
    making the sandwich, getting it onto a tray, moving the tray to the
    child's location, and finally serving the child.

    # Assumptions
    - Each child requires exactly one sandwich of the correct type (gluten-free
      for allergic children, regular otherwise).
    - Trays have sufficient capacity to hold all sandwiches needed at a location.
    - Sufficient bread, content, and sandwich objects exist in the kitchen
      to make all needed sandwiches (i.e., the problem is solvable).
    - The cost of each action is 1.
    - The kitchen is a distinct location.
    - Unserved children remain in their initial waiting location.

    # Heuristic Initialization
    - Identify all children who need to be served from the task goals.
    - Identify all children who are allergic to gluten from static facts
      and initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost `h` to 0.
    2. Identify all children who need to be served based on the task goals.
    3. Pre-process the current state to efficiently query:
       - The location of each tray.
       - The location of each made sandwich (either in the kitchen or on a specific tray).
       - Which sandwiches are gluten-free.
       - The waiting location for each child.
    4. For each child `c` who needs to be served (identified in step 2):
        a. Check if `(served c)` is already true in the current state. If yes,
           this child contributes 0 to the heuristic. Continue to the next child.
        b. If the child `c` is not served, find their current waiting location `p`
           using the pre-processed child waiting locations.
        c. Determine if the child `c` is allergic to gluten using the pre-computed
           set of allergic children. This determines the required sandwich type
           (gluten-free or regular).
        d. Calculate the minimum cost for this child based on the state of a suitable sandwich:
           - Start with a base cost of 1 (for the final `serve` action).
           - Check if a suitable sandwich is already on a tray located *at the child's location p*.
             - If yes, the cost for this child is just 1 (only the `serve` action is needed from this point).
             - If no suitable sandwich is at `p` on a tray:
               - Add cost for getting a suitable sandwich onto a tray and moving it to `p`.
               - Check if *any* suitable sandwich is currently made (either in the kitchen or on any tray).
                 - If a suitable sandwich is made somewhere:
                   - Add 1 to the cost (representing the `put_on_tray` action if the sandwich is in the kitchen, or the start of the transport process if it's on a tray elsewhere).
                   - If the child's location `p` is not the 'kitchen', add another 1 (representing the `move_tray` action needed to bring the tray to the child).
                 - If no suitable sandwich is made yet:
                   - Add 1 to the cost (for the `make_sandwich` action).
                   - Add 1 to the cost (for the `put_on_tray` action, as the new sandwich will be in the kitchen).
                   - If the child's location `p` is not the 'kitchen', add another 1 (for the `move_tray` action).
        e. Add this minimum cost calculated for child `c` to the total heuristic `h`.
    5. Return the total heuristic cost `h`.

    This heuristic sums the estimated costs for each unserved child independently.
    It captures the main dependencies (make -> put -> move -> serve) but does not
    account for shared actions (like one tray movement serving multiple children
    at the same location), which means it might overestimate the true cost.
    However, this overestimation can still be effective for guiding a greedy
    best-first search by prioritizing states where more children are closer
    to being served.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal children and allergy info.
        """
        # Identify all children that need to be served (from goal facts)
        self.children_to_serve = {get_parts(goal)[1] for goal in task.goals if goal.startswith('(served ')}

        # Identify all children who are allergic to gluten (from static facts and initial state)
        self.allergic_children = {
            get_parts(fact)[1]
            for fact in task.static | task.initial_state
            if fact.startswith('(allergic_gluten ')
        }
        # Identify all children who are NOT allergic to gluten
        self.not_allergic_children = {
            get_parts(fact)[1]
            for fact in task.static | task.initial_state
            if fact.startswith('(not_allergic_gluten ')
        }
        # Store initial waiting locations as they are dynamic facts
        self.initial_waiting_locations = {
             get_parts(fact)[1]: get_parts(fact)[2]
             for fact in task.initial_state
             if fact.startswith('(waiting ')
        }


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state
        h = 0

        # Pre-compute locations of trays for quick lookup
        tray_locations = {}
        for fact in state:
            if fact.startswith('(at tray'):
                 parts = get_parts(fact)
                 if len(parts) == 3: # Ensure it's (at ?t ?p)
                    tray, loc = parts[1], parts[2]
                    tray_locations[tray] = loc

        # Pre-compute locations of sandwiches (kitchen or on tray)
        sandwich_locations = {} # Maps sandwich name to its location ('kitchen' or tray name)
        for fact in state:
            if fact.startswith('(at_kitchen_sandwich '):
                s = get_parts(fact)[1]
                sandwich_locations[s] = 'kitchen'
            elif fact.startswith('(ontray '):
                # Fact is (ontray ?s ?t)
                parts = get_parts(fact)
                if len(parts) == 3: # Ensure it's (ontray ?s ?t)
                    s, t = parts[1], parts[2]
                    sandwich_locations[s] = t # Store the tray name

        # Pre-compute which sandwiches are gluten-free
        gf_sandwiches = {get_parts(fact)[1] for fact in state if fact.startswith('(no_gluten_sandwich ')}

        # Pre-compute child waiting locations from the current state
        child_waiting_locations = {}
        for fact in state:
            if fact.startswith('(waiting '):
                parts = get_parts(fact)
                if len(parts) == 3: # Ensure it's (waiting ?c ?p)
                    child, loc = parts[1], parts[2]
                    child_waiting_locations[child] = loc


        # Iterate through all children that need to be served
        for child in self.children_to_serve:
            # If the child is already served, they contribute 0 to the heuristic
            if '(served ' + child + ')' in state:
                continue

            # Child is not served, calculate cost for this child
            child_cost = 1 # Base cost for the 'serve' action

            # Find the child's current waiting location
            # Use current state location, fall back to initial if not found (shouldn't happen in valid states)
            child_location = child_waiting_locations.get(child, self.initial_waiting_locations.get(child))

            if child_location is None:
                 # Should not happen in a valid problem instance where unserved children are waiting
                 # If it does, this child cannot be served, potentially infinite cost.
                 # For a greedy heuristic, we can return a large number or just skip,
                 # but assuming valid states where unserved children are waiting.
                 continue # Skip this child if location is unknown

            is_allergic = child in self.allergic_children

            # Check if a suitable sandwich is already on a tray at the child's location
            suitable_sandwich_at_loc = False
            for sandwich, s_loc in sandwich_locations.items():
                 # Check if sandwich type is suitable
                 is_gf_sandwich = sandwich in gf_sandwiches
                 is_suitable = (is_gf_sandwich and is_allergic) or (not is_gf_sandwich and not is_allergic)

                 if is_suitable:
                     # Check if sandwich is on a tray (s_loc is a tray name) and that tray is at child_location
                     if s_loc in tray_locations and tray_locations[s_loc] == child_location:
                         suitable_sandwich_at_loc = True
                         break # Found a suitable sandwich at the right location

            if not suitable_sandwich_at_loc:
                # Need to get a suitable sandwich onto a tray at this location
                # This adds cost for put_on_tray and/or move_tray

                # Check if a suitable sandwich is already made (either in kitchen or on any tray)
                suitable_sandwich_made = False
                for sandwich in sandwich_locations: # Iterate through sandwiches that are made
                    is_gf_sandwich = sandwich in gf_sandwiches
                    is_suitable = (is_gf_sandwich and is_allergic) or (not is_gf_sandwich and not is_allergic)
                    if is_suitable:
                        suitable_sandwich_made = True
                        break # Found a suitable sandwich that is made

                if suitable_sandwich_made:
                    # Sandwich exists, needs to get to location p on a tray
                    # Cost is 1 if p==kitchen (put), 2 if p!=kitchen (put+move or move)
                    if child_location == 'kitchen':
                         child_cost += 1 # put_on_tray
                    else:
                         child_cost += 2 # put_on_tray + move_tray OR move_tray
                else:
                    # Sandwich needs making, then put on tray, then move tray
                    child_cost += 1 # make_sandwich
                    if child_location == 'kitchen':
                         child_cost += 1 # put_on_tray
                    else:
                         child_cost += 2 # put_on_tray + move_tray

            h += child_cost

        return h
