import sys
# Add path to planner code if needed (assuming it's importable)
# sys.path.append("path/to/planner")
from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch

# Helper function to extract parts of a PDDL fact string like "(pred obj1 obj2)" -> ["pred", "obj1", "obj2"]
def get_parts(fact_str):
    """Safely extracts parts from a PDDL fact string, handling potential errors."""
    fact_str = fact_str.strip()
    if not fact_str.startswith("(") or not fact_str.endswith(")"):
        # Return empty list for malformed strings or comments
        return []
    # Split the content within parentheses
    return fact_str[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain childsnacks.

    # Summary
    This heuristic estimates the number of actions required to serve all children
    as specified in the goal state. It calculates the cost by summing estimates
    for the necessary 'make_sandwich', 'put_on_tray', 'move_tray', and 'serve_sandwich'
    actions required to satisfy the unserved children's needs. The heuristic aims
    to be informative for greedy best-first search and efficiently computable,
    but it is not necessarily admissible.

    # Assumptions
    - The heuristic assumes that sufficient ingredients (bread, content) and
      sandwich objects (`notexist`) are available to make all needed sandwiches.
      It does not model resource contention for ingredients or `notexist` sandwich objects.
    - It assumes gluten-free sandwiches are only used for allergic children,
      even though they could technically serve non-allergic children. This simplifies
      the calculation of needed sandwiches.
    - Tray movement costs are estimated based on the number of distinct target locations
      that need a sandwich delivered (i.e., don't already have one suitable on a tray there)
      and whether a tray needs to visit the kitchen when none are present there.
      It doesn't compute optimal routing or consider tray capacity (which isn't modeled in the domain anyway).
    - Each action type contributes 1 to the cost estimate.

    # Heuristic Initialization
    - Stores goal children: Extracts all `(served ?c)` predicates from the goal.
    - Stores static info: Parses static facts to build maps for:
        - Child allergies: `child -> is_allergic_gluten` (boolean)
        - Child waiting locations: `child -> place`
        - Set of all children and places defined in the static facts.

    # Step-By-Step Thinking for Computing Heuristic (v8 Logic)
    1.  **Identify Unserved Children:** Find all children `c` for which `(served c)` is a goal but not true in the current state. If no unserved goal children exist, the heuristic value H=0. Let `num_unserved` be the count of such children.
    2.  **Analyze Current State:** Parse the current state (`node.state`) to determine:
        - Location of each tray (`tray_locations`: map `tray -> location`).
        - Sandwiches currently in the kitchen, along with their gluten-free status (`kitchen_sandwiches`: map `sandwich -> is_gluten_free`).
        - Sandwiches currently on trays, along with the tray they are on and their gluten-free status (`ontray_sandwiches`: map `sandwich -> (tray, is_gluten_free)`).
    3.  **Calculate Needed Sandwiches (Cost_make):**
        - Count how many unserved children are allergic (`needed_gf`) and how many are not (`needed_reg`).
        - Count how many suitable sandwiches (GF/regular) are currently available, either in the kitchen or already on a tray (`available_gf_total`, `available_reg_total`).
        - Estimate the number of new GF sandwiches that must be made: `make_gf = max(0, needed_gf - available_gf_total)`.
        - Estimate the number of new regular sandwiches that must be made: `make_reg = max(0, needed_reg - available_reg_total)`.
        - The cost for making sandwiches is `Cost_make = make_gf + make_reg`.
    4.  **Calculate Put on Tray Actions (Cost_put):**
        - Count how many sandwiches are already on any tray: `num_on_tray = len(ontray_sandwiches)`.
        - Estimate the number of actions needed to put sandwiches onto trays. This is needed for children who are not yet served and don't already have their sandwich on a tray. `Cost_put = max(0, num_unserved - num_on_tray)`.
    5.  **Calculate Serve Actions (Cost_serve):**
        - Each unserved child will eventually require one `serve_sandwich` (or `_no_gluten`) action.
        - `Cost_serve = num_unserved`.
    6.  **Calculate Move Actions (Cost_move):**
        - Determine if a tray needs to visit the kitchen (`needs_kitchen_visit`). This is necessary if new sandwiches must be made (`cost_make > 0`) or if sandwiches must be put onto a tray (`cost_put > 0`) *and* there are suitable sandwiches available in the kitchen that could fulfill this 'put' need.
        - Check if any tray is currently located at the kitchen (`is_tray_at_kitchen`).
        - Identify the set of distinct locations (`locations_needing_visit`) where an unserved child is waiting but does *not* currently have a suitable sandwich (correct type) on *any* tray already present at that location.
        - The base move cost is the number of such locations needing a delivery: `len(locations_needing_visit)`.
        - If a kitchen visit is needed (`needs_kitchen_visit` is true), but no tray is currently at the kitchen (`is_tray_at_kitchen` is false), *and* there are trays somewhere else (`tray_locations` is not empty), then an additional move is required to bring a tray to the kitchen. Add 1 to the move cost in this specific case.
        - `Cost_move = len(locations_needing_visit)` + (1 if extra kitchen trip needed else 0).
    7.  **Total Heuristic Value:**
        - The final heuristic estimate is the sum of the costs calculated in steps 3-6:
          `H = Cost_make + Cost_put + Cost_serve + Cost_move`.
        - Ensure H is non-negative. If H calculates to 0 but the state is not actually a goal state (i.e., `num_unserved > 0`), return 1 as a minimum cost estimate for any non-goal state.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing static information from the task.
        Args:
            task: The planning task object containing static facts, goals, etc.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # --- Preprocess static information ---
        self.child_is_allergic = {} # Map: child_name -> bool
        self.child_waiting_location = {} # Map: child_name -> place_name
        self.children = set() # Set of all child names
        self.places = set() # Set of all place names (including 'kitchen')

        for fact_str in self.static_facts:
            parts = get_parts(fact_str)
            if not parts: continue # Skip empty or malformed lines
            pred = parts[0]

            try:
                if pred == "allergic_gluten" and len(parts) == 2:
                    child = parts[1]
                    self.child_is_allergic[child] = True
                    self.children.add(child)
                elif pred == "not_allergic_gluten" and len(parts) == 2:
                    child = parts[1]
                    self.child_is_allergic[child] = False
                    self.children.add(child)
                elif pred == "waiting" and len(parts) == 3:
                    child, place = parts[1], parts[2]
                    self.child_waiting_location[child] = place
                    self.children.add(child)
                    self.places.add(place)
                # Other static facts like no_gluten_bread/content are ignored by this heuristic
            except IndexError:
                # Log or ignore malformed static facts if necessary
                # print(f"Warning: Malformed static fact '{fact_str}'")
                pass

        # Identify children mentioned in the goal state
        self.goal_children = set()
        for goal_str in self.goals:
             parts = get_parts(goal_str)
             if not parts: continue
             # Check if the goal is '(served child_name)'
             if parts[0] == "served" and len(parts) == 2:
                 self.goal_children.add(parts[1])

        # Ensure the 'kitchen' constant is included in the set of places
        self.places.add("kitchen")


    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.
        Args:
            node: The state node in the search graph.
        Returns:
            An integer estimate of the remaining actions to reach the goal.
        """
        state = node.state

        # --- Identify unserved goal children ---
        served_children_in_state = set()
        for fact_str in state:
            parts = get_parts(fact_str)
            if not parts: continue
            if parts[0] == "served" and len(parts) == 2:
                 served_children_in_state.add(parts[1])

        # Find children who are goals but not yet served
        unserved_goal_children = self.goal_children - served_children_in_state

        # If all goal children are served, the state is a goal state (or equivalent)
        if not unserved_goal_children:
            return 0

        num_unserved = len(unserved_goal_children)

        # --- Analyze current state ---
        tray_locations = {} # Map: tray_name -> location_name
        kitchen_sandwiches_map = {} # Map: sandwich_name -> is_gluten_free (bool)
        ontray_sandwiches_map = {} # Map: sandwich_name -> (tray_name, is_gluten_free)
        # Temporary map to store GF status as it might appear before location facts
        sandwich_is_gf_map = {} # Map: sandwich_name -> True

        # Parse dynamic facts from the current state
        for fact_str in state:
            parts = get_parts(fact_str)
            if not parts: continue
            pred = parts[0]

            try:
                # Check for tray locations - crude check using 'tray' in name
                if pred == "at" and len(parts) == 3 and "tray" in parts[1]:
                     tray, loc = parts[1], parts[2]
                     tray_locations[tray] = loc
                # Check for gluten-free sandwiches
                elif pred == "no_gluten_sandwich" and len(parts) == 2:
                     sandwich_is_gf_map[parts[1]] = True
                # Check for sandwiches in the kitchen
                elif pred == "at_kitchen_sandwich" and len(parts) == 2:
                     sandwich = parts[1]
                     # Store sandwich, GF status will be resolved later
                     kitchen_sandwiches_map[sandwich] = False # Default to False
                # Check for sandwiches on trays
                elif pred == "ontray" and len(parts) == 3:
                     sandwich, tray = parts[1], parts[2]
                     # Store sandwich and tray, GF status will be resolved later
                     ontray_sandwiches_map[sandwich] = (tray, False) # Default to False
            except IndexError:
                 # Silently ignore malformed state facts
                 pass

        # Resolve GF status for sandwiches in kitchen and on trays
        for sandwich in list(kitchen_sandwiches_map.keys()): # Iterate over keys copy
             if sandwich in sandwich_is_gf_map:
                 kitchen_sandwiches_map[sandwich] = True

        for sandwich, (tray, _) in list(ontray_sandwiches_map.items()): # Iterate over items copy
             if sandwich in sandwich_is_gf_map:
                 ontray_sandwiches_map[sandwich] = (tray, True)

        # --- Calculate Cost_make ---
        needed_gf = 0
        needed_reg = 0
        # Count needs based on unserved children's allergies
        for child in unserved_goal_children:
            if self.child_is_allergic.get(child, False): # Default to False if info missing
                needed_gf += 1
            else:
                needed_reg += 1

        # Count available sandwiches of each type
        available_gf_on_tray = sum(1 for s, (t, is_gf) in ontray_sandwiches_map.items() if is_gf)
        available_reg_on_tray = sum(1 for s, (t, is_gf) in ontray_sandwiches_map.items() if not is_gf)
        available_gf_in_kitchen = sum(1 for s, is_gf in kitchen_sandwiches_map.items() if is_gf)
        available_reg_in_kitchen = sum(1 for s, is_gf in kitchen_sandwiches_map.items() if not is_gf)

        # Total available sandwiches by type
        available_gf_total = available_gf_on_tray + available_gf_in_kitchen
        available_reg_total = available_reg_on_tray + available_reg_in_kitchen

        # Calculate how many sandwiches need to be made
        make_gf = max(0, needed_gf - available_gf_total)
        make_reg = max(0, needed_reg - available_reg_total)
        cost_make = make_gf + make_reg

        # --- Calculate Cost_put ---
        num_on_tray = len(ontray_sandwiches_map)
        # Estimate put actions needed for children not covered by sandwiches already on trays
        cost_put = max(0, num_unserved - num_on_tray)

        # --- Calculate Cost_serve ---
        # Each unserved child needs one serve action
        cost_serve = num_unserved

        # --- Calculate Cost_move (v8 logic) ---
        # Determine if a kitchen visit is needed by a tray
        # Estimate how many sandwiches we *plan* to take from kitchen for putting on tray
        sandwiches_needed_from_kitchen_gf = max(0, needed_gf - available_gf_on_tray - make_gf)
        sandwiches_needed_from_kitchen_reg = max(0, needed_reg - available_reg_on_tray - make_reg)

        # Check if those sandwiches are *actually* available in the kitchen
        put_from_kitchen_possible = (sandwiches_needed_from_kitchen_gf > 0 and available_gf_in_kitchen > 0) or \
                                    (sandwiches_needed_from_kitchen_reg > 0 and available_reg_in_kitchen > 0)

        # Kitchen visit is needed if we make sandwiches OR if we put sandwiches AND can source them from kitchen
        needs_kitchen_visit = (cost_make > 0) or (cost_put > 0 and put_from_kitchen_possible)

        is_tray_at_kitchen = any(loc == 'kitchen' for loc in tray_locations.values())

        # Find distinct locations that still need a delivery
        locations_needing_visit = set()
        for child in unserved_goal_children:
            # Ensure child location info exists
            if child not in self.child_waiting_location: continue
            p = self.child_waiting_location[child]
            child_needs_gf = self.child_is_allergic.get(child, False)

            # Check if a suitable sandwich is already on a tray at this location 'p'
            has_sandwich_at_p = False
            for s, (t, is_gf_s) in ontray_sandwiches_map.items():
                 # Check if the tray 't' exists in current locations and is at 'p'
                 if t in tray_locations and tray_locations[t] == p:
                     # Check if the sandwich type is suitable for the child
                     if (child_needs_gf and is_gf_s) or (not child_needs_gf):
                         has_sandwich_at_p = True
                         break # Found a suitable sandwich at the location

            # If no suitable sandwich is already at the location, this location needs a visit
            if not has_sandwich_at_p:
                locations_needing_visit.add(p)

        # Base move cost is the number of distinct locations needing a delivery
        cost_move = len(locations_needing_visit)

        # Add cost for moving a tray to the kitchen if it's needed,
        # no tray is there, but trays exist elsewhere.
        if needs_kitchen_visit and not is_tray_at_kitchen and tray_locations: # Check tray_locations ensures trays exist
            cost_move += 1

        # --- Total Heuristic Value ---
        heuristic_value = cost_make + cost_put + cost_serve + cost_move
        # Ensure the heuristic value is always non-negative
        heuristic_value = max(0, heuristic_value)

        # Final sanity check: if the state is not a goal state, heuristic should be > 0
        # If the calculation resulted in 0 for a non-goal state, return 1 as a minimum cost.
        if heuristic_value == 0 and num_unserved > 0:
             return 1

        return heuristic_value
