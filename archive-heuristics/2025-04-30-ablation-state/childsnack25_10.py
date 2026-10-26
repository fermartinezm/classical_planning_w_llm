from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions (can be defined outside the class)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    This heuristic estimates the number of actions required to serve all waiting
    children. It breaks down the process into four main stages for each unserved
    child's need: making a suitable sandwich, putting a sandwich on a tray,
    moving a tray to the child's location, and finally serving the child. The
    heuristic sums the estimated costs for these stages across all unserved
    children, taking into account shared resources like trays and the gluten
    allergy constraint.

    # Assumptions
    - The goal is to serve all children who are initially in a 'waiting' state.
    - Allergy information and gluten-free properties of ingredients are static.
    - Ingredients and sandwich objects are assumed to be sufficient to make
      all necessary sandwiches (this simplifies the 'make' cost).
    - Trays are assumed to be available for 'put_on_tray' and 'move_tray' actions
      if they exist in the state, without complex reasoning about their current
      occupancy or exact location (beyond being at a required waiting place).
    - The cost of moving a tray to a location is always 1, regardless of the
      starting location (as long as it's not already at the destination).

    # Heuristic Initialization
    - Extracts static facts from the task, specifically identifying which
      children are allergic or not allergic, and which bread/content types
      are gluten-free. This information is used to determine sandwich requirements.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic estimates the remaining cost as follows:

    1.  **Identify Unserved Children:** Count the number of children who are
        currently in a 'waiting' state but not yet 'served'. Separate them
        by allergy status (allergic needing GF, non-allergic needing any).
        Let N_U be the total number of unserved children. If N_U is 0, the
        heuristic is 0 (goal state).

    2.  **Estimate 'Make Sandwich' Cost:**
        - Count how many gluten-free (GF) sandwiches are needed for allergic
          children (N_Req_GF).
        - Count how many non-GF sandwiches (can be regular or GF) are needed
          for non-allergic children (N_Req_Any).
        - Count currently available made sandwiches (at kitchen or on trays),
          distinguishing GF and regular ones (N_Avail_GF_Current, N_Avail_Reg_Current).
        - Calculate the number of new GF sandwiches that *must* be made:
          `make_gf = max(0, N_Req_GF - N_Avail_GF_Current)`.
        - Calculate the number of new 'any' sandwiches that *must* be made
          for non-allergic children, after using any surplus available GF
          sandwiches: `make_any = max(0, N_Req_Any - (max(0, N_Avail_GF_Current - N_Req_GF) + N_Avail_Reg_Current))`.
        - The cost is the sum: `cost_make = make_gf + make_any`.

    3.  **Estimate 'Put on Tray' Cost:**
        - We need N_U sandwiches to eventually be on trays.
        - Count sandwiches currently on trays (N_Avail_OnTray_Current).
        - The cost is the number of sandwiches that still need to be put on
          trays: `cost_put_on_tray = max(0, N_U - N_Avail_OnTray_Current)`.

    4.  **Estimate 'Move Tray' Cost:**
        - Identify all distinct locations where unserved children are waiting.
        - Identify all distinct locations where trays are currently present.
        - The cost is the number of waiting locations that do *not* currently
          have a tray: `cost_move_tray = len(waiting_locations) - len(waiting_locations.intersection(tray_locations_in_state))`.

    5.  **Estimate 'Serve' Cost:**
        - Each unserved child requires one 'serve' action.
        - The cost is simply the number of unserved children: `cost_serve = N_U`.

    The total heuristic value is the sum of these estimated costs:
    `cost_make + cost_put_on_tray + cost_move_tray + cost_serve`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.
        """
        self.goals = task.goals # Keep goals for potential future use, though not directly used in this sum heuristic

        # Extract static information
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.no_gluten_bread_types = set()
        self.no_gluten_content_types = set()

        for fact_str in task.static:
            parts = get_parts(fact_str)
            if not parts: continue # Skip malformed facts
            if parts[0] == 'allergic_gluten' and len(parts) == 2:
                self.allergic_children.add(parts[1])
            elif parts[0] == 'not_allergic_gluten' and len(parts) == 2:
                self.not_allergic_children.add(parts[1])
            # Note: no_gluten_bread/content are properties of the ingredient objects,
            # not necessarily static facts in the initial state unless explicitly listed.
            # The example static block *does* list them, so we extract them.
            elif parts[0] == 'no_gluten_bread' and len(parts) == 2:
                self.no_gluten_bread_types.add(parts[1])
            elif parts[0] == 'no_gluten_content' and len(parts) == 2:
                self.no_gluten_content_types.add(parts[1])

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Collect current state facts into usable data structures
        waiting_children_in_state = {} # child -> place
        served_children_in_state = set()
        sandwiches_made_current = set() # sandwiches that exist (at_kitchen or ontray)
        ontray_in_state = set() # sandwiches that are on trays
        no_gluten_sandwich_in_state = set() # sandwiches that are GF
        tray_locations_in_state = set() # places where trays are located

        for fact_str in state:
            parts = get_parts(fact_str)
            if not parts: continue # Skip malformed facts

            if parts[0] == 'waiting' and len(parts) == 3:
                waiting_children_in_state[parts[1]] = parts[2]
            elif parts[0] == 'served' and len(parts) == 2:
                served_children_in_state.add(parts[1])
            elif parts[0] == 'at_kitchen_sandwich' and len(parts) == 2:
                sandwiches_made_current.add(parts[1])
            elif parts[0] == 'ontray' and len(parts) == 3:
                sandwiches_made_current.add(parts[1]) # Also a made sandwich
                ontray_in_state.add(parts[1]) # This sandwich is on a tray
            elif parts[0] == 'no_gluten_sandwich' and len(parts) == 2:
                no_gluten_sandwich_in_state.add(parts[1])
            elif parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('tray'): # Assuming 'at' only applies to trays and places
                 tray_locations_in_state.add(parts[2]) # Store the location

        # 1. Identify Unserved Children and Requirements
        unserved_children_details = {} # child -> {'place': p, 'allergic': bool}
        waiting_locations = set() # places where unserved children wait

        for child, place in waiting_children_in_state.items():
            if child not in served_children_in_state:
                unserved_children_details[child] = {
                    'place': place,
                    'allergic': child in self.allergic_children
                }
                waiting_locations.add(place)

        N_U = len(unserved_children_details)
        if N_U == 0:
            return 0 # Goal reached

        N_Req_GF = sum(1 for details in unserved_children_details.values() if details['allergic'])
        N_Req_Any = N_U - N_Req_GF # Non-allergic children need any sandwich

        # 2. Estimate 'Make Sandwich' Cost
        Avail_GF_Current = {s for s in sandwiches_made_current if s in no_gluten_sandwich_in_state}
        Avail_Reg_Current = sandwiches_made_current - Avail_GF_Current

        N_Avail_GF_Current = len(Avail_GF_Current)
        N_Avail_Reg_Current = len(Avail_Reg_Current)

        needed_gf_to_make = max(0, N_Req_GF - N_Avail_GF_Current)

        # Calculate how many GF sandwiches are left after potentially satisfying allergic needs
        rem_gf_avail_for_any = max(0, N_Avail_GF_Current - N_Req_GF)

        # Calculate how many 'any' sandwiches are still needed for non-allergic
        # after using available regular and surplus GF sandwiches
        needed_any_to_make = max(0, N_Req_Any - (rem_gf_avail_for_any + N_Avail_Reg_Current))

        cost_make = needed_gf_to_make + needed_any_to_make

        # 3. Estimate 'Put on Tray' Cost
        N_Avail_OnTray_Current = len(ontray_in_state)
        cost_put_on_tray = max(0, N_U - N_Avail_OnTray_Current)

        # 4. Estimate 'Move Tray' Cost
        # Count locations where unserved children are waiting but no tray is present
        Locs_Need_Tray = waiting_locations - tray_locations_in_state
        cost_move_tray = len(Locs_Need_Tray)

        # 5. Estimate 'Serve' Cost
        cost_serve = N_U

        # Total heuristic is the sum of estimated costs for each stage
        total_heuristic = cost_make + cost_put_on_tray + cost_move_tray + cost_serve

        return total_heuristic
