from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Import math for max

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and remove leading/trailing parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return [] # Return empty list for invalid facts
    return fact[1:-1].split()

# Note: The 'match' helper function is not strictly needed by the final heuristic logic
# but is included as it was present in the example heuristics and is a useful utility.
# def match(fact, *args):
#     """
#     Check if a PDDL fact matches a given pattern.
#
#     - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
#     - `args`: The expected pattern (wildcards `*` allowed).
#     - Returns `True` if the fact matches the pattern, else `False`.
#     """
#     parts = get_parts(fact)
#     if len(parts) != len(args):
#         return False
#     return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children.
    It models the process as a pipeline where sandwiches need to be made,
    put on trays, moved to the correct locations, and finally served to children.
    The heuristic counts the number of 'items' (sandwiches or children) that
    need to pass through each stage of this pipeline.

    # Assumptions
    - All children waiting need to be served.
    - Sufficient bread, content, and 'notexist' sandwich objects are available
      in the kitchen if needed to make sandwiches.
    - Sufficient trays are available if needed (either in the kitchen or movable).
    - Each serve action serves one child.
    - Each put_on_tray action puts one sandwich on one tray.
    - Each move_tray action moves one tray (potentially with multiple sandwiches).
      The heuristic simplifies this by counting sandwiches needing movement.
    - Action costs are uniform (cost 1).

    # Heuristic Initialization
    - Identify all children mentioned in the goal, their waiting locations,
      and allergy status from static facts. If a child is in the goal but
      lacks static info, assume not allergic and location unknown (None).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic counts the number of items (sandwiches or children) that need to
    transition through the following stages to reach the goal state (all children served).
    The counts are cumulative, representing the number of items entering each stage.

    1.  **Serve Stage:** Each unserved child needs a 'serve' action.
        Count: `N_serve = number of unserved children`.

    2.  **Arrival Stage (Move Tray):** Each sandwich needed for an unserved child must arrive
        at the child's waiting location on a tray.
        Count: `N_move = number of sandwiches needed - number of sandwiches already on trays at the correct location and suitable for a child there`.
        This counts how many sandwiches still need to undergo a 'move_tray' step (or prior steps).

    3.  **Put-on-Tray Stage:** Each sandwich needing to arrive at a location must first
        be on a tray. If it's currently in the kitchen (`at_kitchen_sandwich`) or
        needs to be made, it must pass through the 'put_on_tray' stage.
        Count: `N_put = number of sandwiches needing arrival - number of sandwiches already on trays (at wrong locations)`.
        This counts how many sandwiches still need to undergo a 'put_on_tray' step (or prior steps).

    4.  **Make Stage:** Each sandwich needing to be put on a tray must first exist.
        If it's not already in the kitchen (`at_kitchen_sandwich`), it must be made.
        Count: `N_make = number of sandwiches needing put-on-tray - number of sandwiches already in the kitchen`.
        This counts how many sandwiches still need to undergo a 'make' step.

    The total heuristic is the sum of the counts for each stage:
    h = N_make + N_put + N_move + N_serve.

    Detailed steps:
    1. Identify all unserved children (children in goal not marked as served in state),
       their waiting locations, and allergy status (from static info).
    2. Count the total number of unserved children (`num_unserved`). If 0, return 0.
    3. Identify all sandwiches currently on trays and their trays.
    4. Identify the location of each tray.
    5. Identify all sandwiches currently in the kitchen (`at_kitchen_sandwich`).
    6. Determine suitability of each sandwich (GF or Any) based on `no_gluten_sandwich` facts in the state.
    7. Calculate `num_sandwiches_ready_to_serve`: Count unique sandwiches that are on a tray, where the tray is at a location `P` where an unserved child `C` is waiting, and the sandwich is suitable for `C`.
    8. Calculate `num_sandwiches_ontray_wrong_loc`: Count unique sandwiches that are on a tray but are *not* counted in `num_sandwiches_ready_to_serve`.
    9. Calculate `num_sandwiches_kitchen`: Count sandwiches `at_kitchen_sandwich`.
    10. Apply the formulas:
       `N_serve = num_unserved`
       `N_move = max(0, num_unserved - num_sandwiches_ready_to_serve)`
       `N_put = max(0, N_move - num_sandwiches_ontray_wrong_loc)`
       `N_make = max(0, N_put - num_sandwiches_kitchen)`
    11. Return `N_make + N_put + N_move + N_serve`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children
        who are part of the goal.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract static child information for children mentioned in the goal
        self.child_static_info = {} # child_name -> (waiting_location, is_allergic)
        goal_children = set()

        # Identify all children that need to be served from the goal
        for goal in self.goals:
             parts = get_parts(goal)
             if len(parts) == 2 and parts[0] == "served":
                 goal_children.add(parts[1])

        # Extract static info for these goal children
        static_child_allergy = {} # child -> is_allergic
        static_child_waiting = {} # child -> location

        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 2:
                predicate, child = parts
                if child in goal_children:
                    if predicate == "allergic_gluten":
                        static_child_allergy[child] = True
                    elif predicate == "not_allergic_gluten":
                        static_child_allergy[child] = False
            elif len(parts) == 3:
                 predicate, child, place = parts
                 if child in goal_children:
                     if predicate == "waiting":
                         static_child_waiting[child] = place

        # Consolidate static info for goal children
        for child in goal_children:
            location = static_child_waiting.get(child, None) # Location might not be in static, but in initial state
            is_allergic = static_child_allergy.get(child, False) # Assume not allergic if not specified
            self.child_static_info[child] = (location, is_allergic)

        self.goal_children = goal_children


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Extract relevant information from the current state ---
        unserved_children = set()
        sandwiches_kitchen = set()
        sandwiches_ontray_dict = {} # tray -> set of sandwiches
        tray_locations = {} # tray -> place
        sandwich_is_gf = {} # sandwich -> bool (True if gluten-free)
        children_waiting_at_loc = {} # place -> set of unserved children waiting at this place

        # Initialize children_waiting_at_loc for all places where goal children might wait
        all_potential_waiting_places = {loc for loc, _ in self.child_static_info.values() if loc is not None}
        # Also include places where trays are currently located in the state
        for fact in state:
             parts = get_parts(fact)
             if len(parts) == 3 and parts[0] == "at" and parts[1].startswith("tray"):
                 all_potential_waiting_places.add(parts[2])

        for place in all_potential_waiting_places:
             children_waiting_at_loc[place] = set()


        # Process state facts
        served_children_in_state = {get_parts(fact)[1] for fact in state if get_parts(fact) and get_parts(fact)[0] == "served" and len(get_parts(fact)) == 2}

        for child in self.goal_children:
            if child not in served_children_in_state:
                unserved_children.add(child)
                # Find current waiting location from state if available
                current_waiting_loc = None
                for fact in state:
                    parts = get_parts(fact)
                    if len(parts) == 3 and parts[0] == "waiting" and parts[1] == child:
                        current_waiting_loc = parts[2]
                        break

                # Use static location if state location not found (shouldn't happen in valid problems?)
                # Or update static info with state info if state is more dynamic
                # Let's rely on state for current waiting location if available
                if current_waiting_loc:
                     if current_waiting_loc in children_waiting_at_loc:
                         children_waiting_at_loc[current_waiting_loc].add(child)
                     else:
                         # This place wasn't in our initial set, add it
                         children_waiting_at_loc[current_waiting_loc] = {child}
                elif self.child_static_info.get(child, (None, False))[0] is not None:
                     # Child is unserved, but not waiting in state? Use static location if known.
                     # This case might indicate an invalid state representation or problem.
                     # For heuristic robustness, let's add it.
                     static_loc = self.child_static_info[child][0]
                     if static_loc in children_waiting_at_loc:
                          children_waiting_at_loc[static_loc].add(child)
                     else:
                          children_waiting_at_loc[static_loc] = {child}


        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]

            if predicate == "at_kitchen_sandwich" and len(parts) == 2:
                sandwiches_kitchen.add(parts[1])
            elif predicate == "ontray" and len(parts) == 3:
                sandwich, tray = parts[1], parts[2]
                if tray not in sandwiches_ontray_dict:
                    sandwiches_ontray_dict[tray] = set()
                sandwiches_ontray_dict[tray].add(sandwich)
            elif predicate == "at" and len(parts) == 3 and parts[1].startswith("tray"):
                tray, place = parts[1], parts[2]
                tray_locations[tray] = place
            elif predicate == "no_gluten_sandwich" and len(parts) == 2:
                sandwich_is_gf[parts[1]] = True

        # --- Heuristic Calculation ---

        num_unserved = len(unserved_children)

        # If no children need serving, the goal is reached.
        if num_unserved == 0:
            return 0

        # Helper to check if a sandwich is suitable for a child
        def is_suitable(sandwich_name, child_name):
            # Get allergy status from static info (or default)
            is_allergic_c = self.child_static_info.get(child_name, (None, False))[1]
            is_gf_s = sandwich_is_gf.get(sandwich_name, False)
            return (not is_allergic_c) or is_gf_s

        # 1. Count sandwiches ready to serve (on tray, at location, suitable)
        ready_sandwiches = set()
        # Iterate through places where unserved children are waiting
        for place, children_at_place in children_waiting_at_loc.items():
             if not children_at_place: continue # Skip places with no unserved children

             # Find trays at this location
             trays_at_place = {tray for tray, loc in tray_locations.items() if loc == place}

             for tray in trays_at_place:
                 for sandwich in sandwiches_ontray_dict.get(tray, set()):
                     # Check if this sandwich is suitable for *any* unserved child at this location
                     for child in children_at_place:
                         if is_suitable(sandwich, child):
                             ready_sandwiches.add(sandwich)
                             break # This sandwich is ready for *a* child at this location

        num_sandwiches_ready_to_serve = len(ready_sandwiches)

        # 2. Count sandwiches on trays at wrong locations
        ontray_sandwiches_set = set()
        for tray_sandwiches in sandwiches_ontray_dict.values():
             ontray_sandwiches_set.update(tray_sandwiches)

        wrong_loc_sandwiches_set = ontray_sandwiches_set - ready_sandwiches
        num_sandwiches_ontray_wrong_loc = len(wrong_loc_sandwiches_set)


        # 3. Count sandwiches in the kitchen
        num_sandwiches_kitchen = len(sandwiches_kitchen)

        # 4. Apply the pipeline formula
        N_serve = num_unserved
        N_move = max(0, num_unserved - num_sandwiches_ready_to_serve)
        N_put = max(0, N_move - num_sandwiches_ontray_wrong_loc)
        N_make = max(0, N_put - num_sandwiches_kitchen)

        # Total heuristic is the sum of items needing to pass through each stage
        total_cost = N_make + N_put + N_move + N_serve

        return total_cost

