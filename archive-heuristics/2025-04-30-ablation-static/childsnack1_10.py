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
    This heuristic estimates the number of actions required to serve all waiting
    children by summing up the estimated counts of necessary 'serve',
    'make_sandwich', 'put_on_tray', and 'move_tray' actions. It provides a
    non-admissible estimate designed to guide a greedy best-first search.

    # Assumptions
    - The problem is solvable (enough ingredients and sandwich names exist
      eventually to make all needed sandwiches).
    - Resource contention (multiple children needing the same tray or ingredient)
      is simplified by summing up requirements rather than tracking specific assignments.
    - Tray movement cost is estimated based on trays being in locations where
      they are not currently needed by children.

    # Heuristic Initialization
    - Extracts static information about which children are allergic from the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize heuristic value `h = 0`.
    2. Identify all unserved children by checking the goal conditions against the current state.
       For each unserved child, determine their waiting place and allergy status.
       Count the total number of unserved children (`N_unserved`).
       Add `N_unserved` to `h` (each unserved child requires at least one 'serve' action).
       If `N_unserved` is 0, the state is a goal state, and the heuristic is 0.
    3. Count available ingredients in the kitchen (regular and gluten-free bread/content).
       Count available unused sandwich names (`notexist` predicate).
    4. Count existing sandwiches (those not having the `notexist` predicate),
       determine their type (gluten-free or regular), and their current location
       (in the kitchen or on a specific tray).
    5. Estimate 'make_sandwich' actions needed:
       Count unserved allergic children (`N_allergic_unserved`).
       Count available suitable GF sandwiches (those already made, either on a tray or in the kitchen).
       Number of GF sandwiches that still need to be made = `max(0, N_allergic_unserved - available_gf_sandwiches)`.
       Count unserved non-allergic children (`N_regular_unserved`).
       Count available suitable regular sandwiches (those already made, either on a tray or in the kitchen).
       Number of regular sandwiches that still need to be made = `max(0, N_regular_unserved - available_regular_sandwiches)`.
       Add (GF sandwiches to make + regular sandwiches to make) to `h`.
       (This step simplifies by assuming ingredients and unused names are sufficient for the calculated needs).
    6. Estimate 'put_on_tray' actions needed:
       Count suitable sandwiches that are currently `at_kitchen_sandwich`. A sandwich in the kitchen is considered 'suitable' if there is at least one unserved child who could eat it (i.e., GF for an allergic child if any exist, or regular for a non-allergic child if any exist).
       Add this count to `h`. (Each such sandwich needs to be put on a tray).
    7. Estimate 'move_tray' actions needed:
       Identify the set of locations where unserved children are waiting.
       Count trays that are currently located neither in the kitchen nor at any of the locations where unserved children are waiting. These trays are considered 'out of place' and might need to be moved. Add this count to `h`.
       Additionally, if there are unserved children waiting at the kitchen and no tray is currently located at the kitchen, add 1 to `h` (representing the need to move at least one tray to the kitchen).

    The total heuristic value is the sum of these estimated action counts.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts about child allergies.
        """
        self.goals = task.goals # Goal conditions are needed to identify unserved children
        static_facts = task.static # Static facts like allergies

        # Store allergy status for children
        self.allergic_children = set()
        self.not_allergic_children = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "allergic_gluten":
                self.allergic_children.add(parts[1])
            elif parts[0] == "not_allergic_gluten":
                 self.not_allergic_children.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # --- Step 2: Identify unserved children and count serve actions ---
        unserved_children_details = {} # {child_name: {'place': place, 'allergic': bool}}
        n_unserved = 0
        n_allergic_unserved = 0
        n_regular_unserved = 0

        # Find all children mentioned in the goals (assuming all children in goals need serving)
        all_children_in_goals = set()
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == "served":
                 all_children_in_goals.add(parts[1])

        for child in all_children_in_goals:
            if f"(served {child})" not in state:
                n_unserved += 1
                is_allergic = child in self.allergic_children
                place = None
                # Find waiting place for this child in the current state
                for fact in state:
                    if match(fact, "waiting", child, "*"):
                        place = get_parts(fact)[2]
                        break
                # Store details, even if place is None (though problems usually have waiting place)
                unserved_children_details[child] = {'place': place, 'allergic': is_allergic}
                if is_allergic:
                    n_allergic_unserved += 1
                else:
                    n_regular_unserved += 1

        # If no unserved children, goal is reached
        if n_unserved == 0:
            return 0

        h = n_unserved # Cost for the final 'serve' action for each child

        # --- Step 3 & 4: Count available resources and existing sandwiches ---
        # Count ingredients in kitchen
        n_gf_bread_kitchen = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*") and match(fact, "no_gluten_bread", get_parts(fact)[1]))
        n_reg_bread_kitchen = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*") and not match(fact, "no_gluten_bread", get_parts(fact)[1]))
        n_gf_content_kitchen = sum(1 for fact in state if match(fact, "at_kitchen_content", "*") and match(fact, "no_gluten_content", get_parts(fact)[1]))
        n_reg_content_kitchen = sum(1 for fact in state if match(fact, "at_kitchen_content", "*") and not match(fact, "no_gluten_content", get_parts(fact)[1]))
        n_notexist_sandwiches = sum(1 for fact in state if match(fact, "notexist", "*"))

        # Count existing sandwiches and their type/location
        existing_sandwiches = {} # {sandwich_name: {'gf': bool, 'location': 'kitchen' or tray_name}}
        n_gf_avail = 0 # GF sandwiches already made (in kitchen or on tray)
        n_reg_avail = 0 # Regular sandwiches already made (in kitchen or on tray)

        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                s_name = get_parts(fact)[1]
                is_gf = f"(no_gluten_sandwich {s_name})" in state
                existing_sandwiches[s_name] = {'gf': is_gf, 'location': 'kitchen'}
                if is_gf:
                    n_gf_avail += 1
                else:
                    n_reg_avail += 1
            elif match(fact, "ontray", "*", "*"):
                 s_name = get_parts(fact)[1]
                 t_name = get_parts(fact)[2]
                 is_gf = f"(no_gluten_sandwich {s_name})" in state
                 existing_sandwiches[s_name] = {'gf': is_gf, 'location': t_name}
                 if is_gf:
                     n_gf_avail += 1
                 else:
                     n_reg_avail += 1

        # --- Step 5: Estimate make_sandwich actions ---
        # Number of GF sandwiches needed that don't exist yet
        needed_make_gf = max(0, n_allergic_unserved - n_gf_avail)
        # Number of regular sandwiches needed that don't exist yet
        needed_make_reg = max(0, n_regular_unserved - n_reg_avail)

        # Note: This heuristic simplifies by assuming enough ingredients and
        # notexist sandwich names are available to make the needed quantity.
        # A more complex heuristic could check resource limits here.

        h += needed_make_gf + needed_make_reg

        # --- Step 6: Estimate put_on_tray actions ---
        # Count suitable sandwiches that are in the kitchen and need to be put on a tray.
        # A sandwich in the kitchen is 'suitable' if it's GF and an allergic child needs one,
        # or if it's regular and a regular child needs one. We only count if there's
        # at least one child who could potentially need this type.
        n_kitchen_suitable_sandwiches = 0
        for s_name, s_info in existing_sandwiches.items():
            if s_info['location'] == 'kitchen':
                is_suitable_for_unserved = False
                if s_info['gf'] and n_allergic_unserved > 0:
                    is_suitable_for_unserved = True
                elif not s_info['gf'] and n_regular_unserved > 0:
                     is_suitable_for_unserved = True
                if is_suitable_for_unserved:
                    n_kitchen_suitable_sandwiches += 1

        h += n_kitchen_suitable_sandwiches


        # --- Step 7: Estimate move_tray actions ---
        # Count trays at each location
        trays_at_location = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj_name = get_parts(fact)[1]
                loc_name = get_parts(fact)[2]
                # Check if the object is a tray (assuming tray names start with 'tray')
                if obj_name.startswith('tray'):
                     trays_at_location[obj_name] = loc_name

        # Identify locations where unserved children are waiting
        locations_with_children = set()
        children_at_kitchen = False
        for child, info in unserved_children_details.items():
             if info['place']: # Ensure child has a waiting place
                  locations_with_children.add(info['place'])
                  if info['place'] == 'kitchen':
                       children_at_kitchen = True

        # Count trays that are neither in the kitchen nor at a location with children
        n_trays_out_of_place = 0
        for tray_name, loc in trays_at_location.items():
             if loc != 'kitchen' and loc not in locations_with_children:
                  n_trays_out_of_place += 1
        h += n_trays_out_of_place

        # If children are waiting at the kitchen but no tray is there, need one move
        tray_at_kitchen = False
        for tray_loc in trays_at_location.values():
             if tray_loc == 'kitchen':
                  tray_at_kitchen = True
                  break
        if children_at_kitchen and not tray_at_kitchen:
             h += 1 # Need to move a tray to the kitchen


        return h
