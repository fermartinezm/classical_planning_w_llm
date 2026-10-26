# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic # This import should be present in the final environment

from fnmatch import fnmatch

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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

# Assuming childsnackHeuristic inherits from a base class named Heuristic
# class childsnackHeuristic(Heuristic): # This inheritance should be present in the final environment

# Define the class inheriting from the assumed base class
class childsnackHeuristic: # Replace with class childsnackHeuristic(Heuristic): in target environment
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the minimum number of actions required to serve all
    children. It counts the necessary 'serve', 'move_tray', 'put_on_tray', and
    'make_sandwich' actions based on the current state and the requirements
    of unserved children, greedily accounting for available resources.

    # Assumptions
    - The primary goal is to serve all children specified in the task goals.
    - Each unserved child requires one 'serve' action.
    - Trays might need to be moved to locations where unserved children are waiting.
      A single tray move can potentially serve multiple children at that location.
    - Sandwiches need to be made and put on trays before serving.
    - Gluten-allergic children require gluten-free sandwiches. Non-allergic
      children can eat any sandwich.
    - Ingredient availability for making sandwiches is ignored; it is assumed
      that if a 'notexist' sandwich object is available, a sandwich can be made
      (either regular or gluten-free, depending on ingredient availability which is simplified).
    - The heuristic does not attempt to find an optimal assignment of sandwiches
      or trays to children but provides a sum of estimated necessary actions.

    # Heuristic Initialization
    - Extracts the set of all children from the goal conditions.
    - Extracts static information about children's allergy status and waiting locations.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is calculated as the sum of estimated costs for four main types of actions:

    1.  **Serve Actions:** Each unserved child requires one 'serve' action.
        - Count the number of children who are in the initial list of all children
          (derived from goals) but are not currently marked as 'served' in the state.
        - Add this count to the total heuristic value.

    2.  **Move Tray Actions:** For each location where unserved children are waiting,
        if no tray is currently present at that location, a tray must be moved there.
        - Identify all locations where unserved children are waiting.
        - Identify the current location of all trays.
        - Count the number of unique locations with unserved children that do not
          currently have a tray.
        - Add this count to the total heuristic value. (This assumes one move action
          is sufficient to get *a* tray to the location, regardless of its starting point).

    3.  **Put on Tray Actions:** Each sandwich that is needed on a tray at a specific
        location to serve an unserved child, and is *not* already on a tray at that
        location, must be put on a tray.
        - For each location with unserved children, determine how many additional
          gluten-free and regular sandwiches are needed on trays at that location
          to satisfy the waiting children, considering sandwiches already on trays
          at that location.
        - Sum these additional needs across all locations to get the total number
          of sandwiches that need to be put on trays (and moved).
        - Add this total to the heuristic value.

    4.  **Make Sandwich Actions:** Sandwiches needed on trays that are not available
        as 'at_kitchen_sandwich' must be made from 'notexist' sandwich objects.
        - Calculate the total number of additional gluten-free and regular sandwiches
          needed on trays (from step 3).
        - Count the number of suitable sandwiches currently 'at_kitchen_sandwich'
          (gluten-free and regular).
        - Count the number of available 'notexist' sandwich objects.
        - Determine how many of the needed sandwiches cannot be sourced from
          'at_kitchen_sandwich' (prioritizing GF for GF needs, then GF for Reg needs,
          then Reg for Reg needs).
        - The remaining needed sandwiches must be made. The number of sandwiches
          that can actually be made is limited by the available 'notexist' objects.
        - Add the number of sandwiches that need to be made (capped by 'notexist' count)
          to the heuristic value.

    The final heuristic value is the sum of the counts from steps 1, 2, 3, and 4.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        # Assuming task object has .goals and .static attributes (frozensets of facts)
        # In the target environment, this class should inherit from Heuristic
        # super().__init__(task) # Call base class constructor if inheriting

        self.goals = task.goals
        self.static = task.static

        # Extract all children from the goals
        self.all_children = set()
        # Assuming goals is a frozenset of atomic facts like '(served child1)'
        if isinstance(self.goals, frozenset):
             for goal_fact in self.goals:
                 parts = get_parts(goal_fact)
                 if parts and parts[0] == 'served' and len(parts) == 2:
                     self.all_children.add(parts[1])
        # Note: If goals is a complex structure like ('and', (served child1), ...),
        # this parsing needs to be adjusted. Based on state/static examples,
        # assuming frozenset of atomic goal facts.


        # Extract static information about children
        self.child_allergy = {} # {child_name: True if allergic, False if not}
        self.child_waiting_location = {} # {child_name: location}
        # Note: gluten_free_bread/content from static are not used in the current simplified heuristic logic
        # self.gluten_free_bread = set()
        # self.gluten_free_content = set()

        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'allergic_gluten' and len(parts) == 2:
                self.child_allergy[parts[1]] = True
            elif parts[0] == 'not_allergic_gluten' and len(parts) == 2:
                 self.child_allergy[parts[1]] = False
            elif parts[0] == 'waiting' and len(parts) == 3:
                 self.child_waiting_location[parts[1]] = parts[2]
            # elif parts[0] == 'no_gluten_bread' and len(parts) == 2:
            #      self.gluten_free_bread.add(parts[1])
            # elif parts[0] == 'no_gluten_content' and len(parts) == 2:
            #      self.gluten_content.add(parts[1])


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state
        h = 0

        # --- Parse current state ---
        served_children = set()
        tray_locations = {} # {tray: location}
        gf_sandwiches_ontray = {} # {sandwich: tray}
        reg_sandwiches_ontray = {} # {sandwich: tray}
        gf_sandwiches_kitchen = set()
        reg_sandwiches_kitchen = set()
        notexist_sandwiches = set()
        sandwich_gluten_status = {} # {sandwich: True if GF, False if not}

        # First pass to get gluten status of sandwiches already made
        for fact in state:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == 'no_gluten_sandwich' and len(parts) == 2:
                 sandwich_gluten_status[parts[1]] = True

        # Second pass to get locations and status
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'served' and len(parts) == 2:
                served_children.add(parts[1])
            elif parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('tray'):
                tray_locations[parts[1]] = parts[2]
            elif parts[0] == 'ontray' and len(parts) == 3 and parts[1].startswith('sandw'):
                s, t = parts[1], parts[2]
                # Check gluten status, default to False if not explicitly marked GF
                if sandwich_gluten_status.get(s, False):
                    gf_sandwiches_ontray[s] = t
                else:
                    reg_sandwiches_ontray[s] = t
            elif parts[0] == 'at_kitchen_sandwich' and len(parts) == 2 and parts[1].startswith('sandw'):
                 s = parts[1]
                 # Check gluten status, default to False if not explicitly marked GF
                 if sandwich_gluten_status.get(s, False):
                     gf_sandwiches_kitchen.add(s)
                 else:
                     reg_sandwiches_kitchen.add(s)
            elif parts[0] == 'notexist' and len(parts) == 2 and parts[1].startswith('sandw'):
                 notexist_sandwiches.add(parts[1])

        # --- Calculate heuristic components ---

        # 1. Serve Actions
        unserved_children = self.all_children - served_children
        h += len(unserved_children)

        if not unserved_children:
            return h # Goal state

        # Get data for unserved children
        unserved_children_data = {}
        locations_with_unserved = set()
        for c in unserved_children:
             location = self.child_waiting_location.get(c)
             allergy = self.child_allergy.get(c, False) # Default to not allergic if status unknown
             if location is not None: # Only consider children whose waiting location is known
                 unserved_children_data[c] = {'location': location, 'allergic': allergy}
                 locations_with_unserved.add(location)

        # 2. Move Tray Actions
        trays_at_locations = set(tray_locations.values())
        locations_need_tray = locations_with_unserved - trays_at_locations
        h += len(locations_need_tray)

        # 3. & 4. Put on Tray and Make Sandwich Actions
        needed_gf_ontray_at_location = {} # {p: count}
        needed_reg_ontray_at_location = {} # {p: count}

        for p in locations_with_unserved:
            unserved_gf_at_p = [c for c, data in unserved_children_data.items() if data['location'] == p and data['allergic']]
            unserved_reg_at_p = [c for c, data in unserved_children_data.items() if data['location'] == p and not data['allergic']]

            # Count suitable sandwiches already on trays at this location
            avail_gf_ontray_at_p = sum(1 for s, t in gf_sandwiches_ontray.items() if tray_locations.get(t) == p)
            avail_reg_ontray_at_p = sum(1 for s, t in reg_sandwiches_ontray.items() if tray_locations.get(t) == p)

            # How many *more* sandwiches are needed on trays at this location?
            needed_gf_ontray_at_location[p] = max(0, len(unserved_gf_at_p) - avail_gf_ontray_at_p)
            needed_reg_ontray_at_location[p] = max(0, len(unserved_reg_at_p) - avail_reg_ontray_at_p)

        total_needed_gf_ontray = sum(needed_gf_ontray_at_location.values())
        total_needed_reg_ontray = sum(needed_reg_ontray_at_location.values())

        # Add cost for putting sandwiches on trays
        h += total_needed_gf_ontray + total_needed_reg_ontray

        # Determine how many of the needed sandwiches must be made
        avail_gf_kitchen_count = len(gf_sandwiches_kitchen)
        avail_reg_kitchen_count = len(reg_sandwiches_kitchen)
        avail_notexist_count = len(notexist_sandwiches)

        # Prioritize using kitchen sandwiches first
        rem_gf_needed = max(0, total_needed_gf_ontray - avail_gf_kitchen_count)
        rem_reg_needed = max(0, total_needed_reg_ontray - avail_reg_kitchen_count)

        # Use remaining kitchen GF sandwiches for regular needs
        # Calculate how many GF kitchen sandwiches are left after satisfying GF needs
        gf_kitchen_after_gf_needs = max(0, avail_gf_kitchen_count - total_needed_gf_ontray)
        used_kitchen_gf_for_reg = min(rem_reg_needed, gf_kitchen_after_gf_needs)
        rem_reg_needed -= used_kitchen_gf_for_reg

        # Sandwiches that still need to be sourced must be made
        total_to_make = rem_gf_needed + rem_reg_needed

        # Cap the number of sandwiches to make by the available 'notexist' slots
        actual_make = min(total_to_make, avail_notexist_count)

        # Add cost for making sandwiches
        h += actual_make

        return h
