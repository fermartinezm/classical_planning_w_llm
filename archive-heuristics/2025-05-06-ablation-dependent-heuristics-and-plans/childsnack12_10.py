from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved children.
    It does this by summing the estimated cost for each unserved child, based on the
    current state of available sandwiches and resources. It prioritizes using sandwiches
    that are closer to being served (already on a tray at the child's location, then
    on a tray at the kitchen, then at the kitchen raw, then needing to be made).
    Gluten-free requirements are considered, and non-allergic children can accept
    gluten-free sandwiches. Resource consumption is modeled greedily.

    # Assumptions
    - All children listed in the goal need to be served.
    - Children wait at fixed locations specified by 'waiting' facts in the state.
    - Trays can hold multiple sandwiches (implicitly, as we count sandwiches on trays).
    - Tray movement cost is simplified; the primary cost difference is based on the
      sandwich's location (on tray at child's place, on tray at kitchen, at kitchen raw).
    - Making a sandwich requires one bread, one content, and one 'notexist' slot.
    - Enough trays are available at the kitchen for putting sandwiches on (implicitly).
    - Problems are solvable with the given resources (a large penalty is added otherwise).

    # Heuristic Initialization
    - Identify which children are allergic from static facts.
    - Identify which bread and content are gluten-free from static facts.
    - Store these sets for quick lookup.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify all children that need to be served (those in the goal not yet served).
    2.  For each unserved child, determine their waiting location and required sandwich type (gluten-free if allergic, regular otherwise).
    3.  Count the available sandwiches by type (gluten-free/regular) and their current state/location:
        -   On a tray at a child's waiting location (`cost=1` to serve).
        -   On a tray at the kitchen (`cost=2`: move tray + serve).
        -   At the kitchen (raw, not on a tray) (`cost=3`: put on tray + move tray + serve).
    4.  Count the available ingredients (gluten-free bread/content, regular bread/content) and 'notexist' sandwich slots at the kitchen to determine how many new sandwiches of each type (gluten-free/regular) can potentially be made (`cost=4`: make + put + move + serve). Specifically, count GF makeable and total makeable.
    5.  Initialize the total heuristic cost to 0.
    6.  Iterate through the list of unserved children. For each child:
        -   Determine the required sandwich type (GF or Reg).
        -   Find the cheapest available source for a suitable sandwich based on the counts from step 3 and 4, prioritizing sources closer to the child's location.
        -   Add the corresponding cost (1, 2, 3, or 4) to the total heuristic.
        -   Decrement the count of the consumed sandwich/resource from the available pools to simulate resource usage.
        -   If no suitable sandwich is available or can be made with current resources, add a large penalty (e.g., 100) to the heuristic.
    7.  Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about allergies
        and gluten-free ingredients.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract static information
        self.allergic_children = {get_parts(f)[1] for f in self.static if get_parts(f)[0] == 'allergic_gluten'}
        self.no_gluten_bread_static = {get_parts(f)[1] for f in self.static if get_parts(f)[0] == 'no_gluten_bread'}
        self.no_gluten_content_static = {get_parts(f)[1] for f in self.static if get_parts(f)[0] == 'no_gluten_content'}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children and their needs
        all_children_in_goal = {get_parts(g)[1] for g in self.goals if get_parts(g)[0] == 'served'}
        served_children = {get_parts(f)[1] for f in state if get_parts(f)[0] == 'served'}

        unserved_children_needs = [] # List of (child_name, location, type)

        # Get current waiting locations from the state
        current_waiting_locations = {get_parts(f)[1]: get_parts(f)[2] for f in state if get_parts(f)[0] == 'waiting'}

        for child in all_children_in_goal:
            if child not in served_children:
                location = current_waiting_locations.get(child)
                if location is None:
                     # Child is in goal but not waiting in current state - should not happen in valid states
                     # Treat as unserveable for heuristic purposes.
                     unserved_children_needs.append((child, None, 'Penalty'))
                     continue

                sandwich_type = 'GF' if child in self.allergic_children else 'Reg'
                unserved_children_needs.append((child, location, sandwich_type))

        # If no children need serving, goal is reached. Heuristic is 0.
        if not unserved_children_needs:
            return 0

        # 2. Count available sandwiches by type and state
        # Need to be careful: counts are pools, not specific instances.
        # A GF sandwich can serve a Reg need, but not vice-versa.

        # Count sandwiches on trays at specific locations (excluding kitchen)
        gf_ontray_at_loc = {} # {location: count}
        reg_ontray_at_loc = {} # {location: count}

        # Count sandwiches on trays at kitchen
        gf_ontray_kitchen = 0
        reg_ontray_kitchen = 0

        # Count sandwiches at kitchen (raw)
        gf_at_kitchen = 0
        reg_at_kitchen = 0

        # Identify existing sandwiches and their GF status
        sandwich_is_gf_state = {get_parts(f)[1] for f in state if get_parts(f)[0] == 'no_gluten_sandwich'}

        # Identify tray locations
        tray_locations = {get_parts(f)[1]: get_parts(f)[2] for f in state if get_parts(f)[0] == 'at'}

        # Count sandwiches on trays
        for fact in state:
            if get_parts(fact)[0] == 'ontray':
                s, t = get_parts(fact)[1:]
                tray_loc = tray_locations.get(t)
                if tray_loc: # Tray must have a location
                    s_type = 'GF' if s in sandwich_is_gf_state else 'Reg'
                    if tray_loc != 'kitchen':
                        if s_type == 'GF':
                            gf_ontray_at_loc[tray_loc] = gf_ontray_at_loc.get(tray_loc, 0) + 1
                        else:
                            reg_ontray_at_loc[tray_loc] = reg_ontray_at_loc.get(tray_loc, 0) + 1
                    else: # Tray is at kitchen
                        if s_type == 'GF':
                            gf_ontray_kitchen += 1
                        else:
                            reg_ontray_kitchen += 1

        # Count sandwiches at kitchen (raw)
        for fact in state:
            if get_parts(fact)[0] == 'at_kitchen_sandwich':
                s = get_parts(fact)[1]
                s_type = 'GF' if s in sandwich_is_gf_state else 'Reg'
                if s_type == 'GF':
                    gf_at_kitchen += 1
                else:
                    reg_at_kitchen += 1

        # 3. Count makeable sandwiches (potential)
        # Count available ingredients at kitchen
        num_gf_bread_avail = sum(1 for f in state if get_parts(f)[0] == 'at_kitchen_bread' and get_parts(f)[1] in self.no_gluten_bread_static)
        num_reg_bread_avail = sum(1 for f in state if get_parts(f)[0] == 'at_kitchen_bread' and get_parts(f)[1] not in self.no_gluten_bread_static)
        num_gf_content_avail = sum(1 for f in state if get_parts(f)[0] == 'at_kitchen_content' and get_parts(f)[1] in self.no_gluten_content_static)
        num_reg_content_avail = sum(1 for f in state if get_parts(f)[0] == 'at_kitchen_content' and get_parts(f)[1] not in self.no_gluten_content_static)
        num_notexist_slots = sum(1 for f in state if get_parts(f)[0] == 'notexist')

        # How many GF sandwiches can be made? Requires GF bread, GF content, notexist slot.
        gf_makeable_count = min(num_gf_bread_avail, num_gf_content_avail, num_notexist_slots)

        # How many *total* sandwiches can be made? Requires any bread, any content, notexist slot.
        total_bread_avail = num_gf_bread_avail + num_reg_bread_avail
        total_content_avail = num_gf_content_avail + num_reg_content_avail
        total_makeable_any_type = min(total_bread_avail, total_content_avail, num_notexist_slots)


        # Use copies of counts for consumption simulation
        current_gf_ontray_at_loc = {loc: count for loc, count in gf_ontray_at_loc.items()}
        current_reg_ontray_at_loc = {loc: count for loc, count in reg_ontray_at_loc.items()}
        current_gf_ontray_kitchen = gf_ontray_kitchen
        current_reg_ontray_kitchen = reg_ontray_kitchen
        current_gf_at_kitchen = gf_at_kitchen
        current_reg_at_kitchen = reg_at_kitchen
        current_gf_makeable = gf_makeable_count
        current_total_makeable_any_type = total_makeable_any_type


        # 4. Calculate heuristic
        h = 0

        # Process GF needs first
        gf_needs = [(c, l) for c, l, t in unserved_children_needs if t == 'GF']
        for child, location in gf_needs:
            served_by_option = False

            # Option 1: GF Sandwich on tray at child's location (Cost 1)
            if current_gf_ontray_at_loc.get(location, 0) > 0:
                h += 1
                current_gf_ontray_at_loc[location] -= 1
                served_by_option = True
            # Option 2: GF Sandwich on tray at kitchen (Cost 2)
            elif current_gf_ontray_kitchen > 0:
                h += 2
                current_gf_ontray_kitchen -= 1
                served_by_option = True
            # Option 3: GF Sandwich at kitchen raw (Cost 3)
            elif current_gf_at_kitchen > 0:
                h += 3
                current_gf_at_kitchen -= 1
                served_by_option = True
            # Option 4: Make a GF sandwich (Cost 4)
            elif current_gf_makeable > 0:
                h += 4
                current_gf_makeable -= 1
                current_total_makeable_any_type -= 1 # Making a GF uses up a total slot
                served_by_option = True
            else:
                # Cannot serve this GF child with available/makeable GF sandwiches
                h += 100 # Penalty

        # Process Reg needs next
        reg_needs = [(c, l) for c, l, t in unserved_children_needs if t == 'Reg']
        for child, location in reg_needs:
            served_by_option = False

            # Option 1: Reg or GF Sandwich on tray at child's location (Cost 1)
            if current_reg_ontray_at_loc.get(location, 0) > 0:
                h += 1
                current_reg_ontray_at_loc[location] -= 1
                served_by_option = True
            elif current_gf_ontray_at_loc.get(location, 0) > 0:
                h += 1
                current_gf_ontray_at_loc[location] -= 1
                served_by_option = True
            # Option 2: Reg or GF Sandwich on tray at kitchen (Cost 2)
            elif current_reg_ontray_kitchen > 0:
                h += 2
                current_reg_ontray_kitchen -= 1
                served_by_option = True
            elif current_gf_ontray_kitchen > 0:
                h += 2
                current_gf_ontray_kitchen -= 1
                served_by_option = True
            # Option 3: Reg or GF Sandwich at kitchen raw (Cost 3)
            elif current_reg_at_kitchen > 0:
                h += 3
                current_reg_at_kitchen -= 1
                served_by_option = True
            elif current_gf_at_kitchen > 0:
                h += 3
                current_gf_at_kitchen -= 1
                served_by_option = True
            # Option 4: Make a Reg or GF sandwich (Cost 4)
            # Can make *any* sandwich if ingredients and slot exist
            elif current_total_makeable_any_type > 0:
                 h += 4
                 current_total_makeable_any_type -= 1
                 served_by_option = True
            else:
                # Cannot serve this Reg child with available/makeable sandwiches
                h += 100 # Penalty

        return h
