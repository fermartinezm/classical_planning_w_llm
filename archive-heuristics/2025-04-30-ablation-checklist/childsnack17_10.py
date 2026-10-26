from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts with patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args is empty
    if args and len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions required to serve all children.

    Heuristic Components:
    1.  Number of unserved children (each needs a 'serve' action).
    2.  Number of sandwiches that need to be made from ingredients.
    3.  Number of sandwiches that need to be put on a tray from the kitchen stock.
    4.  Number of tray movements needed to get trays to locations where unserved
        children are waiting and need a sandwich delivery.

    Assumptions:
    - Enough ingredients and sandwiches exist in total to serve all children.
    - A tray move is needed for a location if it requires any sandwich delivery
      and no tray is currently present there.
    - Sandwiches already on trays at the correct location for a waiting child
      reduce the need for new deliveries to that child's location.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Allergy status for each child.
        - Waiting location for each child.
        - Gluten-free status for bread and content portions.
        """
        self.goals = task.goals # Goal conditions (all children served)
        static_facts = task.static # Facts that don't change

        # Store static child information
        self.child_info = {} # child -> {'allergic': bool, 'location': place}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ['allergic_gluten', 'not_allergic_gluten']:
                child_name = parts[1]
                if child_name not in self.child_info:
                    self.child_info[child_name] = {}
                self.child_info[child_name]['allergic'] = (parts[0] == 'allergic_gluten')
            elif parts[0] == 'waiting':
                 child_name = parts[1]
                 location_name = parts[2]
                 if child_name not in self.child_info:
                    self.child_info[child_name] = {}
                 self.child_info[child_name]['location'] = location_name

        # Store static gluten-free ingredient information
        self.no_gluten_bread_facts = {fact for fact in static_facts if match(fact, 'no_gluten_bread', '*')}
        self.no_gluten_content_facts = {fact for fact in static_facts if match(fact, 'no_gluten_content', '*')}

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state # Current world state facts

        # 1. Count unserved children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, 'served', '*')}
        unserved_children = {child for child in self.child_info if child not in served_children}
        num_unserved_children = len(unserved_children)

        # If no children are unserved, the goal is reached.
        if num_unserved_children == 0:
            return 0

        # Heuristic starts with the cost of serving each child (1 action each)
        heuristic_cost = num_unserved_children

        # Dictionaries to track needed and available sandwiches at each location
        needed_at_loc_type = {} # (location, 'regular'/'gf') -> count
        available_ontray_at_loc_type = {} # (location, 'regular'/'gf') -> count

        # Identify needed sandwiches per location and type based on unserved children
        locations_with_unserved = set()
        for child in unserved_children:
            info = self.child_info.get(child)
            if info and 'location' in info:
                location = info['location']
                sandwich_type = 'gf' if info['allergic'] else 'regular'
                key = (location, sandwich_type)
                needed_at_loc_type[key] = needed_at_loc_type.get(key, 0) + 1
                locations_with_unserved.add(location)

        # Identify available sandwiches on trays at locations
        sandwiches_ontray = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, 'ontray', '*', '*')}
        sandwich_is_gf = {get_parts(fact)[1] for fact in state if match(fact, 'no_gluten_sandwich', '*')}
        tray_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, 'at', '*', '*')} # tray -> location

        for sandwich, tray in sandwiches_ontray.items():
            location = tray_locations.get(tray)
            if location: # Only consider trays with a known location
                sandwich_type = 'gf' if sandwich in sandwich_is_gf else 'regular'
                key = (location, sandwich_type)
                available_ontray_at_loc_type[key] = available_ontray_at_loc_type.get(key, 0) + 1

        # Calculate missing sandwiches per location and type
        missing_at_loc_type = {}
        total_missing_regular = 0
        total_missing_gf = 0

        for (location, s_type), needed_count in needed_at_loc_type.items():
            available_count = available_ontray_at_loc_type.get((location, s_type), 0)
            missing_count = max(0, needed_count - available_count)
            if missing_count > 0:
                missing_at_loc_type[(location, s_type)] = missing_count
                if s_type == 'regular':
                    total_missing_regular += missing_count
                else:
                    total_missing_gf += missing_count

        # If no sandwiches are missing at locations, we only need the serve actions
        if total_missing_regular == 0 and total_missing_gf == 0:
             return num_unserved_children # Should be 0 if all served, but safety check

        # 2. & 3. Cost for making sandwiches and putting them on trays (from kitchen)
        # These costs are incurred for the 'missing' sandwiches that need to be sourced
        # from the kitchen (either existing stock or made from ingredients).

        # Count available kitchen stock
        at_kitchen_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, 'at_kitchen_sandwich', '*')}
        ak_regular = len([s for s in at_kitchen_sandwiches if s not in sandwich_is_gf])
        ak_gf = len([s for s in at_kitchen_sandwiches if s in sandwich_is_gf])

        # Count available ingredients
        at_kitchen_bread = {get_parts(fact)[1] for fact in state if match(fact, 'at_kitchen_bread', '*')}
        at_kitchen_content = {get_parts(fact)[1] for fact in state if match(fact, 'at_kitchen_content', '*')}
        ak_bread_regular = len([b for b in at_kitchen_bread if '(no_gluten_bread ' + b + ')' not in self.no_gluten_bread_facts])
        ak_bread_gf = len([b for b in at_kitchen_bread if '(no_gluten_bread ' + b + ')' in self.no_gluten_bread_facts])
        ak_content_regular = len([c for c in at_kitchen_content if '(no_gluten_content ' + c + ')' not in self.no_gluten_content_facts])
        ak_content_gf = len([c for c in at_kitchen_content if '(no_gluten_content ' + c + ')' in self.no_gluten_content_facts])

        # Available ingredient pairs
        air_regular = min(ak_bread_regular, ak_content_regular)
        air_gf = min(ak_bread_gf, ak_content_gf)

        # Calculate how many missing sandwiches must be made vs taken from kitchen stock
        # Assume we prioritize using kitchen stock first
        take_from_kitchen_regular = min(total_missing_regular, ak_regular)
        needed_from_ingredients_regular = total_missing_regular - take_from_kitchen_regular
        make_regular = min(needed_from_ingredients_regular, air_regular) # Limited by ingredients

        take_from_kitchen_gf = min(total_missing_gf, ak_gf)
        needed_from_ingredients_gf = total_missing_gf - take_from_kitchen_gf
        make_gf = min(needed_from_ingredients_gf, air_gf) # Limited by ingredients

        # Cost for making sandwiches
        heuristic_cost += make_regular + make_gf

        # Cost for putting sandwiches on trays (those taken from kitchen stock or just made)
        heuristic_cost += take_from_kitchen_regular + make_regular + take_from_kitchen_gf + make_gf

        # 4. Cost for tray movements
        # Identify locations that need deliveries (where missing_at_loc_type > 0)
        locations_needing_delivery = {loc for (loc, s_type), count in missing_at_loc_type.items() if count > 0}

        # Check if a tray is already present at these locations
        trays_at_locations = set(tray_locations.values()) # Set of locations with a tray

        num_locations_need_tray_move = 0
        for location in locations_needing_delivery:
            if location != 'kitchen' and location not in trays_at_locations:
                 # We need to move a tray to this location
                 num_locations_need_tray_move += 1

        heuristic_cost += num_locations_need_tray_move

        return heuristic_cost

