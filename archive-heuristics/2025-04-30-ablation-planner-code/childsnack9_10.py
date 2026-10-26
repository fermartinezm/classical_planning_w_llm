from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
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
    This heuristic estimates the number of actions required to serve all children.
    It counts the number of unserved children (representing the final 'serve' action
    for each), the number of sandwiches that need to be made, and the number of
    sandwich-tray deliveries needed to get suitable sandwiches to the children's
    waiting places (representing 'put_on_tray' and 'move_tray' actions).

    # Assumptions
    - Resources (bread, content, unused sandwich objects, trays in kitchen) are
      sufficient to perform the estimated 'make' and 'put_on_tray' actions.
    - Each "deficit delivery" (a suitable sandwich needed at a location where none
      is currently on a tray) requires one 'put_on_tray' action and one 'move_tray' action.
      This simplifies resource sharing for trays and kitchen access.
    - The cost of each action is 1.

    # Heuristic Initialization
    - Extracts all objects of relevant types (child, sandwich, tray, place).
    - Extracts static information about children: their waiting place and allergy status.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  Identify all children from the goal that need to be served.
    2.  Identify which of these goal children are currently served in the given state.
    3.  Count the number of unserved goal children. This contributes 1 to the heuristic
        for each child, representing the final 'serve' action.
    4.  Group the unserved goal children by their waiting place and the type of sandwich
        they need (regular or gluten-free), using information extracted from static facts.
        This gives us the total demand for each sandwich type at each location.
    5.  Count the number of suitable sandwiches that are currently on trays and
        already located at the correct waiting places for the children, based on the current state.
    6.  For each combination of (place, sandwich_type), calculate the 'deficit':
        the number of sandwiches of that type needed at that place minus the number
        already available on trays at that place. The total deficit across all
        locations and types represents the minimum number of sandwich-tray deliveries
        that still need to be completed.
    7.  Each deficit delivery requires getting a sandwich onto a tray and moving
        that tray to the correct location. Add 1 to the heuristic for each deficit
        delivery (representing a 'put_on_tray' action) and another 1 for each
        deficit delivery (representing a 'move_tray' action).
    8.  Count the total number of regular and gluten-free sandwiches needed across
        all unserved goal children.
    9.  Count the total number of regular and gluten-free sandwiches that currently
        exist anywhere (in the kitchen or on any tray), based on the current state.
    10. Calculate the number of regular and gluten-free sandwiches that still need
        to be 'made' by comparing the total needed vs. total available. Add 1 to
        the heuristic for each sandwich that needs to be made.
    11. The total heuristic value is the sum of costs from steps 3, 7, and 10.
        This is: (unserved goal children) + 2 * (total deficit deliveries) + (sandwiches to make).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting child information and object lists.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract objects
        self.children = [obj for obj, obj_type in task.objects if obj_type == 'child']
        self.sandwiches = [obj for obj, obj_type in task.objects if obj_type == 'sandwich']
        self.trays = [obj for obj, obj_type in task.objects if obj_type == 'tray']
        self.places = [obj for obj, obj_type in task.objects if obj_type == 'place']
        # Add kitchen constant if not already in places
        if 'kitchen' not in self.places:
             self.places.append('kitchen')

        # Extract child info: {child_name: (place, needs_gluten_free)}
        self.child_info = {}
        static_facts_set = set(static_facts) # Use set for faster lookups

        for child in self.children:
            place = None
            needs_gf = None

            # Find waiting place
            for fact in static_facts_set:
                 parts = get_parts(fact)
                 if parts and parts[0] == 'waiting' and len(parts) == 3 and parts[1] == child:
                     place = parts[2]
                     break # Found waiting place

            # Find allergy status
            if '(allergic_gluten ' + child + ')' in static_facts_set:
                needs_gf = True
            elif '(not_allergic_gluten ' + child + ')' in static_facts_set:
                needs_gf = False

            # Store info only if both place and allergy status are found
            # Children in the goal *must* have this info in a valid instance
            if place is not None and needs_gf is not None:
                 self.child_info[child] = (place, needs_gf)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state # state is already a frozenset
        total_cost = 0

        # 1. Identify unserved children from the goal
        # The goal is a list of facts like (served childX)
        goal_served_children = {get_parts(g)[1] for g in self.goals if get_parts(g) and get_parts(g)[0] == 'served'}

        served_in_state = {c for c in goal_served_children if '(served ' + c + ')' in state}
        unserved_children = [c for c in goal_served_children if c not in served_in_state]

        # If all children in the goal are served, the heuristic is 0
        if not unserved_children:
            return 0

        # Cost for the final 'serve' action for each unserved child
        total_cost += len(unserved_children)

        # 2. Group needed sandwiches by place and type
        needed_at_place_type = defaultdict(int) # {(place, needs_gf): count}
        for child in unserved_children:
            # Ensure child has waiting/allergy info - this should be guaranteed by valid PDDL
            if child in self.child_info:
                place, needs_gf = self.child_info[child]
                needed_at_place_type[(place, needs_gf)] += 1
            # else: This child is in the goal but not in child_info (missing static facts).
            # This indicates an invalid problem instance according to domain rules.
            # We proceed assuming valid instances where all goal children have info.


        # 3. Count available sandwiches on trays at each location
        available_ontray_at_place_type = defaultdict(int) # {(place, is_gf): count}
        sandwich_on_tray_map = {} # {sandwich: tray}
        tray_location_map = {} # {tray: place}
        sandwich_is_gf_map = {} # {sandwich: is_gf}

        # Build maps from state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'ontray' and len(parts) == 3:
                s, t = parts[1], parts[2]
                sandwich_on_tray_map[s] = t
            elif predicate == 'at' and len(parts) == 3:
                 # This predicate is used for trays and potentially other things, filter for trays
                 obj, place = parts[1], parts[2]
                 if obj in self.trays:
                    tray_location_map[obj] = place
            elif predicate == 'no_gluten_sandwich' and len(parts) == 2:
                s = parts[1]
                sandwich_is_gf_map[s] = True

        # Populate available_ontray_at_place_type
        for s, t in sandwich_on_tray_map.items():
            if t in tray_location_map:
                place = tray_location_map[t]
                is_gf = sandwich_is_gf_map.get(s, False) # Default to False if not explicitly marked GF
                available_ontray_at_place_type[(place, is_gf)] += 1

        # 4. Calculate total deficit deliveries
        total_deficit_deliveries = 0
        for (place, needs_gf), needed_count in needed_at_place_type.items():
            is_gf = needs_gf # Match needs_gf with is_gf for available count
            available_count = available_ontray_at_place_type.get((place, is_gf), 0)
            deficit = max(0, needed_count - available_count)
            total_deficit_deliveries += deficit

        # 5. Add cost for 'put_on_tray' and 'move_tray' for each deficit delivery
        total_cost += total_deficit_deliveries # Cost for put_on_tray
        total_cost += total_deficit_deliveries # Cost for move_tray


        # 6. Count sandwiches that need to be made
        # Count available sandwiches anywhere (kitchen or on tray)
        available_reg_anywhere = 0
        available_gf_anywhere = 0

        for s in self.sandwiches:
            # A sandwich object 's' exists if it's not marked as '(notexist s)'
            # We only count sandwiches that have been 'made' (i.e., not notexist)
            # and are not yet served (implicitly removed from state).
            # Sandwiches that are 'at_kitchen_sandwich' or 'ontray' are definitely made and not served.
            # Sandwiches that were made but are not at_kitchen_sandwich or ontray
            # implies they were served. So we only need to check at_kitchen_sandwich or ontray.

            is_at_kitchen = '(at_kitchen_sandwich ' + s + ')' in state
            is_ontray = s in sandwich_on_tray_map # Check if key exists in map built from state

            if is_at_kitchen or is_ontray:
                 is_gf_state = sandwich_is_gf_map.get(s, False)
                 if is_gf_state:
                     available_gf_anywhere += 1
                 else:
                     available_reg_anywhere += 1

        # Calculate total needed sandwiches across all locations
        total_reg_needed = sum(count for (place, needs_gf), count in needed_at_place_type.items() if not needs_gf)
        total_gf_needed = sum(count for (place, needs_gf), count in needed_at_place_type.items() if needs_gf)

        # Calculate sandwiches to make
        num_reg_to_make = max(0, total_reg_needed - available_reg_anywhere)
        num_gf_to_make = max(0, total_gf_needed - available_gf_anywhere)

        # Add cost for 'make' actions
        total_cost += num_reg_to_make
        total_cost += num_gf_to_make

        return total_cost
