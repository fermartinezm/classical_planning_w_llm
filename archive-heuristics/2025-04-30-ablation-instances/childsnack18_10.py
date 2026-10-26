from fnmatch import fnmatch
# Assuming Heuristic base class is available from a separate file/module
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing if needed
# In a real planner environment, this would be provided.
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError

# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# --- Childsnacks Heuristic ---

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It decomposes the problem into stages: making sandwiches, putting them on trays,
    moving trays to children's locations, and finally serving the children. The heuristic
    sums the estimated number of actions needed for each stage based on the current state
    and the requirements of the unserved children.

    # Assumptions
    - All children waiting in the initial state must be served.
    - Resources (bread, content) are sufficient to make needed sandwiches (resource availability is not explicitly modeled in the heuristic cost).
    - Trays are available as needed (though their location matters).
    - The primary goal is serving all children.

    # Heuristic Initialization
    The heuristic extracts static information from the task:
    - Which children are allergic to gluten.
    - The waiting location for each child.
    - The set of all children, sandwiches, and trays defined in the problem (extracted pragmatically from initial/goal/static facts).
    - The set of gluten-free bread and content types (though resource availability is simplified in the heuristic calculation).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic estimates the total cost by summing up the estimated costs for four main stages, based on the deficit of required conditions:

    1.  **Serving:** Each unserved child requires a final 'serve' action.
        - Count the number of children who are not yet marked as 'served'. Add this count to the heuristic.

    2.  **Delivery (Move Tray):** Suitable sandwiches must be on trays at the locations where children are waiting.
        - For each location where unserved children are waiting, determine the number of gluten-free and regular sandwiches needed (based on child allergies).
        - Count the number of suitable sandwiches already on trays at that specific location.
        - Calculate the deficit of suitable sandwiches at each location.
        - Sum the deficits across all locations. This total represents the number of "sandwich-at-location" requirements that are not met. Add this sum to the heuristic. This estimates the number of 'move_tray' actions needed to bring sandwiches to locations.

    3.  **Placement on Tray (Put on Tray):** Needed sandwiches that are not already on trays anywhere must be put on a tray.
        - Calculate the total number of suitable sandwiches needed to cover the deficits at locations (from step 2).
        - Count the number of suitable sandwiches that are currently on trays, but not at the needed locations (i.e., on trays elsewhere).
        - The number of sandwiches that need to be put on trays is the total needed for deficits minus those already on trays elsewhere (minimum 0). Add this count to the heuristic. This estimates the number of 'put_on_tray' actions needed.

    4.  **Creation (Make Sandwich):** Needed sandwiches that do not exist anywhere must be made.
        - Calculate the number of sandwiches that need to be put on trays (from step 3).
        - Count the number of suitable sandwiches that currently exist in the kitchen (but are not on trays).
        - The number of sandwiches that need to be made is the number needing to be put on trays minus those already in the kitchen (minimum 0). Add this count to the heuristic. This estimates the number of 'make_sandwich' actions needed.

    The total heuristic value is the sum of the counts from steps 1, 2, 3, and 4.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children,
        locations, and object types.
        """
        super().__init__(task)

        self.allergic_children = set()
        self.not_allergic_children = set()
        self.waiting_location = {} # child -> place
        self.all_children = set()
        self.all_sandwiches = set()
        self.all_trays = set()
        self.all_places = {'kitchen'} # kitchen is a constant place

        # Extract static information
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
                self.all_children.add(parts[1])
            elif parts[0] == 'not_allergic_gluten':
                self.not_allergic_children.add(parts[1])
                self.all_children.add(parts[1])
            elif parts[0] == 'waiting':
                child, place = parts[1], parts[2]
                self.waiting_location[child] = place
                self.all_children.add(child) # Ensure child is added even if allergy isn't static
                self.all_places.add(place) # Ensure place is added

        # Extract all possible sandwiches and trays from initial state/goals if not in static
        # This is a pragmatic approach given the input format, assuming relevant objects appear in facts.
        all_facts = set(task.initial_state) | set(task.goals) | set(task.static)
        for fact in all_facts:
             parts = get_parts(fact)
             if parts[0] in ['at_kitchen_sandwich', 'ontray', 'notexist', 'no_gluten_sandwich']:
                 if len(parts) > 1: self.all_sandwiches.add(parts[1])
             elif parts[0] in ['ontray', 'at']:
                 if len(parts) > 2: self.all_trays.add(parts[2]) # (ontray s t), (at t p)
             elif parts[0] in ['at', 'waiting']:
                 if len(parts) > 2: self.all_places.add(parts[2]) # (at t p), (waiting c p)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        heuristic_value = 0

        # 1. Serving: Count unserved children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, 'served', '*')}
        unserved_children = self.all_children - served_children
        num_unserved = len(unserved_children)
        heuristic_value += num_unserved # Cost for the 'serve' action for each

        if num_unserved == 0:
            return 0 # Goal state reached

        # Identify gluten status of existing sandwiches
        sandwich_is_gf = {s: '(no_gluten_sandwich ' + s + ')' in state for s in self.all_sandwiches}

        # Identify locations of trays
        tray_location = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, 'at', '*', '*')} # tray -> place

        # Identify which sandwiches are on which trays
        sandwiches_on_trays = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, 'ontray', '*', '*')} # sandwich -> tray

        # Identify sandwiches in the kitchen (not on trays)
        sandwiches_in_kitchen = {get_parts(fact)[1] for fact in state if match(fact, 'at_kitchen_sandwich', '*')}

        # 2. Delivery (Move Tray): Calculate sandwich needs at locations and deficit
        needs_at_location = {} # {place: {'gf': count, 'any': count}}
        for child in unserved_children:
            place = self.waiting_location.get(child) # Get place from static info
            if place: # Child might not have a waiting location in some initial states?
                if place not in needs_at_location:
                    needs_at_location[place] = {'gf': 0, 'any': 0}
                if child in self.allergic_children:
                    needs_at_location[place]['gf'] += 1
                else:
                    needs_at_location[place]['any'] += 1 # Non-allergic can take any

        # Count available sandwiches on trays at locations where children are waiting
        available_on_trays_at_location = {p: {'gf': 0, 'any': 0} for p in needs_at_location}

        for sandwich, tray in sandwiches_on_trays.items():
            place = tray_location.get(tray)
            if place and place in needs_at_location:
                 if sandwich_is_gf.get(sandwich, False): # Default to False if sandwich not in state/no_gluten check failed
                     available_on_trays_at_location[place]['gf'] += 1
                 else:
                     available_on_trays_at_location[place]['any'] += 1

        # Calculate deficit at each location
        total_deficit_gf = 0
        total_deficit_any = 0
        for place, needed in needs_at_location.items():
            available = available_on_trays_at_location.get(place, {'gf': 0, 'any': 0})

            needed_gf = needed['gf']
            needed_any = needed['any']
            available_gf = available['gf']
            available_any = available['any']

            # GF sandwiches satisfy GF needs first
            satisfied_gf = min(needed_gf, available_gf)
            needed_gf -= satisfied_gf
            available_gf -= satisfied_gf

            # Remaining GF sandwiches satisfy 'any' needs
            satisfied_any_by_gf = min(needed_any, available_gf)
            needed_any -= satisfied_any_by_gf
            available_gf -= satisfied_any_by_gf

            # Remaining 'any' needs satisfied by non-GF sandwiches
            satisfied_any_by_any = min(needed_any, available_any)
            needed_any -= satisfied_any_by_any
            available_any -= satisfied_any_by_any

            # The remaining needed_gf and needed_any are the deficit at this location
            total_deficit_gf += needed_gf
            total_deficit_any += needed_any

        heuristic_value += total_deficit_gf + total_deficit_any # Cost for 'move_tray' actions

        # 3. Placement on Tray (Put on Tray): Calculate items needing put_on_tray
        # These are the items needed at locations (deficit) that are NOT on trays elsewhere.

        # Count available on trays elsewhere
        available_ontray_elsewhere = {'gf': 0, 'any': 0}
        for sandwich, tray in sandwiches_on_trays.items():
             place = tray_location.get(tray)
             if place and place not in needs_at_location: # It's on a tray, but not at a needed location
                 if sandwich_is_gf.get(sandwich, False):
                     available_ontray_elsewhere['gf'] += 1
                 else:
                     available_ontray_elsewhere['any'] += 1

        needed_put_gf = total_deficit_gf
        needed_put_any = total_deficit_any

        # Satisfy needed_put_gf from available_ontray_elsewhere_gf
        satisfied_put_gf_by_ontray_gf = min(needed_put_gf, available_ontray_elsewhere['gf'])
        needed_put_gf -= satisfied_put_gf_by_ontray_gf
        available_ontray_elsewhere['gf'] -= satisfied_put_gf_by_ontray_gf

        # Satisfy needed_put_any from remaining available_ontray_elsewhere_gf
        satisfied_put_any_by_ontray_gf = min(needed_put_any, available_ontray_elsewhere['gf'])
        needed_put_any -= satisfied_put_any_by_ontray_gf
        available_ontray_elsewhere['gf'] -= satisfied_put_any_by_ontray_gf

        # Satisfy needed_put_any from available_ontray_elsewhere_any
        satisfied_put_any_by_ontray_any = min(needed_put_any, available_ontray_elsewhere['any'])
        needed_put_any -= satisfied_put_any_by_ontray_any
        available_ontray_elsewhere['any'] -= satisfied_put_any_by_ontray_any

        # The remaining needed_put_gf + needed_put_any is the number of items
        # that must come from sandwiches NOT currently on trays anywhere.
        num_need_put_on_tray_items = needed_put_gf + needed_put_any
        heuristic_value += num_need_put_on_tray_items # Cost for 'put_on_tray' actions

        # 4. Creation (Make Sandwich): Calculate items needing to be made
        # These are the items needing put_on_tray that are not in the kitchen

        # Count available in kitchen (only sandwiches not on trays are relevant for 'put_on_tray' source)
        available_kitchen = {'gf': 0, 'any': 0}
        for sandwich in sandwiches_in_kitchen:
             if sandwich_is_gf.get(sandwich, False):
                 available_kitchen['gf'] += 1
             else:
                 available_kitchen['any'] += 1

        needed_make_gf = needed_put_gf # Items needing put_on_tray (GF)
        needed_make_any = needed_put_any # Items needing put_on_tray (Any)

        # Satisfy needed_make_gf from available_kitchen_gf
        satisfied_make_gf_by_kitchen_gf = min(needed_make_gf, available_kitchen['gf'])
        needed_make_gf -= satisfied_make_gf_by_kitchen_gf
        available_kitchen['gf'] -= satisfied_make_gf_by_kitchen_gf

        # Satisfy needed_make_any from remaining available_kitchen_gf
        satisfied_make_any_by_kitchen_gf = min(needed_make_any, available_kitchen['gf'])
        needed_make_any -= satisfied_make_any_by_kitchen_gf
        available_kitchen['gf'] -= satisfied_make_any_by_kitchen_gf

        # Satisfy needed_make_any from available_kitchen_any
        satisfied_make_any_by_kitchen_any = min(needed_make_any, available_kitchen['any'])
        needed_make_any -= satisfied_make_any_by_kitchen_any
        available_kitchen['any'] -= satisfied_make_any_by_kitchen_any

        # The remaining needed_make_gf + needed_make_any must be made
        num_need_make_items = needed_make_gf + needed_make_any
        heuristic_value += num_need_make_items # Cost for 'make_sandwich' actions

        return heuristic_value
