# The provided code structure implies the heuristic class is defined at the top level.
# The helper function get_object needs to be accessible by the class.

# Helper function to extract objects from PDDL fact strings
def get_object(fact_string, index):
    """
    Extracts the object at the given index from a PDDL fact string.
    Assumes fact_string is like '(predicate obj1 obj2 ...)'
    Index 1 is the first object, index 2 is the second, etc.
    """
    # Remove surrounding brackets and split by space
    parts = fact_string[1:-1].split()
    # Objects are at index 1 and onwards
    if index < 1 or index >= len(parts):
        # This should not happen with valid PDDL facts and correct index usage
        # but adding a safeguard.
        # print(f"Warning: Object index {index} out of bounds for fact '{fact_string}'")
        return None # Or raise an error
    return parts[index]


class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the remaining actions required to serve all waiting children.
    It counts the number of children not yet served, the number of sandwiches that need
    to be made, the number of sandwiches that need to be put on trays at the kitchen,
    and the number of tray movements required to get trays to the locations where
    they are needed (child locations and potentially the kitchen).

    Assumptions:
    - The problem is solvable (enough bread, content, sandwich objects, and trays exist).
    - Trays have infinite capacity for sandwiches.
    - The static facts correctly define child allergies, gluten-free items, and initial waiting locations.
    - The `kitchen` constant exists and is represented as the string 'kitchen'.

    Heuristic Initialization:
    The constructor pre-processes the static facts to quickly determine:
    - Which children are allergic or not allergic.
    - The initial waiting place for each child.
    - All children mentioned in static waiting facts (as the set of children to be served).
    This information is stored in sets and dictionaries for efficient lookup during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Check if the goal is reached: Identify all children who are marked as 'served' in the current state. If this set includes all children who were initially waiting (as per static facts), the heuristic is 0.
    2.  Identify waiting children: Count the number of children who are defined as 'waiting' in the static facts but are not 'served' in the current state (`N_waiting`). This is a base cost representing the minimum number of 'serve' actions needed.
    3.  Count available sandwiches: Determine the number of sandwiches currently `at_kitchen_sandwich` (`N_sandwich_kitchen`) and `ontray` (`N_sandwich_ontray`).
    4.  Estimate sandwiches to make: Calculate how many new sandwiches need to be made to meet the demand of waiting children. This is `max(0, N_waiting - (N_sandwich_kitchen + N_sandwich_ontray))`. Each needs a 'make_sandwich' action. Add this count to the heuristic.
    5.  Estimate sandwiches needing 'put_on_tray': Sandwiches that are `at_kitchen_sandwich` (initially or newly made) need to be put on a tray. The number is `N_sandwich_kitchen + sandwiches_to_make`. Each needs a 'put_on_tray' action. Add this count to the heuristic.
    6.  Identify places needing tray delivery: Determine the set of places where trays are required. This includes all places where children are waiting (from static facts). If there are sandwiches needing to be put on trays at the kitchen (`sandwiches_needing_put_on_tray > 0`), the kitchen also needs a tray.
    7.  Estimate tray movements: Count the number of places identified in step 6 that currently do *not* have a tray according to the state. Each such place requires at least one 'move_tray' action to bring a tray there. Add this count to the heuristic.
    8.  Sum the components: The total heuristic value is the sum of the counts from steps 2, 4, 5, and 7. This represents an estimate of the total actions (serve, make, put_on_tray, move_tray) required.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by pre-processing static information from the task.

        @param task: The planning task object.
        """
        self.static_waiting_place = {} # child -> place
        self.all_children_in_static_waiting = set() # All children mentioned in static waiting facts

        # Pre-process static facts
        for f in task.static:
            if f.startswith('(waiting '):
                child = get_object(f, 1)
                place = get_object(f, 2)
                self.static_waiting_place[child] = place
                self.all_children_in_static_waiting.add(child)

        # Note: Other static facts like allergies or no_gluten status are not strictly
        # needed for *this specific* heuristic calculation approach, which focuses
        # on counts of items/locations needing processing rather than specific
        # sandwich-child matching. They might be useful for a more refined heuristic.


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state: The current state (frozenset of facts).
        @return: The estimated number of actions to reach the goal.
        """
        # 1. Check if goal is reached
        served_children_in_state = set()
        for fact in state:
            if fact.startswith('(served '):
                served_children_in_state.add(get_object(fact, 1))

        # Goal is reached if all children who were initially waiting are now served
        if self.all_children_in_static_waiting.issubset(served_children_in_state):
             return 0

        # 2. Identify waiting children
        # Children are waiting if they are in static_waiting_place and not served
        waiting_children_count = 0
        for child in self.all_children_in_static_waiting:
            if child not in served_children_in_state:
                waiting_children_count += 1

        # If waiting_children_count is 0, it means all children initially waiting are now served.
        # This should be covered by the initial goal check, but good safety.
        if waiting_children_count == 0:
             return 0

        # Parse state for dynamic counts and locations
        at_kitchen_sandwich_count = 0
        ontray_count = 0
        at_tray_map = {} # tray -> place

        for fact in state:
            if fact.startswith('(at_kitchen_sandwich '):
                at_kitchen_sandwich_count += 1
            elif fact.startswith('(ontray '):
                ontray_count += 1
            elif fact.startswith('(at ?t '):
                # Extract tray and place objects
                parts = fact[1:-1].split()
                if len(parts) == 3: # Expected format '(at tray place)'
                    tray = parts[1]
                    place = parts[2]
                    at_tray_map[tray] = place
                # else: ignore malformed fact? Assume valid PDDL representation.


        # 3. Estimate sandwiches to make
        # We need 'waiting_children_count' sandwiches in total.
        # Some are already available (at kitchen or on tray).
        # The rest need to be made.
        sandwiches_available = at_kitchen_sandwich_count + ontray_count
        sandwiches_to_make = max(0, waiting_children_count - sandwiches_available)
        h = sandwiches_to_make # Cost for make actions

        # 4. Estimate sandwiches needing 'put_on_tray'
        # Sandwiches that are currently at the kitchen (initially or newly made)
        # need to be put on a tray.
        sandwiches_needing_put_on_tray = at_kitchen_sandwich_count + sandwiches_to_make
        h += sandwiches_needing_put_on_tray # Cost for put_on_tray actions

        # 5. Identify places needing tray delivery and estimate tray movements
        places_needing_tray_delivery = set()
        # Add places where children are waiting
        for child, place in self.static_waiting_place.items():
             if child not in served_children_in_state: # Only consider places for children still waiting
                 places_needing_tray_delivery.add(place)

        # Add kitchen if sandwiches need putting on trays there
        if sandwiches_needing_put_on_tray > 0:
             places_needing_tray_delivery.add('kitchen')

        # Count places that need a tray but don't have one
        trays_at_place = {} # place -> set of trays
        for tray, place in at_tray_map.items():
            trays_at_place.setdefault(place, set()).add(tray)

        tray_moves_needed = 0
        for place in places_needing_tray_delivery:
            # Check if the place has *any* tray
            if place not in trays_at_place or not trays_at_place[place]:
                 tray_moves_needed += 1

        h += tray_moves_needed # Cost for move_tray actions

        # 6. Add cost for serve actions
        # Each waiting child needs a serve action eventually.
        h += waiting_children_count

        return h
