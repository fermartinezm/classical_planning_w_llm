# Import necessary classes
from heuristics.heuristic_base import Heuristic
from task import Task # Used for type hinting and accessing task info

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a predicate and arguments.
    e.g., '(at tray1 kitchen)' -> ('at', ['tray1', 'kitchen'])
    """
    # Remove leading/trailing parentheses and split by space
    parts = fact_string[1:-1].split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the number of actions required to reach the goal
    state (all children served). It breaks down the problem into stages:
    making needed sandwiches, putting them on trays, moving trays to the
    children's locations, and finally serving the children. It sums the
    estimated minimum actions for each stage, aggregating needs and available
    items across all unserved children.

    Assumptions:
    - Children wait at non-kitchen places (based on domain examples).
    - Tray capacity is effectively infinite for heuristic calculation purposes.
    - Resource availability (bread, content, sandwich objects) is sufficient
      to make any needed sandwiches (this is a relaxation).
    - The 'kitchen' constant is always named 'kitchen'.
    - Any object with an '(at ?obj ?place)' predicate where ?place is a known
      place (kitchen or waiting place) is considered a tray for location tracking.

    Heuristic Initialization:
    The constructor processes the static facts from the task description.
    It identifies:
    - Which children are allergic to gluten and which are not.
    - The waiting place for each child.
    - Which bread and content portions are gluten-free.
    - All unique child names and place names (excluding kitchen).

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1.  Parse the state to identify served children, sandwiches at the kitchen,
        sandwiches on trays, gluten-free sandwiches that have been made,
        and tray locations.
    2.  Identify the set of unserved children. If this set is empty, the goal
        is reached, and the heuristic is 0.
    3.  Calculate H_serve: This is the number of unserved children. Each requires
        a final 'serve' action. Add this count to the total heuristic.
    4.  Calculate H_put: Determine the total number of gluten-free and regular
        sandwiches needed on trays (equal to the number of unserved allergic
        and non-allergic children, respectively). Subtract the number of
        suitable sandwiches already on trays (anywhere). The result is the number
        of sandwiches that still need to be put on trays. Add this count to the
        total heuristic.
    5.  Calculate H_make: Determine the number of sandwiches needed for H_put
        that are not currently available at the kitchen. This is the number
        of sandwiches that still need to be made. Differentiate between GF and
        regular sandwiches. Add this count to the total heuristic.
    6.  Calculate H_move: For each place where unserved children are waiting,
        determine if the suitable sandwiches currently on trays at that specific
        place are sufficient for the children waiting there. Count the number
        of such places where there is a deficit of suitable sandwiches on local
        trays. Each such place requires at least one tray movement to deliver
        needed sandwiches. Add this count to the total heuristic.
    7.  The total heuristic value is the sum of H_serve, H_put, H_make, and H_move.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goals = task.goals
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.waiting_places = {} # child -> place
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()
        self.all_children = set()
        self.places = {'kitchen'} # Start with kitchen, add others from waiting facts

        # Parse static facts
        for fact_string in task.static:
            predicate, args = parse_fact(fact_string)
            if predicate == 'allergic_gluten':
                self.allergic_children.add(args[0])
            elif predicate == 'not_allergic_gluten':
                self.not_allergic_children.add(args[0])
            elif predicate == 'waiting':
                child, place = args
                self.waiting_places[child] = place
                self.places.add(place) # Collect all places
            elif predicate == 'no_gluten_bread':
                self.no_gluten_breads.add(args[0])
            elif predicate == 'no_gluten_content':
                self.no_gluten_contents.add(args[0])

        self.all_children = self.allergic_children | self.not_allergic_children

    def __call__(self, node):
        state = node.state

        # --- Parse State ---
        served_children = set()
        at_kitchen_sandwiches = set()
        ontray_map = {} # tray -> set of sandwiches
        no_gluten_sandwiches_made = set()
        tray_locations = {} # tray -> place

        for fact_string in state:
            predicate, args = parse_fact(fact_string)
            if predicate == 'served':
                served_children.add(args[0])
            elif predicate == 'at_kitchen_sandwich':
                at_kitchen_sandwiches.add(args[0])
            elif predicate == 'ontray':
                sandwich, tray = args
                if tray not in ontray_map:
                    ontray_map[tray] = set()
                ontray_map[tray].add(sandwich)
            elif predicate == 'no_gluten_sandwich':
                 # This predicate can be true for sandwiches at kitchen or on tray
                 no_gluten_sandwiches_made.add(args[0])
            elif predicate == 'at' and len(args) == 2 and args[1] in self.places:
                 # Assume anything 'at' a known place is a tray
                 tray, place = args
                 tray_locations[tray] = place
            # Ignore other predicates like at_kitchen_bread, notexist etc. for heuristic calculation

        # --- Calculate Heuristic Components ---

        unserved_children = self.all_children - served_children
        num_unserved = len(unserved_children)

        # Goal reached
        if num_unserved == 0:
            return 0

        h = 0

        # H_serve: Cost for the final serve action for each unserved child
        h += num_unserved

        # Identify unserved children by allergy status
        unserved_allergic = [c for c in unserved_children if c in self.allergic_children]
        unserved_non_allergic = [c for c in unserved_children if c in self.not_allergic_children]
        num_allergic_unserved = len(unserved_allergic)
        num_non_allergic_unserved = len(unserved_non_allergic)

        # Helper to check if a sandwich is GF or Regular
        def is_gf_sandwich(s):
            return s in no_gluten_sandwiches_made

        # H_put: Sandwiches needing to be put on trays
        # Count available sandwiches already on trays (anywhere)
        avail_on_tray_gf = 0
        avail_on_tray_reg = 0
        for tray, sandwiches_on_this_tray in ontray_map.items():
             for s in sandwiches_on_this_tray:
                 if is_gf_sandwich(s):
                     avail_on_tray_gf += 1
                 else:
                     avail_on_tray_reg += 1 # Assumes any non-GF made sandwich is regular

        # Sandwiches needed on trays to serve all unserved children
        needed_on_tray_gf = max(0, num_allergic_unserved - avail_on_tray_gf)
        needed_on_tray_reg = max(0, num_non_allergic_unserved - avail_on_tray_reg)

        h_put = needed_on_tray_gf + needed_on_tray_reg
        h += h_put

        # H_make: Sandwiches needing to be made
        # Count available sandwiches at kitchen
        avail_k_gf = 0
        avail_k_reg = 0
        for s in at_kitchen_sandwiches:
            if is_gf_sandwich(s):
                avail_k_gf += 1
            else:
                avail_k_reg += 1 # Assumes any non-GF made sandwich is regular

        # Sandwiches that need to be made (those needed for H_put minus those at kitchen)
        needed_make_gf = max(0, needed_on_tray_gf - avail_k_gf)
        needed_make_reg = max(0, needed_on_tray_reg - avail_k_reg)

        h_make = needed_make_gf + needed_make_reg
        h += h_make

        # H_move: Tray movements needed
        # Count places where unserved children are waiting and calculate deficit
        places_with_unserved_children = set()
        needed_at_place = {} # place -> {'gf': count, 'reg': count}

        for child in unserved_children:
            place = self.waiting_places.get(child) # Get place from static info
            if place and place != 'kitchen': # Children wait at non-kitchen places
                places_with_unserved_children.add(place)
                if place not in needed_at_place:
                    needed_at_place[place] = {'gf': 0, 'reg': 0}
                if child in self.allergic_children:
                    needed_at_place[place]['gf'] += 1
                else:
                    needed_at_place[place]['reg'] += 1

        # Count suitable sandwiches already on trays at each place
        avail_at_place = {} # place -> {'gf': count, 'reg': count}
        for tray, place in tray_locations.items():
             if place not in avail_at_place:
                 avail_at_place[place] = {'gf': 0, 'reg': 0}
             if tray in ontray_map:
                 for s in ontray_map[tray]:
                     if is_gf_sandwich(s):
                         avail_at_place[place]['gf'] += 1
                     else:
                         avail_at_place[place]['reg'] += 1

        # Count places that have a deficit of suitable sandwiches on local trays
        h_move = 0
        for place in places_with_unserved_children:
            needed_gf_p = needed_at_place.get(place, {}).get('gf', 0)
            needed_reg_p = needed_at_place.get(place, {}).get('reg', 0)
            avail_gf_p = avail_at_place.get(place, {}).get('gf', 0)
            avail_reg_p = avail_at_place.get(place, {}).get('reg', 0)

            # Calculate deficit at this place
            current_deficit_p = max(0, needed_gf_p - avail_gf_p) + max(0, needed_reg_p - avail_reg_p)

            if current_deficit_p > 0:
                h_move += 1 # This place needs at least one delivery trip

        h += h_move

        return h
