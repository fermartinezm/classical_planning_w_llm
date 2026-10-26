from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It sums up the estimated costs for different stages of the process:
    serving the child, moving a tray to the child's location, putting a sandwich
    on a tray, and making a sandwich. It counts the number of items/tasks
    needed at each stage based on the current state and the total number of
    unserved children.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - A tray can carry multiple sandwiches needed at the same location.
    - A single move action can bring a tray to a location needing sandwiches.
    - Ingredients and 'notexist' sandwich objects are sufficient to make any needed sandwich,
      unless explicitly counted as insufficient (though the current implementation
      primarily counts based on the number of sandwiches needed vs available).
    - The primary bottleneck for making sandwiches is the availability of 'notexist' objects
      and ingredients (implicitly assumed sufficient if 'notexist' objects exist).

    # Heuristic Initialization
    - Extracts the set of all children who need to be served from the goal state.
    - Extracts which children are allergic to gluten from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children that need to be served (from the goal state).
    2. Identify which of these children are already served in the current state.
    3. Count the total number of unserved children (`N_unserved`). This contributes `N_unserved` to the heuristic (representing the `serve` actions). If `N_unserved` is 0, the goal is reached, and the heuristic is 0.
    4. Group unserved children by their waiting location (from the current state).
    5. Count the total number of sandwiches currently on trays anywhere (`Avail_Total_ontray_anywhere`).
    6. Count the total number of sandwiches currently in the kitchen (`Avail_Total_kitchen_S`).
    7. Count the total number of sandwiches already made (either ontray or in kitchen) (`Avail_Total_made_anywhere = Avail_Total_ontray_anywhere + Avail_Total_kitchen_S`).
    8. For each location where unserved children are waiting, count the number of suitable sandwiches already on trays *at that specific location*. A sandwich is suitable at a location if it's on a tray there and is suitable for at least one unserved child waiting there (GF for allergic, any for non-allergic).
    9. For each location with unserved children, compare the number of unserved children there with the number of suitable sandwiches already on trays at that location. If the number of children exceeds the available suitable sandwiches, this location needs more sandwiches brought to it. Count the number of such locations. This represents the minimum number of `move_tray` actions needed to bring trays with sandwiches to these locations. Add this count to the heuristic.
    10. Calculate the number of sandwiches that need to be put on trays. This is the total number of sandwiches needed (`N_unserved`) minus those already on trays anywhere (`Avail_Total_ontray_anywhere`). Add this count (`Needed_put`) to the heuristic (representing `put_on_tray` actions).
    11. Calculate the number of sandwiches that need to be made. This is the total number of sandwiches needed (`N_unserved`) minus those already made (`Avail_Total_made_anywhere`). Add this count (`Needed_make`) to the heuristic (representing `make_sandwich` actions).
    12. The total heuristic value is the sum of the counts from steps 3, 9, 10, and 11.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract all children from the goal state
        self.goal_children = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'served':
                self.goal_children.add(parts[1])

        # Extract allergy status from static facts
        self.allergic_children = set()
        self.not_allergic_children = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif parts[0] == 'not_allergic_gluten':
                 self.not_allergic_children.add(parts[1])


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Data structures to hold state information parsed from the current state
        served_children = set()
        waiting_children_in_state = {} # {child: place}
        ontray_sandwiches_set = set() # Just the set of sandwiches on trays
        sandwich_to_tray = {} # {sandwich: tray}
        at_kitchen_sandwich = set()
        no_gluten_sandwiches_in_state = set()
        tray_locations = {} # {tray: place}
        # notexist_sandwich_objects = set() # Not strictly needed for this heuristic calculation

        # Parse the current state facts in a single pass
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'served':
                served_children.add(parts[1])
            elif parts[0] == 'waiting':
                child, place = parts[1], parts[2]
                waiting_children_in_state[child] = place
            elif parts[0] == 'ontray':
                s, t = parts[1], parts[2]
                ontray_sandwiches_set.add(s)
                sandwich_to_tray[s] = t
            elif parts[0] == 'at_kitchen_sandwich':
                s = parts[1]
                at_kitchen_sandwich.add(s)
            elif parts[0] == 'no_gluten_sandwich':
                s = parts[1]
                no_gluten_sandwiches_in_state.add(s)
            elif parts[0] == 'at' and parts[1].startswith('tray'):
                 t, p = parts[1], parts[2]
                 tray_locations[t] = p
            # elif parts[0] == 'notexist':
            #      s_obj = parts[1]
            #      notexist_sandwich_objects.add(s_obj)


        # 1, 2, 3. Identify unserved children and calculate base cost (serve actions)
        unserved_children = self.goal_children - served_children
        N_unserved = len(unserved_children)

        # If all children are served, the goal is reached.
        if N_unserved == 0:
            return 0

        # Initialize heuristic cost
        h = N_unserved # Cost for serve actions

        # 4. Group unserved children by their current waiting location
        unserved_children_by_location = {} # {place: {child1, child2, ...}}
        for child in unserved_children:
            place = waiting_children_in_state.get(child)
            if place: # Child should always be waiting if not served
                if place not in unserved_children_by_location:
                    unserved_children_by_location[place] = set()
                unserved_children_by_location[place].add(child)

        # 5, 6, 7. Count available sandwiches
        Avail_Total_ontray_anywhere = len(ontray_sandwiches_set)
        Avail_Total_kitchen_S = len(at_kitchen_sandwich)
        Avail_Total_made_anywhere = Avail_Total_ontray_anywhere + Avail_Total_kitchen_S

        # 8. Count suitable sandwiches ontray at each location with unserved children
        suitable_ontray_at_location_count = {} # {place: count}

        # Iterate through locations that have unserved children
        for place in unserved_children_by_location:
            count_suitable = 0
            # Find trays at this location
            trays_at_this_place = {t for t, p in tray_locations.items() if p == place}

            # Find sandwiches on these trays
            sandwiches_on_trays_at_this_place = {s for s, t in sandwich_to_tray.items() if t in trays_at_this_place}

            # Check suitability for any unserved child at this location
            for s in sandwiches_on_trays_at_this_place:
                sandwich_is_gf = s in no_gluten_sandwiches_in_state
                is_suitable_for_any_child_at_p = False
                for child in unserved_children_by_location[place]:
                    child_is_allergic = child in self.allergic_children
                    if (child_is_allergic and sandwich_is_gf) or (not child_is_allergic):
                         is_suitable_for_any_child_at_p = True
                         break
                if is_suitable_for_any_child_at_p:
                     count_suitable += 1
            if count_suitable > 0: # Only store if there's at least one suitable sandwich
                suitable_ontray_at_location_count[place] = count_suitable

        # 9. Calculate locations needing sandwiches brought and count move actions
        locations_need_move = 0
        for place, children_at_p in unserved_children_by_location.items():
            needed_at_p = len(children_at_p)
            available_ontray_at_p = suitable_ontray_at_location_count.get(place, 0)
            # If needed > available, we need to bring more sandwiches to this location.
            if needed_at_p > available_ontray_at_p:
                 locations_need_move += 1

        h += locations_need_move # Cost for move_tray actions

        # 10. Calculate sandwiches needing to be put on trays
        Needed_put = max(0, N_unserved - Avail_Total_ontray_anywhere)
        h += Needed_put # Cost for put_on_tray actions

        # 11. Calculate sandwiches needing to be made
        Needed_make = max(0, N_unserved - Avail_Total_made_anywhere)
        h += Needed_make # Cost for make_sandwich actions

        # 12. Total heuristic value
        return h
