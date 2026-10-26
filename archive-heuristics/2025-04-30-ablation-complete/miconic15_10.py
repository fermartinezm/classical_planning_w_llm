from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions (outside the class as in examples)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# The main heuristic class
class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    Estimates the number of actions (board, depart, move) required to serve all passengers.
    It counts the pending board and depart actions and estimates the minimum lift movement
    needed to visit all relevant floors (current lift floor, passenger origins, passenger destinations).

    # Assumptions
    - The floor structure is a simple linear sequence defined by `above` predicates, where `(above f_high f_low)` means `f_high` is immediately above `f_low`.
    - Each board and depart action costs 1.
    - Each move action (up/down one floor) costs 1.
    - The minimum moves to visit a set of floors on a line starting from a point is the distance to one end of the required floor range plus the total span of the range.

    # Heuristic Initialization
    - Parses the `above` predicates from static facts to determine the floor order and create a mapping from floor name to its index.
    - Extracts the destination floor for each passenger from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift.
    2. Identify all unserved passengers (those not in the `served` predicate).
    3. For each unserved passenger, determine if they are waiting at their origin or are boarded.
    4. Collect the set of "relevant floors":
       - The lift's current floor.
       - The origin floor for every unserved passenger who is waiting.
       - The destination floor for every unserved passenger (both waiting and boarded).
    5. Count the number of pending board actions: This is the number of unserved passengers who are waiting.
    6. Count the number of pending depart actions: This is the total number of unserved passengers.
    7. Estimate the number of move actions:
       - If there are no unserved passengers, the move cost is 0.
       - Otherwise, find the minimum and maximum floor indices among the relevant floors.
       - Calculate the minimum moves required to travel from the current lift floor to cover the entire range of relevant floors. This is the span of the relevant floors plus the minimum distance from the current lift floor to either end of the span.
    8. The total heuristic value is the sum of pending board actions, pending depart actions, and estimated move actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse floor order and create floor_name -> index map
        # Assuming (above f_high f_low) means f_high is immediately above f_low
        floor_above_map = {} # Maps floor -> floor immediately above it
        all_floors = set()
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f_high, f_low = get_parts(fact)[1:]
                floor_above_map[f_low] = f_high
                all_floors.add(f_high)
                all_floors.add(f_low)

        # Find the lowest floor (a floor that is not a value in floor_above_map)
        lowest_floor = None
        floors_that_are_above_others = set(floor_above_map.values())
        for floor in all_floors:
            if floor not in floors_that_are_above_others:
                lowest_floor = floor
                break

        # Build the ordered list of floors starting from the lowest
        self.floor_order = []
        current = lowest_floor
        while current is not None:
            self.floor_order.append(current)
            current = floor_above_map.get(current) # Get the floor immediately above the current one

        # Create floor_name -> index map
        self.floor_to_index = {floor: i for i, floor in enumerate(self.floor_order)}

        # 2. Store goal locations for each passenger
        self.passenger_destinations = {}
        for fact in static_facts:
             if match(fact, "destin", "*", "*"):
                 p, f_destin = get_parts(fact)[1:]
                 self.passenger_destinations[p] = f_destin

        # Identify all passengers from destinations (assuming all passengers have a destination)
        self.all_passengers = set(self.passenger_destinations.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break
        # Assuming lift-at predicate is always present in a valid state

        # 2. Identify unserved passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_passengers = self.all_passengers - served_passengers

        if not unserved_passengers:
            return 0 # Goal state

        # 3. Categorize unserved passengers and collect relevant floors
        unserved_waiting = set()
        unserved_boarded = set()
        relevant_floors = set()

        # Add current lift floor if it's a valid floor
        if current_lift_floor in self.floor_to_index:
             relevant_floors.add(current_lift_floor)

        for fact in state:
            if match(fact, "origin", "*", "*"):
                p, f_origin = get_parts(fact)[1:]
                if p in unserved_passengers:
                    unserved_waiting.add(p)
                    if f_origin in self.floor_to_index:
                         relevant_floors.add(f_origin)
            elif match(fact, "boarded", "*"):
                 p = get_parts(fact)[1]
                 if p in unserved_passengers:
                     unserved_boarded.add(p)

        # Add destination floors for all unserved passengers
        for p in unserved_passengers:
             f_destin = self.passenger_destinations.get(p)
             if f_destin and f_destin in self.floor_to_index:
                 relevant_floors.add(f_destin)


        # 5. Count pending board actions
        board_actions_needed = len(unserved_waiting)

        # 6. Count pending depart actions
        depart_actions_needed = len(unserved_passengers)

        # 7. Estimate move actions
        relevant_floor_indices = {self.floor_to_index[f] for f in relevant_floors if f in self.floor_to_index}

        if len(relevant_floor_indices) <= 1: # 0 or 1 relevant floor means no moves needed between distinct floors
             move_actions_needed = 0
        else:
            min_idx = min(relevant_floor_indices)
            max_idx = max(relevant_floor_indices)
            # Assuming current_lift_floor is always a valid and mapped floor in a reachable state
            current_idx = self.floor_to_index[current_lift_floor]

            # Minimum moves to visit all floors in the range [min_idx, max_idx] starting from current_idx
            # This is the span + min distance to either end
            span = max_idx - min_idx
            move_actions_needed = span + min(abs(current_idx - min_idx), abs(current_idx - max_idx))

        # 8. Total heuristic
        total_cost = board_actions_needed + depart_actions_needed + move_actions_needed

        return total_cost
