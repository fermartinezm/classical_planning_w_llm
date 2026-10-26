from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
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
    # The number of parts in the fact must match the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the number of actions (board, depart, move) required to serve
    all passengers. It counts the number of pending board/depart actions
    and adds an estimate of the minimum vertical movement needed to visit
    all relevant floors (origins of waiting passengers, destinations of
    boarded passengers).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # 1. Parse floor order and create floor_to_index map
        # The 'above' predicate defines the order: (above f_higher f_lower)
        below_relations = {} # {floor_higher: floor_lower}
        all_floors = set()
        floors_that_are_lower_in_above_facts = set() # These are floors with something above them

        for fact in self.static_facts:
            if match(fact, "above", "*", "*"):
                f_higher, f_lower = get_parts(fact)[1], get_parts(fact)[2]
                below_relations[f_higher] = f_lower
                all_floors.add(f_lower)
                all_floors.add(f_higher)
                floors_that_are_lower_in_above_facts.add(f_lower)

        # Find the highest floor (a floor that is not the lower floor in any 'above' relation)
        # This assumes there is at least one floor.
        # Handle case with only one floor
        if not all_floors:
             self.floor_order = []
             self.floor_to_index = {}
        else:
            highest_floor = (all_floors - floors_that_are_lower_in_above_facts).pop()

            # Build the ordered list of floors (highest to lowest) and the index map
            self.floor_order = [] # List of floor names, highest first
            self.floor_to_index = {} # Map floor name to index (0 for highest, 1 for next lower, etc.)
            current_floor = highest_floor
            index = 0
            while current_floor is not None:
                self.floor_order.append(current_floor)
                self.floor_to_index[current_floor] = index
                index += 1
                current_floor = below_relations.get(current_floor) # Get the floor directly below

            # Reverse the order and indices so index 0 is the lowest floor
            self.floor_order.reverse()
            self.floor_to_index = {floor: len(self.floor_order) - 1 - index for floor, index in self.floor_to_index.items()}


        # 2. Store passenger destinations and identify all passengers
        self.passenger_destinations = {}
        self.all_passengers = set()

        # Passenger names come from goal facts (served) and static facts (destin)
        for goal in self.goals:
            if match(goal, "served", "*"):
                passenger = get_parts(goal)[1]
                self.all_passengers.add(passenger)

        for fact in self.static_facts:
             if match(fact, "destin", "*", "*"):
                 p, f = get_parts(fact)[1], get_parts(fact)[2]
                 self.passenger_destinations[p] = f
                 self.all_passengers.add(p) # Ensure passenger is added even if not in goal (unlikely in miconic)

        # Ensure we have destinations for all passengers identified
        # assert self.all_passengers.issubset(self.passenger_destinations.keys())


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # If the goal is reached, the heuristic is 0.
        if self.goals <= state:
            return 0

        # 1. Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break
        # If lift location is not found, something is wrong with the state representation.
        # Assuming lift-at fact is always present in valid states.
        # If there's only one floor, current_lift_floor might be None if all_floors was empty.
        # Handle the single-floor case explicitly if needed, but the logic below should work
        # as required_stop_indices will likely be empty or contain only the single floor's index.
        if current_lift_floor is None and self.floor_order:
             # This shouldn't happen in a valid miconic state with >1 floor
             # For a single floor problem, the only floor is the current one.
             current_lift_floor = self.floor_order[0] # Assume the single floor
        elif current_lift_floor is None and not self.floor_order:
             # Problem with no floors defined? Or single floor not in above facts?
             # This case is unexpected for typical miconic problems. Return infinity? Or 0 if no passengers?
             # Let's assume valid miconic problems have floors and lift-at.
             pass # Will raise KeyError if current_lift_floor is None and used in floor_to_index

        current_lift_index = self.floor_to_index.get(current_lift_floor, 0) # Default to 0 if floor not found (shouldn't happen)


        # 2. Identify unserved passengers and their state (waiting or boarded)
        unserved_passengers_waiting = {} # {passenger: origin_floor}
        unserved_passengers_boarded = set() # {passenger}

        served_passengers = set()
        for fact in state:
            if match(fact, "served", "*"):
                served_passengers.add(get_parts(fact)[1])

        for passenger in self.all_passengers:
            if passenger not in served_passengers:
                # Check if waiting at origin
                is_waiting = False
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                        origin_floor = get_parts(fact)[2]
                        unserved_passengers_waiting[passenger] = origin_floor
                        is_waiting = True
                        break
                # Check if boarded
                if not is_waiting:
                     # If not served and not waiting, they must be boarded
                     # Check explicitly just in case, though domain invariant should hold
                     if f"(boarded {passenger})" in state:
                         unserved_passengers_boarded.add(passenger)
                     # else: Invalid state or passenger doesn't need service (not in goal)
                     # Assuming passengers not in goal don't exist or don't need service.
                     # Assuming passengers in goal are either waiting, boarded, or served.


        # 3. Calculate heuristic

        # Count pending board and depart actions
        h_actions = len(unserved_passengers_waiting) + len(unserved_passengers_boarded)

        # Estimate movement cost
        # The lift needs to visit all origins of waiting passengers and all destinations of boarded passengers.
        required_stop_indices = set()
        for passenger, origin_floor in unserved_passengers_waiting.items():
             # Ensure origin_floor is in our floor map (should be if problem is well-formed)
             if origin_floor in self.floor_to_index:
                 required_stop_indices.add(self.floor_to_index[origin_floor])
        for passenger in unserved_passengers_boarded:
             destin_floor = self.passenger_destinations.get(passenger)
             # Ensure destination floor exists and is in our floor map
             if destin_floor and destin_floor in self.floor_to_index:
                 required_stop_indices.add(self.floor_to_index[destin_floor])

        estimated_movement_cost = 0
        if required_stop_indices:
            min_idx = min(required_stop_indices)
            max_idx = max(required_stop_indices)
            current_idx = current_lift_index

            # Minimum moves to visit all stops in the range [min_idx, max_idx] starting from current_idx
            # This is the distance to one end of the required range plus the width of the range.
            cost_via_min = abs(current_idx - min_idx) + (max_idx - min_idx)
            cost_via_max = abs(current_idx - max_idx) + (max_idx - min_idx)
            estimated_movement_cost = min(cost_via_min, cost_via_max)

        # Total heuristic is the sum of pending actions and estimated movement
        h = h_actions + estimated_movement_cost

        return h
