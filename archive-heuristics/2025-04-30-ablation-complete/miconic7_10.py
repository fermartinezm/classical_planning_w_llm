from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
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

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total cost to serve all unserved passengers.
    It sums the estimated movement cost for the lift to visit all necessary floors
    (origins of waiting passengers, destinations of boarded passengers) and the
    estimated number of board and depart actions required for unserved passengers.

    # Assumptions
    - Floors are ordered numerically based on their names (e.g., f1 < f2 < f3).
    - The lift must visit the origin floor of a waiting passenger to board them.
    - The lift must visit the destination floor of a boarded passenger to drop them off.
    - The movement cost is estimated as the minimum travel distance to cover the range
      of required floors (from the lowest to the highest required floor) starting
      from the current lift position.
    - Each board and depart action costs 1.

    # Heuristic Initialization
    - Parses the floor ordering from the static facts to create a mapping from
      floor names to numerical indices. Assumes floor names are like 'f<number>'.
    - Stores the goal conditions (served passengers).
    - Stores the destination floor for each passenger from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state (all passengers served). If yes, return 0.
    2. Identify the lift's current floor.
    3. Identify all passengers who are currently waiting at their origin floors and are part of the goal.
    4. Identify all passengers who are currently boarded in the lift and are part of the goal.
    5. Determine the set of "required floors": this includes the origin floor for every
       waiting passenger and the destination floor for every boarded passenger.
    6. Calculate the "action cost": This is the sum of board and depart actions needed.
       Each waiting passenger needs 1 board and 1 depart (2 actions).
       Each boarded passenger needs 1 depart action.
       Total action cost = (number of waiting passengers * 2) + (number of boarded passengers * 1).
    7. Calculate the "movement cost":
       - If there are no required floors, the movement cost is 0.
       - Otherwise, find the minimum and maximum floor indices among the required floors.
       - Calculate the distance the lift must travel from its current floor to reach
         the minimum required floor and then sweep up to the maximum required floor,
         OR travel from its current floor to the maximum required floor and then
         sweep down to the minimum required floor. The movement cost is the minimum
         of these two travel distances. This is calculated as
         `min(abs(current_idx - min_req_idx) + (max_req_idx - min_req_idx), abs(current_idx - max_req_idx) + (max_req_idx - min_req_idx))`.
    8. The total heuristic value is the sum of the movement cost and the action cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        super().__init__(task)

        # 1. Parse floor order and create floor_to_index map
        # Find all floor objects by looking at all facts involving floors
        all_floors = set()
        # Check static facts
        for fact in self.static:
            parts = get_parts(fact)
            if parts:
                if parts[0] == 'above':
                    all_floors.add(parts[1])
                    all_floors.add(parts[2])
                elif parts[0] in ['origin', 'destin']: # origin/destin might involve floors in static init
                     if len(parts) == 3: # (predicate passenger floor)
                         all_floors.add(parts[2])

        # Check initial state facts (lift-at, origin)
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts:
                 if parts[0] == 'lift-at' and len(parts) == 2:
                     all_floors.add(parts[1])
                 elif parts[0] == 'origin' and len(parts) == 3:
                     all_floors.add(parts[2])

        # Check goal facts (served - no floor, but sometimes lift-at is in goal)
        for goal in self.goals:
             parts = get_parts(goal)
             if parts:
                 if parts[0] == 'lift-at' and len(parts) == 2:
                     all_floors.add(parts[1])


        # Sort floors numerically based on the number suffix (e.g., f1, f2, f10)
        # This assumes floor names are consistently 'f' followed by a number.
        try:
            self.ordered_floors = sorted(list(all_floors), key=lambda f: int(f[1:]))
        except (ValueError, IndexError):
             # Fallback or error handling if floor names are not f<number>
             # In a real system, might raise an error or use a more complex parser
             # For typical IPC Miconic, f<number> is standard.
             print("Warning: Floor names do not follow 'f<number>' pattern or are malformed. Sorting might be incorrect.")
             # Attempt sorting alphabetically as a fallback, though likely incorrect for numeric order
             self.ordered_floors = sorted(list(all_floors))


        self.floor_to_index = {floor: i for i, floor in enumerate(self.ordered_floors)}

        # 2. Store passenger destinations
        self.passenger_destinations = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, destination = get_parts(fact)
                self.passenger_destinations[passenger] = destination

        # 3. Store goal passengers (passengers who need to be served)
        self.goal_passengers = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                _, passenger = get_parts(goal)
                self.goal_passengers.add(passenger)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Check if the current state is a goal state
        if self.goals <= state:
            return 0

        # 2. Identify the lift's current floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break
        # If lift location is not found, state is likely invalid or terminal (goal handled above)
        if current_lift_floor is None or current_lift_floor not in self.floor_to_index:
             # This indicates an invalid state representation or problem definition
             # Return infinity as this state is likely unreachable or problematic
             return float('inf')

        current_idx = self.floor_to_index[current_lift_floor]


        # 3. Identify waiting passengers and their origin floors (only for goal passengers)
        waiting_passengers = set()
        pickup_floors = set()
        for fact in state:
            if match(fact, "origin", "*", "*"):
                _, passenger, origin_floor = get_parts(fact)
                # Only consider passengers who are goals (need to be served)
                if passenger in self.goal_passengers:
                    waiting_passengers.add(passenger)
                    if origin_floor in self.floor_to_index:
                        pickup_floors.add(origin_floor)
                    # else: origin floor not in known floors? Invalid state.
                    #    return float('inf')


        # 4. Identify boarded passengers and their destination floors (only for goal passengers)
        boarded_passengers = set()
        dropoff_floors = set()
        for fact in state:
            if match(fact, "boarded", "*"):
                _, passenger = get_parts(fact)
                 # Only consider passengers who are goals (need to be served)
                if passenger in self.goal_passengers:
                    boarded_passengers.add(passenger)
                    # Look up destination from pre-calculated map
                    destination_floor = self.passenger_destinations.get(passenger)
                    if destination_floor and destination_floor in self.floor_to_index:
                        dropoff_floors.add(destination_floor)
                    # else: Passenger boarded but no destination or destination unknown? Invalid state.
                    #    return float('inf')


        # 5. Determine the set of "required floors"
        all_required_floors = pickup_floors.union(dropoff_floors)

        # 6. Calculate the "action cost"
        # Each waiting passenger needs 1 board + 1 depart = 2 actions
        # Each boarded passenger needs 1 depart action
        action_cost = (len(waiting_passengers) * 2) + len(boarded_passengers)

        # 7. Calculate the "movement cost"
        movement_cost = 0
        if all_required_floors: # Only calculate movement if there are floors to visit
            required_indices = {self.floor_to_index[f] for f in all_required_floors}

            min_req_idx = min(required_indices)
            max_req_idx = max(required_indices)

            # Movement cost is the minimum travel to cover the range [min_req_idx, max_req_idx]
            # starting from current_idx.
            # Path 1: current -> min_req -> max_req
            cost1 = abs(current_idx - min_req_idx) + (max_req_idx - min_req_idx)
            # Path 2: current -> max_req -> min_req
            cost2 = abs(current_idx - max_req_idx) + (max_req_idx - min_req_idx)

            movement_cost = min(cost1, cost2)

        # 8. Total heuristic value
        total_heuristic = movement_cost + action_cost

        return total_heuristic
