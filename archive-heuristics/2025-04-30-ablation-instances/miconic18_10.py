from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all passengers
    to their destination floors. It combines the estimated lift movement cost to visit
    all necessary floors (origins of waiting passengers and destinations of boarded passengers)
    with the number of 'board' and 'depart' actions required for unserved passengers.

    # Assumptions
    - All passengers need to be served (reach their destination).
    - The lift can carry multiple passengers.
    - Floor levels are ordered linearly based on the 'above' predicate.
    - Action costs are 1.

    # Heuristic Initialization
    - Parse the 'above' facts to determine the floor order and create mappings
      between floor names and their numerical indices.
    - Parse the 'destin' facts to create a mapping from each passenger
      to their destination floor.
    - Identify the set of all passenger names in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state (all passengers are served). If yes, the heuristic value is 0.
    2. Identify the current floor of the lift by finding the fact `(lift-at ?f)` in the state.
    3. Initialize sets to store floors that need to be visited: `pickup_floors` (origins of waiting passengers) and `dropoff_floors` (destinations of boarded passengers).
    4. Initialize counters for passengers: `N_waiting` (passengers at origin floors) and `N_boarded` (passengers currently inside the lift).
    5. Iterate through all known passengers in the problem:
       - Check if the passenger is *not* served (`(served ?p)` is not in the state).
       - If unserved:
         - Check if the passenger is boarded (`(boarded ?p)` is in the state). If yes, increment `N_boarded` and add their destination floor (looked up from the precomputed destination map) to `dropoff_floors`.
         - If the passenger is not boarded, they must be waiting at an origin floor (`(origin ?p ?f)` is in the state). Find this origin floor `?f`, increment `N_waiting`, and add `?f` to `pickup_floors`.
    6. Combine `pickup_floors` and `dropoff_floors` into a single set `FloorsToVisit`.
    7. Calculate the estimated minimum number of move actions (`move_cost`):
       - If `FloorsToVisit` is empty (which implies all unserved passengers are either in an invalid state or the goal is reached), the move cost is 0.
       - Otherwise, find the minimum and maximum floor indices among the floors in `FloorsToVisit`.
       - Get the index of the current lift floor.
       - The minimum moves to cover the range of necessary floors starting from the current floor is `min(abs(current_floor_index - min_floor_index), abs(current_floor_index - max_floor_index)) + (max_floor_index - min_floor_index)`.
    8. Calculate the estimated number of non-move actions (`action_cost`):
       - Each passenger waiting at an origin needs a 'board' action. Total board actions = `N_waiting`.
       - Each unserved passenger (waiting or boarded) needs a 'depart' action at their destination. Total unserved passengers = `N_waiting + N_boarded`. Total depart actions = `N_waiting + N_boarded`.
       - Total action cost = `N_waiting` (boards) + `(N_waiting + N_boarded)` (departs).
    9. The total heuristic value is the sum of the estimated move cost and the estimated action cost: `move_cost + N_waiting + (N_waiting + N_boarded)`.

    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information."""
        self.goals = task.goals
        self.static_facts = task.static
        self.all_passengers = set()

        # 1. Parse floor order and create index mapping
        above_facts = [get_parts(fact) for fact in self.static_facts if match(fact, "above", "*", "*")]
        floors_above = {f1: f2 for _, f1, f2 in above_facts}
        floors_below = {f2: f1 for f1, f2 in floors_above.items()}

        # Find the bottom floor (a floor that is not a 'second' argument in any 'above' fact)
        all_floors_in_above = set(floors_above.keys()) | set(floors_above.values())
        bottom_floor = None
        # Find a floor that is mentioned as f1 but never as f2
        potential_bottoms = set(floors_above.keys()) - set(floors_above.values())
        if potential_bottoms:
             bottom_floor = list(potential_bottoms)[0] # Assuming a single connected component

        # Build the ordered list of floors and the index map
        self.ordered_floors = []
        self.floor_to_index = {}
        current_floor = bottom_floor
        index = 0
        while current_floor is not None:
            self.ordered_floors.append(current_floor)
            self.floor_to_index[current_floor] = index
            index += 1
            current_floor = floors_above.get(current_floor)

        # 2. Parse passenger destinations
        self.passenger_destinations = {}
        for fact in self.static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, destination = get_parts(fact)
                self.passenger_destinations[passenger] = destination
                self.all_passengers.add(passenger)

        # 3. Identify all passenger names (also get from initial state or goals if not in destin)
        # Add passengers mentioned in initial state or goals
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] in ["origin", "boarded", "served"] and len(parts) > 1:
                 self.all_passengers.add(parts[1])
        for fact in task.goals:
             parts = get_parts(fact)
             if parts[0] == "served" and len(parts) > 1:
                 self.all_passengers.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Check for goal state
        # A more robust goal check is using the task's method
        if self.goals <= state:
             return 0

        # 2. Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break
        # If lift location is not specified, heuristic is infinite (or a large value)
        # but assuming valid states always have lift-at
        if current_lift_floor is None:
             # This should not happen in valid miconic states, but as a fallback:
             # print("Warning: Lift location not found in state.")
             return float('inf') # Or handle as an invalid state

        current_floor_index = self.floor_to_index[current_lift_floor]

        # 3. Identify necessary floors and count passengers
        pickup_floors = set()
        dropoff_floors = set()
        N_waiting = 0
        N_boarded = 0

        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        boarded_passengers_in_state = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}
        origin_facts_in_state = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "origin", "*", "*")}

        unserved_passengers = set()
        for passenger in self.all_passengers:
            if passenger not in served_passengers:
                unserved_passengers.add(passenger)
                if passenger in boarded_passengers_in_state:
                    N_boarded += 1
                    # Add destination floor to dropoff floors
                    if passenger in self.passenger_destinations: # Should always be true in valid problems
                         dropoff_floors.add(self.passenger_destinations[passenger])
                    # else: boarded passenger without destination? Invalid problem.
                elif passenger in origin_facts_in_state:
                    N_waiting += 1
                    # Add origin floor to pickup floors
                    pickup_floors.add(origin_facts_in_state[passenger])
                # else: passenger is unserved but neither boarded nor at origin? Invalid state.

        N_unserved = len(unserved_passengers)

        # 6. Combine necessary floors
        floors_to_visit = pickup_floors | dropoff_floors

        # 7. Calculate move cost
        move_cost = 0
        if floors_to_visit:
            min_idx = min(self.floor_to_index[f] for f in floors_to_visit)
            max_idx = max(self.floor_to_index[f] for f in floors_to_visit)
            move_cost = min(abs(current_floor_index - min_idx), abs(current_floor_index - max_idx)) + (max_idx - min_idx)

        # 8. Calculate action cost
        # Each waiting passenger needs 1 board action
        # Each unserved passenger needs 1 depart action
        action_cost = N_waiting + N_unserved

        # 9. Total heuristic
        total_heuristic = move_cost + action_cost

        return total_heuristic
