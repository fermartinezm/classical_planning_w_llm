from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re # To parse floor numbers

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string if necessary, though PDDL facts are not empty.
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
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    The heuristic estimates the number of necessary actions (board, depart, and lift movements)
    to get all unserved passengers to their destination floors. It counts each required
    board and depart action once and estimates the minimum lift movement cost to visit
    all necessary floors (origins of waiting passengers and destinations of boarded passengers).

    # Assumptions
    - Floors are named 'f1', 'f2', ..., 'fn' and correspond to integer levels 1, 2, ..., n.
    - The 'above' predicate implies a linear ordering of floors, consistent with f<i> mapping to level i.
    - The lift has unlimited capacity.
    - The cost of moving between adjacent floors is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.

    # Heuristic Initialization
    - Identify all passengers and their destination floors from the static facts and goals.
    - Identify all floors and create a mapping from floor name to integer level based on the number in the name.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current floor of the lift from the state. Map this floor name to its integer level using the precomputed mapping.
    2. Initialize counters for required board actions and required depart actions to zero. Initialize sets for required pickup floor levels and required dropoff floor levels to empty.
    3. Iterate through all passengers identified during initialization. For each passenger:
       - Check if the passenger is already served by looking for the `(served passenger)` fact in the current state. If served, skip this passenger.
       - If the passenger is not served, check if they are waiting at their origin floor by looking for the `(origin passenger floor)` fact in the current state. If found:
         - Increment the required board actions counter by 1.
         - Add the integer level of their origin floor to the set of required pickup levels.
       - If the passenger is not served and not waiting at their origin, check if they are boarded in the lift by looking for the `(boarded passenger)` fact in the current state. If found:
         - Increment the required depart actions counter by 1.
         - Retrieve the passenger's destination floor (precomputed during initialization). Add the integer level of this destination floor to the set of required dropoff levels.
    4. Combine the required pickup levels and required dropoff levels into a single set of all required floor levels the lift must visit.
    5. Calculate the estimated lift movement cost:
       - If the set of required levels is empty, the move cost is 0. This occurs only when all unserved passengers are already at their destination (which means they are served, handled in step 2).
       - If the set of required levels is not empty, find the minimum and maximum levels within this set.
       - The estimated move cost is calculated as the distance between the minimum and maximum required levels (`max_required_level - min_required_level`) plus the minimum distance from the current lift level to either the minimum or maximum required level (`min(abs(current_level - min_required_level), abs(current_level - max_required_level))`). This represents the cost of sweeping through the required floors after reaching the closest end of the range.
    6. The total heuristic value for the state is the sum of the required board actions, the required depart actions, and the estimated lift movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels, passenger destinations,
        and identifying all passengers.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        self.destinations = {}
        self.passengers = set()
        floor_names = set()

        # Extract destinations and identify passengers/floors from static and goals
        for fact in static_facts.union(self.goals):
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "destin":
                # Ensure fact has enough parts before accessing indices
                if len(parts) > 2:
                    passenger, floor = parts[1], parts[2]
                    self.destinations[passenger] = floor
                    self.passengers.add(passenger)
                    floor_names.add(floor)
            elif predicate in ["above", "lift-at", "origin", "boarded", "served"]:
                 # Add objects from other relevant predicates
                 # Check parts length before iterating
                 if len(parts) > 1:
                     for obj in parts[1:]:
                         if obj.startswith('p'): self.passengers.add(obj)
                         if obj.startswith('f'): floor_names.add(obj)

        # Also check initial state for any objects not in static/goals (unlikely but safe)
        for fact in initial_state:
             parts = get_parts(fact)
             if not parts: continue
             if len(parts) > 1:
                 for obj in parts[1:]:
                     if obj.startswith('p'): self.passengers.add(obj)
                     if obj.startswith('f'): floor_names.add(obj)


        # Create floor name to level mapping
        # Assumes floors are named f1, f2, ... and correspond to levels 1, 2, ...
        # Robustly handle floor names that might not strictly follow f<number>
        # Filter names that look like floors and sort them numerically
        valid_floor_names = sorted([f for f in floor_names if re.match(r'^f\d+$', f)],
                                   key=lambda f: int(f[1:]))

        self.floor_to_level = {floor_name: i + 1 for i, floor_name in enumerate(valid_floor_names)}
        # self.level_to_floor = {i + 1: floor_name for i, floor_name in enumerate(valid_floor_names)} # Not strictly needed for this heuristic


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 2. Check if goal is reached
        # Goal is reached if all passengers identified in __init__ are served
        all_served = True
        for p in self.passengers:
            if f"(served {p})" not in state:
                all_served = False
                break
        if all_served:
             return 0 # Heuristic is 0 in goal states

        # 1. Get current lift floor level
        current_lift_floor_name = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor_name = get_parts(fact)[1]
                break

        # If lift location is not found or floor name is not in our mapping,
        # the state is likely invalid or represents an unhandleable scenario.
        # Return infinity to prune this path.
        if current_lift_floor_name not in self.floor_to_level:
             return float('inf')

        current_level = self.floor_to_level[current_lift_floor_name]


        # 3. Identify unserved passengers and required actions/stops
        board_actions_needed = 0
        depart_actions_needed = 0
        pickup_levels = set()
        dropoff_levels = set()

        # Create sets/dicts for quick lookup of passenger states
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        origin_passengers = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "origin", "*", "*")}
        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}


        for passenger in self.passengers:
            if passenger in served_passengers:
                continue # Passenger is already served

            # Passenger is unserved
            if passenger in origin_passengers:
                # Passenger is waiting at origin
                board_actions_needed += 1
                origin_floor = origin_passengers[passenger]
                if origin_floor in self.floor_to_level:
                    pickup_levels.add(self.floor_to_level[origin_floor])

            elif passenger in boarded_passengers:
                # Passenger is boarded
                depart_actions_needed += 1
                # Need destination floor for boarded passenger
                destin_floor = self.destinations.get(passenger)
                if destin_floor and destin_floor in self.floor_to_level:
                     dropoff_levels.add(self.floor_to_level[destin_floor])

            # else: Passenger is unserved but neither at origin nor boarded? Invalid state?
                  # This case should ideally not occur in valid problem states.
                  # We could potentially add a large penalty here if desired,
                  # but for standard problems, this branch won't be taken for unserved passengers.


        # 4. Combine required stops
        required_levels = pickup_levels.union(dropoff_levels)

        # 5. Calculate estimated lift movement cost
        move_cost = 0
        if required_levels:
            min_req_level = min(required_levels)
            max_req_level = max(required_levels)

            # Cost to reach the range of required floors + cost to sweep the range
            # This is the minimum vertical distance to visit all floors in required_levels
            # starting from current_level, assuming monotonic travel within the range.
            move_cost = (max_req_level - min_req_level) + min(abs(current_level - min_req_level), abs(current_level - max_req_level))

        # 6. Total heuristic value
        total_cost = board_actions_needed + depart_actions_needed + move_cost

        return total_cost
