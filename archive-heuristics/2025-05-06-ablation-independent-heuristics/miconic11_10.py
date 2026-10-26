from heuristics.heuristic_base import Heuristic
from task import Task
import re
import math

# Helper function to parse facts
def parse_fact(fact_str):
    """Parses a fact string like '(predicate arg1 arg2)' into (predicate, [arg1, arg2])."""
    # Remove surrounding brackets
    fact_str = fact_str.strip()
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
         return None, [] # Indicate parsing failure
    fact_str = fact_str[1:-1].strip()
    if not fact_str: # Handle empty fact string inside brackets
        return None, []
    parts = fact_str.split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

# Helper function to get floor number for sorting
def get_floor_number(floor_name):
    """Extracts the numerical part from a floor name like 'f10'."""
    match = re.match(r'f(\d+)', floor_name)
    if match:
        return int(match.group(1))
    # Assign a large number to put floors with non-standard names last
    return float('inf')

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the miconic domain.

    Summary:
    Estimates the cost to reach the goal by summing the number of board actions
    needed, the number of depart actions needed, and the estimated minimum
    movement actions required by the lift to visit all necessary floors.
    - Board actions needed: Number of passengers currently waiting at their origin.
    - Depart actions needed: Number of passengers not yet served (waiting or boarded).
    - Movement actions needed: Estimated minimum moves for the lift to travel
      from its current floor to visit all origin floors of waiting passengers
      and all destination floors of boarded passengers. This is calculated as
      the distance from the current floor to the closer of the minimum or maximum
      required floor index plus the distance between the minimum and maximum
      required floor indices.

    Assumptions:
    - Floor names follow the pattern 'f<number>', allowing numerical sorting.
    - The 'above' predicate defines a linear ordering of floors consistent with
      the numerical suffix of floor names (e.g., (above f1 f2) implies f1 is below f2).
    - Passenger names are unique.
    - Each passenger has exactly one origin (in initial state or current state)
      and one destination (in static facts).
    - The '(lift-at ?f)' fact is always present in a valid state.
    - The goal is typically a conjunction of '(served ?p)' facts for all passengers.

    Heuristic Initialization:
    The constructor parses the static facts and initial state to:
    1. Identify all floor names and create a mapping from floor name string
       to an integer index based on their sorted numerical suffix.
    2. Store the destination floor for each passenger by parsing the 'destin'
       facts from the static information.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Check if the state is a goal state by verifying if all facts in `self.task.goals`
       are present in the state. If yes, return 0.
    2. Identify the current floor of the lift by finding the '(lift-at ?f)' fact.
       Convert the floor name to its integer index using the precomputed mapping.
       If lift location is not found or floor name is unknown, return infinity.
    3. Initialize counters for waiting and boarded passengers to zero.
    4. Initialize sets for required pickup floors and required dropoff floors to empty.
    5. Iterate through all facts in the current state:
       - If a fact is '(origin p o)', parse 'p' and 'o'. Increment the waiting passenger count and add floor 'o' to the set of required pickup floors.
       - If a fact is '(boarded p)', parse 'p'. Increment the boarded passenger count. Look up the destination 'd' for passenger 'p' using the precomputed passenger destinations mapping and add floor 'd' to the set of required dropoff floors.
    6. Combine the required pickup and dropoff floors into a single set of 'required stops'.
    7. If the set of required stops is empty (which should only happen in goal states for valid problems), return infinity as a defensive measure against potentially unresolvable states.
    8. Calculate the number of board actions needed: This is equal to the count of waiting passengers.
    9. Calculate the number of depart actions needed: This is equal to the total number of unserved passengers, which is the sum of waiting and boarded passengers.
    10. Calculate the estimated movement actions needed:
       - Find the minimum and maximum floor indices among the required stops using the precomputed floor index mapping.
       - The minimum movement cost is the distance from the current lift floor index to the closer of the minimum or maximum required floor index, plus the distance between the minimum and maximum required floor indices.
         Movement Cost = min(abs(current_floor_idx - min_required_idx), abs(current_floor_idx - max_required_idx)) + (max_required_idx - min_required_idx).
    11. The total heuristic value is the sum of board actions needed, depart actions needed, and estimated movement actions needed.
       Heuristic Value = waiting_passengers + (waiting_passengers + boarded_passengers) + movement_cost
                     = 2 * waiting_passengers + boarded_passengers + movement_cost.
    12. Return the calculated heuristic value.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.floor_to_idx = {}
        self.passenger_destinations = {}

        # 1. Identify all floor names and create floor_to_idx mapping
        all_floor_names = set()
        # Look for floors in initial state and static facts
        for fact_str in task.static | task.initial_state:
            predicate, args = parse_fact(fact_str)
            if predicate in ['above', 'origin', 'destin', 'lift-at']:
                # Floors are arguments in these predicates
                for arg in args:
                    # Simple check if the argument looks like a floor name
                    if isinstance(arg, str) and arg.startswith('f'):
                         all_floor_names.add(arg)

        # Sort floor names numerically
        sorted_floor_names = sorted(list(all_floor_names), key=get_floor_number)
        self.floor_to_idx = {floor_name: idx for idx, floor_name in enumerate(sorted_floor_names)}

        # 2. Store passenger destinations
        for fact_str in task.static:
            predicate, args = parse_fact(fact_str)
            if predicate == 'destin':
                # Fact is '(destin p d)'
                if len(args) == 2:
                    passenger_name = args[0]
                    destination_floor = args[1]
                    self.passenger_destinations[passenger_name] = destination_floor
                # else: log warning about malformed destin fact?

    def __call__(self, node):
        state = node.state

        # 1. Check if the state is a goal state
        if self.task.goals.issubset(state):
             return 0

        # 2. Identify current lift floor
        current_lift_floor_str = None
        for fact_str in state:
            predicate, args = parse_fact(fact_str)
            if predicate == 'lift-at':
                if len(args) == 1:
                    current_lift_floor_str = args[0]
                    break

        if current_lift_floor_str is None:
             # This state is likely invalid or a dead end if lift location is unknown.
             return float('inf')

        current_floor_idx = self.floor_to_idx.get(current_lift_floor_str)
        if current_floor_idx is None:
             # Should not happen if floor_to_idx is built correctly and state is valid
             return float('inf')


        # 3. & 4. Initialize counters and sets
        waiting_passengers = 0
        boarded_passengers = 0
        pickup_floors = set()
        dropoff_floors = set()

        # 5. Iterate through state facts
        for fact_str in state:
            predicate, args = parse_fact(fact_str)
            if predicate == 'origin':
                # Fact is '(origin p o)'
                if len(args) == 2:
                    passenger_name = args[0]
                    origin_floor = args[1]
                    waiting_passengers += 1
                    pickup_floors.add(origin_floor)
            elif predicate == 'boarded':
                # Fact is '(boarded p)'
                if len(args) == 1:
                    passenger_name = args[0]
                    boarded_passengers += 1
                    # Get destination from precomputed map
                    destination_floor = self.passenger_destinations.get(passenger_name)
                    if destination_floor: # Should always exist for a valid problem
                        dropoff_floors.add(destination_floor)
                    # else: log warning about boarded passenger with no destination?

        # 6. Combine required stops
        required_stops = pickup_floors | dropoff_floors

        # 7. Defensive check: If not goal state but no required stops, something is wrong
        # This check is important. If there are unserved passengers but no required stops,
        # it means the heuristic calculation based on required stops breaks down.
        # E.g., maybe a passenger is waiting but their origin floor is not in floor_to_idx?
        # Or a boarded passenger has no destination?
        # In a well-formed problem, this shouldn't happen if not in a goal state.
        if not required_stops:
             return float('inf')

        # 8. & 9. Calculated implicitly in the final sum

        # 10. Calculate estimated movement actions
        required_indices = {self.floor_to_idx[f] for f in required_stops if f in self.floor_to_idx}
        if not required_indices: # Should not happen if required_stops is not empty and floors are valid
             return float('inf') # Defensive check

        min_required_idx = min(required_indices)
        max_required_idx = max(required_indices)

        # Movement cost = distance to closest extreme + distance between extremes
        movement_cost = min(abs(current_floor_idx - min_required_idx), abs(current_floor_idx - max_required_idx)) + (max_required_idx - min_required_idx)

        # 11. Calculate total heuristic value
        # Heuristic = board_actions_needed + depart_actions_needed + movement_cost
        # Heuristic = waiting_passengers + (waiting_passengers + boarded_passengers) + movement_cost
        heuristic_value = 2 * waiting_passengers + boarded_passengers + movement_cost

        # 12. Return the calculated heuristic value.
        return heuristic_value
