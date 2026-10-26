from heuristics.heuristic_base import Heuristic
from task import Task
import math # Import math for float('inf')

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into a list of strings."""
    # Remove surrounding brackets and split by space
    parts = fact_string[1:-1].split()
    return parts

# Helper function to extract number from floor name
def get_floor_number(floor_name):
    """Extracts the number from a floor name like 'f1' or 'f10'."""
    # Assumes floor names are 'f' followed by a number
    try:
        return int(floor_name[1:])
    except ValueError:
        # Should not happen with standard miconic problems based on examples
        raise ValueError(f"Unexpected floor name format: {floor_name}")


class miconicHeuristic(Heuristic):
    """
    Summary:
    A domain-dependent heuristic for the Miconic domain. It estimates the cost
    to reach the goal state (all passengers served) by summing the estimated
    travel cost for the lift and the number of remaining board/depart actions.
    The travel cost is estimated as the minimum distance required for the lift
    to traverse the range of floors where unserved passengers need to be picked
    up or dropped off, starting from the current lift floor. The action cost
    is simply the number of passengers who are currently waiting at their origin
    or are boarded in the lift.

    Assumptions:
    - Floor names follow the pattern 'f' followed by a number (e.g., 'f1', 'f10').
    - The 'above' predicates in the static facts define a total order of floors
      such that '(above fi fj)' implies floor fi is immediately below floor fj,
      and floors are ordered numerically by their index (e.g., f1 < f2 < ...).
    - The goal is to serve all passengers.
    - Problem instances are solvable and states conform to domain rules
      (e.g., a passenger is either at origin, boarded, or served, but not both
      origin and boarded, etc.).

    Heuristic Initialization:
    In the constructor, the heuristic pre-processes static information from the
    task description:
    1. It identifies all floor objects by parsing 'above' facts from the static
       information.
    2. It sorts the floors based on the numerical part of their names to establish
       a consistent floor order.
    3. It creates mappings between floor names (strings) and their corresponding
       numerical indices in the sorted order.
    4. It identifies all passenger objects by parsing 'destin' facts from the
       static information.
    5. It creates a mapping from each passenger to their destination floor, also
       parsed from the static 'destin' facts. It also collects passenger names
       from initial state 'origin' facts to ensure all passengers are known.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state (node):
    1. Identify the current floor of the lift by finding the '(lift-at ?f)' fact
       in the state.
    2. Initialize sets to store the origin floors of waiting passengers and the
       destination floors of boarded passengers.
    3. Initialize counters for the number of waiting passengers and boarded passengers.
    4. Initialize a set to track served passengers.
    5. Iterate through all facts in the current state:
       - If a fact is '(lift-at ?f)', store ?f as the current lift floor.
       - If a fact is '(origin ?p ?f)', add floor ?f to the set of origin floors
         needed and increment the waiting passenger count.
       - If a fact is '(boarded ?p)', look up the destination floor for passenger ?p
         using the pre-computed passenger-destination map. Add this destination
         floor to the set of destination floors needed and increment the boarded
         passenger count.
       - If a fact is '(served ?p)', add ?p to the set of served passengers.
    6. Check if the number of served passengers equals the total number of passengers
       known from initialization. If yes, the goal is reached, and the heuristic
       value is 0.
    7. If the goal is not reached:
       - Combine the sets of origin floors needed and destination floors needed to get
         the set of all floors that must be visited to pick up or drop off passengers.
       - If this combined set of needed floors is empty (which should not happen
         in a solvable non-goal state given domain rules), return a large value
         like float('inf') to indicate a likely dead end or invalid state.
       - If the set of needed floors is not empty:
         - Convert the needed floor names to their numerical indices using the
           pre-computed mapping.
         - Find the minimum and maximum indices among the needed floors.
         - Get the index of the current lift floor.
         - Calculate the estimated travel cost: `min(abs(current_idx - min_needed_idx),
           abs(current_idx - max_needed_idx)) + (max_needed_idx - min_needed_idx)`.
         - Calculate the estimated action cost: `num_waiting + num_boarded`.
         - The total heuristic value is the sum of the estimated travel cost and
           the estimated action cost.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task # Store task for access to static info etc.

        self.floor_to_idx = {}
        self.idx_to_floor = {}
        self.passengers = set()
        self.passenger_dest = {}

        # 1. Collect all floor names from static 'above' facts
        floor_names = set()
        for fact_string in task.static:
            parts = parse_fact(fact_string)
            if parts[0] == 'above':
                # parts[1] is above parts[2], so parts[2] is lower than parts[1]
                floor_names.add(parts[1])
                floor_names.add(parts[2])

        # 2. Sort floor names numerically and create mappings
        # Assuming 'f1' < 'f2' < ... < 'f10' etc.
        sorted_floor_names = sorted(list(floor_names), key=get_floor_number)
        for idx, floor_name in enumerate(sorted_floor_names):
            self.floor_to_idx[floor_name] = idx
            self.idx_to_floor[idx] = floor_name

        # 3. Collect all passenger names and destinations from static 'destin' facts
        for fact_string in task.static:
            parts = parse_fact(fact_string)
            if parts[0] == 'destin':
                passenger_name = parts[1]
                dest_floor = parts[2]
                self.passengers.add(passenger_name)
                self.passenger_dest[passenger_name] = dest_floor

        # Also collect passengers from initial state origin facts, just in case
        # a passenger exists but has no destin fact (unlikely in PDDL, but robust)
        for fact_string in task.initial_state:
             parts = parse_fact(fact_string)
             if parts[0] == 'origin':
                 passenger_name = parts[1]
                 self.passengers.add(passenger_name)


    def __call__(self, node):
        state = node.state

        current_floor = None
        origin_floors_needed = set()
        destin_floors_needed = set()
        num_waiting = 0
        num_boarded = 0
        served_passengers = set()

        # 5. Iterate through state facts to find current lift floor and passenger statuses
        for fact_string in state:
            parts = parse_fact(fact_string)
            predicate = parts[0]

            if predicate == 'lift-at':
                current_floor = parts[1]
            elif predicate == 'origin':
                passenger_name = parts[1]
                origin_floor = parts[2]
                origin_floors_needed.add(origin_floor)
                num_waiting += 1
            elif predicate == 'boarded':
                passenger_name = parts[1]
                # Passenger is boarded, needs to go to destination
                if passenger_name in self.passenger_dest: # Should always be true in valid state
                    destin_floors_needed.add(self.passenger_dest[passenger_name])
                    num_boarded += 1
            elif predicate == 'served':
                 served_passengers.add(parts[1])

        # 6. Check if all passengers are served (goal state)
        if len(served_passengers) == len(self.passengers):
             return 0

        # 7. If the goal is not reached:
        # Combine needed floors
        all_needed_floors = origin_floors_needed | destin_floors_needed

        # If no floors are needed, but not all passengers are served, something is wrong.
        # This implies unserved passengers are neither at origin nor boarded.
        # Return infinity to prune this branch.
        if not all_needed_floors:
             return float('inf') # Should not happen in solvable non-goal states

        # Calculate heuristic components
        needed_indices = {self.floor_to_idx[f] for f in all_needed_floors}
        min_needed_idx = min(needed_indices)
        max_needed_idx = max(needed_indices)
        current_idx = self.floor_to_idx[current_floor]

        # Calculate travel cost
        # Minimum distance to reach the range [min_needed_idx, max_needed_idx]
        # starting from current_idx, plus the distance to traverse the range.
        travel_cost = min(abs(current_idx - min_needed_idx), abs(current_idx - max_needed_idx)) + (max_needed_idx - min_needed_idx)

        # Calculate action cost (board + depart for unserved passengers)
        action_cost = num_waiting + num_boarded

        # Total heuristic
        h_value = travel_cost + action_cost

        return h_value
