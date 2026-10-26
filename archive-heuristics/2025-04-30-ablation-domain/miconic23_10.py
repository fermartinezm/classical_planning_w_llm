# Helper functions from Logistics example
from fnmatch import fnmatch
import re # Needed for parsing floor numbers

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or fact[0] != '(' or fact[-1] != ')':
         return [] # Return empty list for invalid format
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Heuristic class
from heuristics.heuristic_base import Heuristic

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It sums the required board/depart actions for each unserved passenger,
    adds the number of unique floors the lift must stop at (origins for waiting,
    destinations for boarded), and adds the estimated lift movement cost
    based on the span of required floors including the current lift floor.

    # Assumptions
    - Floors are linearly ordered and named like 'f1', 'f2', etc., where the number
      indicates the floor level (f1 is the lowest, f20 is the highest).
    - Passengers need to be picked up at their origin and dropped off at their destination.
    - The goal is reached when all passengers are in the 'served' state.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from static facts.
    - Creates a mapping from floor names (e.g., 'f5') to integer indices (e.g., 5).
    - Identifies the total set of passengers in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify the current floor of the lift. Map it to its integer index.
    2.  Identify all passengers who are not yet in the 'served' state by comparing
        the total set of passengers with those currently in the 'served' state.
    3.  Categorize unserved passengers into 'waiting' (at their origin floor)
        and 'boarded' (inside the lift) based on the current state facts.
    4.  Calculate the base action cost:
        - Each waiting passenger needs a 'board' action (cost 1).
        - Each unserved passenger (waiting or boarded) needs a 'depart' action (cost 1).
        - Total base cost = (number of waiting passengers) + (number of unserved passengers).
    5.  Identify the set of required floor indices:
        - Origin floor indices for all waiting passengers.
        - Destination floor indices for all boarded passengers.
    6.  Calculate the number of required stops: This is the number of unique floor indices in the set from step 5.
    7.  Calculate the estimated lift movement cost:
        - If there are no required stops, movement cost is 0.
        - Otherwise, consider the set of required floor indices plus the current lift's floor index. The movement cost is the difference between the maximum and minimum index in this combined set (the span). This estimates the minimum vertical travel distance required to visit all necessary floors.
    8.  The total heuristic value is the sum of the base action cost, the number of required stops, and the estimated movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor mapping, passenger destinations,
        and the set of all passengers.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Map floor names (f1, f2, ...) to integer indices (1, 2, ...).
        # Assumes floor names are 'f' followed by a number.
        self.floor_to_idx = {}
        all_floors = set()

        # Collect all potential floor objects from static facts and initial state
        # Iterate through all facts and collect arguments that look like floors
        for fact in static_facts | initial_state:
             parts = get_parts(fact)
             if parts:
                 # Check all arguments except the predicate name
                 for part in parts[1:]:
                     if part.startswith('f'):
                         all_floors.add(part)

        # Sort floors based on the number in their name and create the mapping
        # This relies on floor names being consistently f<number>
        try:
            sorted_floors = sorted(list(all_floors), key=lambda f: int(f[1:]))
            for i, floor in enumerate(sorted_floors):
                self.floor_to_idx[floor] = i + 1 # Use 1-based indexing
        except ValueError as e:
             # If floor names are not in the expected format, the heuristic cannot be built
             print(f"Error parsing floor names: {e}. Expected 'f<number>' format.")
             # Raising an error is appropriate if the format is strictly expected.
             raise ValueError(f"Unexpected floor name format encountered: {e}") from e


        # Store goal locations for each passenger.
        self.destin_map = {}
        # Destinations are defined by (destin ?p ?f) facts, typically static.
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, destination_floor = get_parts(fact)
                self.destin_map[passenger] = destination_floor

        # Collect all passenger names mentioned in static facts or initial state
        self.all_passengers = set()
        for fact in static_facts | initial_state:
             parts = get_parts(fact)
             if parts:
                 predicate = parts[0]
                 # Passengers appear in origin, destin, boarded, served predicates
                 if predicate in ["origin", "destin", "boarded", "served"]:
                     if len(parts) > 1 and parts[1].startswith('p'): # Assuming passenger names start with 'p'
                         self.all_passengers.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        current_floor = None
        waiting_passengers = set() # Passengers at origin, not boarded
        boarded_passengers = set() # Passengers inside lift
        served_passengers = set() # Passengers who reached destination
        origin_map = {} # Map passenger to their origin floor if waiting

        # Extract current state information
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "lift-at":
                current_floor = parts[1]
            elif predicate == "origin":
                passenger, floor = parts[1:]
                waiting_passengers.add(passenger)
                origin_map[passenger] = floor
            elif predicate == "boarded":
                passenger = parts[1]
                boarded_passengers.add(passenger)
            elif predicate == "served":
                passenger = parts[1]
                served_passengers.add(passenger)

        # Identify all unserved passengers
        unserved_passengers = self.all_passengers - served_passengers

        # If no unserved passengers, goal is reached
        if not unserved_passengers:
            return 0

        # 4. Calculate base action cost
        # Each waiting passenger needs 1 board action
        # Each unserved passenger needs 1 depart action
        # Note: In a consistent state, unserved_passengers = waiting_passengers | boarded_passengers
        base_action_cost = len(waiting_passengers) + len(unserved_passengers)

        # 5. Identify required floor indices
        required_indices = set()

        # Origin floors for waiting passengers
        for p in waiting_passengers:
             # origin_map should contain p if state is consistent
             if p in origin_map and origin_map[p] in self.floor_to_idx:
                 required_indices.add(self.floor_to_idx[origin_map[p]])

        # Destination floors for boarded passengers
        for p in boarded_passengers:
            if p in self.destin_map and self.destin_map[p] in self.floor_to_idx:
                required_indices.add(self.floor_to_idx[self.destin_map[p]])

        # 6. Calculate number of required stops
        num_stops = len(required_indices)

        # 7. Calculate estimated lift movement cost
        movement_cost = 0
        # Movement is needed if there are floors to visit
        if required_indices:
            current_idx = self.floor_to_idx.get(current_floor)
            # current_floor should always be a valid floor in a reachable state
            if current_idx is not None:
                 all_relevant_indices = required_indices | {current_idx}
                 min_idx = min(all_relevant_indices)
                 max_idx = max(all_relevant_indices)
                 movement_cost = max_idx - min_idx
            # else: Should not happen in valid problem instances/states

        # 8. Total heuristic value
        total_cost = base_action_cost + num_stops + movement_cost

        return total_cost
