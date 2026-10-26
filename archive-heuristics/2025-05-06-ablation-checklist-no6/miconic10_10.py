from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions for PDDL fact parsing
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
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

    Estimates the cost by summing:
    1. Minimum elevator movement to visit all floors with waiting passengers
       or boarded passengers' destinations.
    2. Number of board actions needed (passengers waiting at origin).
    3. Number of depart actions needed (passengers boarded).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal conditions (to identify all passengers).
        - Static facts (`destin` and `above`).
        - Floor mapping (name to index).
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Extract all passengers from goals
        self.all_passengers = set()
        for goal in self.goals:
             # Goal is typically (served ?p)
             # Assuming task.goals is a frozenset of fact strings
             parts = get_parts(goal)
             if parts and parts[0] == "served" and len(parts) == 2:
                 self.all_passengers.add(parts[1])


        # Extract passenger destinations from static facts
        self.passenger_destinations = {}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "destin" and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                self.passenger_destinations[passenger] = floor

        # Extract floors and create floor name to index mapping
        floor_names = set()
        # Get floors from static facts
        for fact in self.static_facts:
             parts = get_parts(fact)
             if parts and parts[0] == "above" and len(parts) == 3:
                 floor_names.update(parts[1:])
             elif parts and parts[0] == "destin" and len(parts) == 3:
                 floor_names.add(parts[2]) # destin ?p ?f
        # Get floors from initial state facts (available in task object)
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "lift-at" and len(parts) == 2:
                 floor_names.add(parts[1])
             elif parts and parts[0] == "origin" and len(parts) == 3:
                 floor_names.add(parts[2])

        # Sort floors numerically based on the number in the name (e.g., f1 < f2)
        # This assumes floor names are consistently formatted like 'f<number>'
        # Handle potential errors if floor names are not in f<number> format
        try:
            sorted_floor_names = sorted(list(floor_names), key=lambda f: int(f[1:]))
            # Use 1-based indexing for floors if names are f1, f2, ...
            self.floor_to_index = {f: int(f[1:]) for f in sorted_floor_names}
        except ValueError:
             # Fallback if floor names are not f<number> format
             # This is less ideal as it doesn't guarantee correct floor order
             # Based on examples, f<number> seems standard.
             print("Warning: Floor names not in 'f<number>' format. Using alphabetical sort.")
             sorted_floor_names = sorted(list(floor_names))
             # Use 0-based indexing for alphabetical sort
             self.floor_to_index = {f: i for i, f in enumerate(sorted_floor_names)}


    def __call__(self, node):
        """Compute the domain-dependent heuristic value for the given state."""
        state = node.state

        # Check if goal is reached
        # The Task.goal_reached method is more robust as it handles 'and' goals
        # However, the heuristic signature only provides the node.
        # We can check if all individual goal facts are in the state.
        if self.goals <= state:
            return 0

        # Find elevator's current floor
        elevator_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                elevator_floor = get_parts(fact)[1]
                break
        # If elevator location is unknown, heuristic is infinite or a large value
        # Assuming valid states always have lift-at
        if elevator_floor is None:
             # This should not happen in a valid state reachable from initial state
             # but as a safeguard, return a large value or infinity
             return float('inf')

        # Handle case where elevator_floor might not be in the floor_to_index map
        # (e.g., if the initial state had a floor not mentioned in static/goals)
        # This case is handled by including initial state floors in __init__
        # But if a state fact has a floor not seen before, it's an issue.
        # Assuming all floors are properly initialized.
        if elevator_floor not in self.floor_to_index:
             # Should not happen with robust __init__
             return float('inf') # Or handle as error

        elevator_floor_idx = self.floor_to_index[elevator_floor]

        # Identify unserved passengers who are not yet at their destination
        # We only care about passengers who are waiting or boarded.
        # Passengers already served are not in the goal, so they don't contribute.
        # Passengers not in state (neither origin nor boarded) are assumed served.
        # Let's identify unserved passengers by checking against the goal set.
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_passengers_set = self.all_passengers - served_passengers

        # Identify waiting and boarded unserved passengers
        waiting_passengers = {} # {passenger: origin_floor}
        boarded_passengers = set() # {passenger}

        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == "origin" and len(parts) == 3:
                  p, f = parts[1], parts[2]
                  if p in unserved_passengers_set:
                       waiting_passengers[p] = f
             elif parts and parts[0] == "boarded" and len(parts) == 2:
                  p = parts[1]
                  if p in unserved_passengers_set:
                       boarded_passengers.add(p)

        # Identify required stops (floors the elevator must visit) for unserved passengers
        pickup_floors = {floor for floor in waiting_passengers.values()}
        # Dropoff floors are destinations of boarded unserved passengers
        dropoff_floors = set()
        for p in boarded_passengers:
             if p in self.passenger_destinations:
                  dropoff_floors.add(self.passenger_destinations[p])
             # else: This boarded passenger doesn't have a destination? Problematic state.
             # Assume all passengers in self.all_passengers have a destination in static.

        required_stops = sorted(list(pickup_floors | dropoff_floors), key=lambda f: self.floor_to_index.get(f, float('inf'))) # Handle floors not found

        # Calculate movement cost
        movement_cost = 0
        if required_stops:
            # Filter out required stops that weren't found in floor_to_index (shouldn't happen with robust init)
            valid_required_stops = [f for f in required_stops if f in self.floor_to_index]
            if valid_required_stops:
                required_stop_indices = [self.floor_to_index[f] for f in valid_required_stops]
                min_idx = required_stop_indices[0]
                max_idx = required_stop_indices[-1]

                # Minimum travel to visit all floors in the range [min_idx, max_idx] starting from elevator_floor_idx
                # Go from current to one end, then sweep to the other end.
                dist_to_min = abs(elevator_floor_idx - min_idx)
                dist_to_max = abs(elevator_floor_idx - max_idx)

                # Option 1: Go to min_idx, then sweep up to max_idx
                cost1 = dist_to_min + (max_idx - min_idx)
                # Option 2: Go to max_idx, then sweep down to min_idx
                cost2 = dist_to_max + (max_idx - min_idx)

                movement_cost = min(cost1, cost2)
            # else: required_stops was not empty, but none of the floors were indexed.
            # This implies a problem with floor extraction or state validity.
            # The movement_cost remains 0, but action_cost will be > 0 if there are waiting/boarded passengers.
            # This seems acceptable as a fallback.


        # Calculate action cost
        # Each waiting passenger needs 1 board action
        # Each boarded passenger needs 1 depart action
        action_cost = len(waiting_passengers) + len(boarded_passengers)

        return movement_cost + action_cost
