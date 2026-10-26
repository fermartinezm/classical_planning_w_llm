import re
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential whitespace issues and ensure correct splitting
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the remaining cost by summing:
    1. The number of 'board' actions needed (one for each passenger waiting at their origin).
    2. The number of 'depart' actions needed (one for each passenger currently boarded).
    3. An estimate of the floor movement cost, calculated as the minimum travel distance
       to visit all floors where passengers are waiting or need to be dropped off,
       starting from the lift's current floor.

    # Assumptions
    - Floors are ordered linearly, and their names (e.g., f1, f2, ...) allow for
      easy mapping to integers representing their level. We assume f1 is the lowest,
      f2 is above f1, etc., and can parse the number from the floor name.
    - The goal is to serve all passengers.
    - 'destin' facts are static.
    - 'above' facts define the floor order and are static.

    # Heuristic Initialization
    - Extract the destination floor for each passenger from the static facts.
    - Build a mapping from floor names to integer levels based on the 'above' facts.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Identify the current floor of the lift.
    2. Identify all passengers who are waiting at their origin (`(origin ?p ?f)`).
       - Each such passenger needs a 'board' action (cost 1) and a 'depart' action (cost 1). Add 2 to the heuristic.
       - Add their origin floor to the set of floors the lift must stop at.
    3. Identify all passengers who are currently boarded (`(boarded ?p)`).
       - Each such passenger needs a 'depart' action (cost 1). Add 1 to the heuristic.
       - Find their destination floor using the pre-computed goal locations.
       - Add their destination floor to the set of floors the lift must stop at.
    4. Determine the set of unique floors the lift must stop at (origin floors of waiting passengers + destination floors of boarded passengers).
    5. If this set of floors is empty, all relevant passengers must be served, so the heuristic is 0.
    6. If the set is not empty, calculate the minimum and maximum floor levels among these required stops.
    7. Calculate the integer level of the lift's current floor.
    8. Estimate the movement cost: This is the distance from the current floor to the *nearest* extreme required floor (min or max), plus the total distance spanning all required floors (max - min). This models the lift going to one end of the required range and then sweeping across to the other end.
       Movement Cost = min(|current_floor_level - min_stop_level|, |current_floor_level - max_stop_level|) + (max_stop_level - min_stop_level)
    9. Add the estimated movement cost to the heuristic.
    10. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each passenger.
        - A mapping from floor names to integer levels.
        """
        self.goals = task.goals  # Goal conditions (used implicitly, goal is all served)
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each passenger from static 'destin' facts.
        self.goal_locations = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                passenger, location = get_parts(fact)[1:]
                self.goal_locations[passenger] = location

        # Build floor name to integer level mapping.
        # We assume floor names are f<number> and sort them numerically.
        floor_names = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "above":
                floor_names.add(parts[1])
                floor_names.add(parts[2])

        # Sort floor names based on the number part (e.g., f1, f2, ..., f10, f11)
        # This assumes a consistent naming convention.
        def floor_sort_key(floor_name):
            match_obj = re.match(r'f(\d+)', floor_name)
            if match_obj:
                return int(match_obj.group(1))
            return floor_name # Fallback if naming is inconsistent

        sorted_floor_names = sorted(list(floor_names), key=floor_sort_key)

        self.floor_to_int = {name: i + 1 for i, name in enumerate(sorted_floor_names)}
        # We also need the reverse mapping for debugging or if needed, but not strictly for this heuristic
        # self.int_to_floor = {i + 1: name for i, name in enumerate(sorted_floor_names)}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        h = 0  # Initialize heuristic cost.
        current_lift_floor_name = None
        floors_to_stop_names = set()

        # Identify current lift location and passengers needing service
        waiting_passengers = set()
        boarded_passengers = set()

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "lift-at":
                current_lift_floor_name = parts[1]
            elif predicate == "origin":
                passenger, floor = parts[1:]
                waiting_passengers.add(passenger)
                floors_to_stop_names.add(floor)
            elif predicate == "boarded":
                passenger = parts[1]
                boarded_passengers.add(passenger)

        # Add cost for board and depart actions
        # Each waiting passenger needs 1 board + 1 depart = 2 actions
        h += len(waiting_passengers) * 2
        # Each boarded passenger needs 1 depart action
        h += len(boarded_passengers) * 1

        # Add destination floors for boarded passengers to stops
        for passenger in boarded_passengers:
             # Check if the passenger is not yet served (should be true if they are boarded)
             if f'(served {passenger})' not in state:
                 # Find their destination floor using the pre-computed mapping
                 if passenger in self.goal_locations:
                     dest_floor = self.goal_locations[passenger]
                     floors_to_stop_names.add(dest_floor)
                 # else: This case implies a problem definition issue where a boarded passenger has no destination.

        # If no floors need stops, all relevant passengers must be served.
        if not floors_to_stop_names:
            # Double check if all passengers are actually served.
            # This check is technically redundant if the state is valid and goal is all served.
            # But as a safety check:
            all_served = True
            for goal in self.goals:
                 if goal not in state:
                      all_served = False
                      break
            if all_served:
                return 0
            else:
                 # This scenario shouldn't happen in a well-formed problem where
                 # unserved passengers are either waiting or boarded.
                 # If it does, it means there's an unserved passenger not in
                 # (origin) or (boarded) state, which is impossible by domain definition.
                 # Return a large value or handle as an error state?
                 # For this heuristic, if floors_to_stop_names is empty, we assume goal is met.
                 return 0 # Assuming problem is well-formed

        # Calculate movement cost
        if current_lift_floor_name is None:
             # This state is invalid according to the domain (lift must be somewhere)
             # Return a large value to discourage this state, or 0 if it implies goal?
             # A lift must be at a floor, so this indicates a malformed state representation.
             # Let's return a large value.
             return float('inf') # Or a large constant

        try:
            current_floor_int = self.floor_to_int[current_lift_floor_name]
            floors_to_stop_ints = [self.floor_to_int[f] for f in floors_to_stop_names]

            min_stop_int = min(floors_to_stop_ints)
            max_stop_int = max(floors_to_stop_ints)

            # Movement cost estimate: Go to the nearest extreme floor, then sweep across the range.
            move_cost = min(abs(min_stop_int - current_floor_int), abs(max_stop_int - current_floor_int)) + (max_stop_int - min_stop_int)

            h += move_cost

        except KeyError as e:
            # This means a floor name in the state or goals wasn't found in the static 'above' facts.
            # Indicates a problem definition issue. Return infinity.
            print(f"Error: Floor name {e} not found in floor mapping.")
            return float('inf')


        return h

