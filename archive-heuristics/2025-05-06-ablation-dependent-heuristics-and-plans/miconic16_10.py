# Need fnmatch for pattern matching in facts
from fnmatch import fnmatch

# Assuming Heuristic base class is available if needed for the environment
# from heuristics.heuristic_base import Heuristic

# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of non-wildcard args,
    # or if args contains wildcards, just check prefix match up to min length.
    # The current implementation using zip and all() checks prefix match.
    # This is sufficient for the patterns used in this heuristic.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Define the heuristic class. Inherit from Heuristic if required by the environment.
# class miconicHeuristic(Heuristic):
class miconicHeuristic:
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    passengers. It sums the number of boarding/departing actions needed
    for unserved passengers and adds an estimate of the vertical movement
    cost for the lift to visit all necessary floors.

    # Assumptions
    - Floors are ordered numerically based on their names (e.g., f1 < f2 < f10).
    - The lift can carry multiple passengers simultaneously.
    - The movement cost is estimated by the total vertical span the lift
      must cover from its current position to encompass all required stops.

    # Heuristic Initialization
    - Extract the destination floor for each passenger from the static facts.
    - Build a mapping from floor names (e.g., 'f1', 'f2') to integer levels
      based on the numerical suffix in their names.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost to 0.
    2. Find the current floor of the lift.
    3. Identify all passengers who are not yet served (i.e., `(served ?p)` is not in the state).
    4. Initialize a set `required_floors` to store floors the lift must visit.
    5. For each unserved passenger:
       - If the passenger is waiting at an origin floor (`(origin ?p ?f)` is true):
         - Add 1 to the total cost (for the `board` action).
         - Add the origin floor `?f` to `required_floors`.
       - If the passenger is boarded (`(boarded ?p)` is true):
         - Add 1 to the total cost (for the `depart` action).
         - Find the passenger's destination floor `?d` (using the pre-calculated destination map).
         - Add the destination floor `?d` to `required_floors`.
    6. Calculate the movement cost:
       - If `required_floors` is empty, the movement cost is 0.
       - If `required_floors` is not empty:
         - Get the integer level for the current lift floor.
         - Get the integer levels for all floors in `required_floors`.
         - Find the minimum and maximum integer levels among the `required_floors`.
         - The movement cost is the difference between the maximum of (current floor level, max required level) and the minimum of (current floor level, min required level). This represents the total vertical span the lift must cover.
    7. Add the calculated movement cost to the total heuristic cost.
    8. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Destination floor for each passenger.
        - Mapping from floor names to integer levels.
        """
        # self.goals = task.goals # Not strictly needed for this heuristic logic
        self.static = task.static

        # Extract destination for each passenger from static facts
        self.destin_map = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destin_map[passenger] = floor

        # Build floor name to integer level mapping
        # Find all floor names from static facts (e.g., above predicates)
        floor_names = set()
        for fact in self.static:
            if match(fact, "above", "*", "*"):
                _, f1, f2 = get_parts(fact)
                floor_names.add(f1)
                floor_names.add(f2)
        # Also consider floors mentioned in destin facts, just in case
        floor_names.update(self.destin_map.values())


        # Sort floor names based on the integer suffix
        # Assumes floor names are like 'f1', 'f10', 'f2'
        def get_floor_level(floor_name):
            # Handle potential errors if floor name format is unexpected
            try:
                # Extract integer after 'f'. This handles f1, f10, f2 etc.
                return int(floor_name[1:])
            except (ValueError, IndexError):
                # This should not happen with standard miconic benchmarks.
                # Assign a default level or handle error if necessary.
                # For robustness, return 0 or a value that won't disrupt sorting
                # significantly if non-standard names appear, though it might
                # affect heuristic quality. Assuming standard fN format.
                # print(f"Warning: Unexpected floor name format: {floor_name}") # Avoid printing in heuristic
                return 0 # Default level for unexpected format

        self.ordered_floors = sorted(list(floor_names), key=get_floor_level)
        # Map floor name to its 1-based index in the sorted list
        self.floor_to_int = {floor: i + 1 for i, floor in enumerate(self.ordered_floors)}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        action_cost = 0
        required_floors = set()
        current_lift_floor = None

        # Find current lift location
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break

        # Identify unserved passengers and required stops/actions
        # Collect all passengers mentioned in destin facts (static) or state facts (origin, boarded, served)
        all_passengers = set(self.destin_map.keys())
        passengers_in_state = set()
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] in ["origin", "boarded", "served"]:
                 if len(parts) > 1: # Ensure there's a passenger object
                    passengers_in_state.add(parts[1])

        passengers_to_check = all_passengers.union(passengers_in_state)

        for passenger in passengers_to_check:
            # Check if passenger is served
            if f"(served {passenger})" not in state:
                # Passenger is not served, calculate cost/stops
                is_waiting = False
                origin_floor = None

                # Check if waiting at origin
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                        is_waiting = True
                        _, _, origin_floor = get_parts(fact)
                        break

                # Check if boarded
                is_boarded = f"(boarded {passenger})" in state

                if is_waiting:
                    action_cost += 1 # Cost for board action
                    if origin_floor: # Should always be present if is_waiting is True
                        required_floors.add(origin_floor)
                elif is_boarded:
                    action_cost += 1 # Cost for depart action
                    destin_floor = self.destin_map.get(passenger)
                    if destin_floor: # Should always exist for unserved passengers
                         required_floors.add(destin_floor)
                    # else: Invalid state or problem definition

                # Note: A passenger should be either waiting, boarded, or served.
                # If not served, they must be waiting or boarded in a valid state.
                # The logic covers these two cases.

        # Calculate movement cost
        movement_cost = 0
        # Only calculate movement cost if there are floors to visit AND the lift location is known
        if required_floors and current_lift_floor in self.floor_to_int:
            current_f_int = self.floor_to_int[current_lift_floor]

            # Filter required_floors to only include known floors
            valid_required_floors = {f for f in required_floors if f in self.floor_to_int}

            if valid_required_floors:
                required_ints = [self.floor_to_int[f] for f in valid_required_floors]
                min_f_int = min(required_ints)
                max_f_int = max(required_ints)

                # Movement cost is the span from current floor to the min/max required floor
                # This is max(current, max_req) - min(current, min_req)
                movement_cost = max(current_f_int, max_f_int) - min(current_f_int, min_f_int)
            # else: required_floors were all unknown, movement_cost remains 0.
        # else: required_floors is empty or current_lift_floor is unknown, movement_cost remains 0.


        return action_cost + movement_cost
