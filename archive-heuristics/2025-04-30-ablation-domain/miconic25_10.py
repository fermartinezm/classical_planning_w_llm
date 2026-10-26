from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern.
    # This assumes fixed-arity predicates in the pattern.
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    The heuristic estimates the number of actions needed to serve all passengers.
    It counts the required 'board' and 'disembark' actions for unserved passengers
    and adds an estimate of the lift's travel distance to visit all necessary floors
    (origins of waiting passengers and destinations of boarded passengers).

    # Assumptions
    - Passengers are initially at their origin floor and need to be transported
      to their destination floor.
    - A passenger is served when they disembark at their destination.
    - The lift moves one floor at a time (up or down).
    - The cost of board, disembark, and move actions is 1.
    - The floor structure is linear, defined by 'above' predicates.

    # Heuristic Initialization
    - Parses the 'above' facts from static information to determine the floor order
      and assign numerical levels to each floor. Assumes 'above f1 f2' means f1 is
      one level higher than f2.
    - Extracts the destination floor for each passenger from static information.
    - Identifies all passengers mentioned in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current floor of the lift from the state. If the lift location
       cannot be determined, return infinity (or a large value).
    2. Convert the current lift floor name to its numerical level using the
       pre-calculated floor levels mapping.
    3. Initialize the heuristic cost `h` to 0.
    4. Initialize two sets: `pickup_floors_needed` and `dropoff_floors_needed`.
    5. Iterate through all passengers identified during initialization:
       - Check if the passenger is already 'served' in the current state. If yes,
         this passenger contributes 0 to the heuristic, so continue to the next passenger.
       - If the passenger is not served, check their current status:
         - If the passenger is at their 'origin' floor (predicate `(origin p o)` is true),
           add their origin floor `o` to `pickup_floors_needed`. Add 1 to `h` for the
           required 'board' action.
         - If the passenger is 'boarded' (predicate `(boarded p)` is true), add their
           destination floor `d` (looked up from initialization) to `dropoff_floors_needed`.
           Add 1 to `h` for the required 'disembark' action.
         - (Implicit assumption: An unserved passenger is either at their origin or boarded).
    6. Combine the required pickup and dropoff floors into a single set `required_floors`.
    7. If `required_floors` is empty, it means all unserved passengers are already
       boarded and their destination is the current floor (disembark cost already added),
       or there are no unserved passengers (goal state, handled at the start).
       In this case, the estimated travel cost is 0.
    8. If `required_floors` is not empty:
       - Get the numerical levels for all floors in `required_floors`.
       - Find the minimum (`min_req_level`) and maximum (`max_req_level`) levels
         among the required floors.
       - Estimate the travel cost as the minimum number of floor moves required
         for the lift to travel from its current level to visit all required levels.
         A reasonable estimate is the distance from the current level to the closest
         extreme required level (`min_req_level` or `max_req_level`), plus the
         distance to traverse the entire range between the minimum and maximum
         required levels (`max_req_level - min_req_level`).
       - Estimated travel cost = min(abs(current_lift_level - min_req_level),
                                     abs(current_lift_level - max_req_level)) +
                                (max_req_level - min_req_level).
    9. The total heuristic value is `h` (board/disembark costs) plus the estimated
       travel cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Parse floor levels from 'above' facts
        self.floor_levels = {}
        floors = set()
        above_relations = []
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f_high, f_low = get_parts(fact)[1:]
                above_relations.append((f_high, f_low))
                floors.add(f_high)
                floors.add(f_low)

        # Build a map from a floor to the floor immediately below it
        below_map = {f_high: f_low for f_high, f_low in above_relations}

        # Find the highest floor (a floor f for which there is no (above f ?other) fact)
        # In a linear structure, this is the floor that appears as f_high but never as f_low
        floors_that_are_above_others = set(below_map.keys())
        floors_that_are_below_others = set(below_map.values())
        highest_floors = list(floors_that_are_above_others - floors_that_are_below_others)

        if not floors:
             # No floors defined, likely an empty or malformed problem
             pass # Heuristic will return 0 for goal state, inf otherwise if lift-at is missing
        elif len(floors) == 1:
             # Single floor case
             self.floor_levels[list(floors)[0]] = 1
        elif highest_floors:
            # Assuming a single highest floor in a linear structure
            highest_floor = highest_floors[0]
            current_level = len(floors)
            self.floor_levels[highest_floor] = current_level
            current_floor = highest_floor

            # Iterate downwards to assign levels
            while current_floor in below_map:
                current_floor = below_map[current_floor]
                current_level -= 1
                self.floor_levels[current_floor] = current_level
        else:
             # Fallback for unexpected floor structure (e.g., disconnected, cycles)
             # Assign levels based on sorted names as a last resort
             sorted_floors = sorted(list(floors))
             self.floor_levels = {f: i+1 for i, f in enumerate(sorted_floors)}


        # Extract passenger destinations and identify all passengers
        self.passenger_destinations = {}
        all_passengers = set()
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                p, d = get_parts(fact)[1:]
                self.passenger_destinations[p] = d
                all_passengers.add(p)
            if match(fact, "origin", "*", "*"):
                 p, o = get_parts(fact)[1:]
                 all_passengers.add(p)

        # Also add passengers from initial state origins, just in case
        for fact in task.initial_state:
             if match(fact, "origin", "*", "*"):
                  p, o = get_parts(fact)[1:]
                  all_passengers.add(p)

        self.all_passengers = list(all_passengers)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # If lift location is unknown or floor not in our mapping, heuristic is infinite
        if current_lift_floor is None or current_lift_floor not in self.floor_levels:
             return float('inf')

        current_lift_level = self.floor_levels[current_lift_floor]

        cost = 0
        pickup_floors_needed = set()
        dropoff_floors_needed = set()

        # Identify unserved passengers and their needs
        for passenger in self.all_passengers:
             # If passenger is served, they don't contribute to the heuristic
             if f"(served {passenger})" in state:
                  continue

            # Check if passenger is at origin or boarded
             is_at_origin = False
             origin_floor = None
             for fact in state:
                  if match(fact, "origin", passenger, "*"):
                       is_at_origin = True
                       origin_floor = get_parts(fact)[2]
                       break

             is_boarded = f"(boarded {passenger})" in state

             # Determine required actions and floors
             if is_at_origin:
                 # Passenger is waiting at origin, needs boarding
                 if origin_floor: # Ensure origin floor was found
                     pickup_floors_needed.add(origin_floor)
                     cost += 1 # Cost for 'board' action
             elif is_boarded:
                 # Passenger is boarded, needs disembarking at destination
                 destination_floor = self.passenger_destinations.get(passenger)
                 if destination_floor: # Ensure destination floor exists
                     dropoff_floors_needed.add(destination_floor)
                     cost += 1 # Cost for 'disembark' action
             # Note: Passengers neither at origin nor boarded are not accounted for
             # by this heuristic, assuming they don't exist in valid unserved states.


        # Calculate travel cost
        required_floors = pickup_floors_needed.union(dropoff_floors_needed)

        if not required_floors:
            # If no required stops, travel cost is 0.
            # This covers cases where all unserved passengers are boarded
            # and their destination is the current floor (disembark cost added),
            # or no unserved passengers exist (goal state, handled above).
            travel_cost = 0
        else:
            # Get levels for required floors, ensuring they are in our mapping
            required_levels = {self.floor_levels[f] for f in required_floors if f in self.floor_levels}

            if not required_levels:
                 # Fallback if required floors somehow didn't map to levels
                 travel_cost = 0
            else:
                min_req_level = min(required_levels)
                max_req_level = max(required_levels)

                # Estimate travel distance to visit all required floors
                # This is the distance to reach the closest extreme required level
                # plus the distance to traverse the entire range of required levels.
                travel_cost = min(abs(current_lift_level - min_req_level), abs(current_lift_level - max_req_level)) + (max_req_level - min_req_level)

        total_cost = cost + travel_cost

        return total_cost
