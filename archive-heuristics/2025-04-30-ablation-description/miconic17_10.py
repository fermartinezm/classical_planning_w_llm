from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the remaining effort to serve all passengers.
    It considers the number of unserved passengers and the lift's position
    relative to the floors that need visiting (origin floors for waiting
    passengers and destination floors for boarded passengers).

    # Assumptions
    - Floors are linearly ordered, defined by the 'above' predicate.
    - The cost of moving between adjacent floors is 1.
    - The cost of boarding or departing is 1.
    - The heuristic sums the number of board/depart actions needed for unserved
      passengers and the minimum distance from the current lift location to any
      required floor.

    # Heuristic Initialization
    - Parses the 'above' predicates from static facts to build a mapping
      from floor names to their numerical levels. This allows calculating
      the distance between any two floors. Assumes a linear floor structure.
    - Stores the destination floor for each passenger by looking up 'destin'
      facts in the initial state and static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is the goal state. If yes, heuristic is 0.
    2. Identify the current floor of the lift.
    3. Identify all passengers that are not yet served.
    4. For each unserved passenger, determine if they need to be boarded (are at origin)
       or departed (are boarded).
    5. Count the total number of board actions needed (for unserved passengers at origin).
    6. Count the total number of depart actions needed (for unserved boarded passengers).
    7. Collect the set of all unique floors that need to be visited: origin floors
       for unserved passengers at origin, and destination floors for unserved
       boarded passengers.
    8. Calculate the minimum floor distance from the current lift location to any
       floor in the set of required floors. If the set is empty or the lift location
       is unknown, the minimum distance is 0.
    9. The heuristic value is the sum of the number of board actions needed, the
       number of depart actions needed, and the minimum floor distance calculated
       in step 8.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Need initial state to find destin facts

        # Build floor level mapping from 'above' facts.
        # (above f_lower f_higher) means f_higher is directly above f_lower.
        # We want to assign levels such that level(f_higher) = level(f_lower) + 1.

        floors_directly_above = {} # Map: f_lower -> f_higher
        floors_directly_below = {} # Map: f_higher -> f_lower
        all_floors_mentioned_in_above = set()

        for fact in self.static:
            if match(fact, "above", "*", "*"):
                f_lower, f_higher = get_parts(fact)[1:]
                floors_directly_above[f_lower] = f_higher
                floors_directly_below[f_higher] = f_lower
                all_floors_mentioned_in_above.add(f_lower)
                all_floors_mentioned_in_above.add(f_higher)

        self.floor_levels = {}
        lowest_floors = []

        # Find floors that are not the higher floor in any 'above' fact (i.e., have no floor below them defined by 'above')
        for f in all_floors_mentioned_in_above:
            if f not in floors_directly_below:
                lowest_floors.append(f)

        # Collect all potential floors from init/goals/static, even if not in 'above' chain
        all_potential_floors = set()
        for fact in task.initial_state | task.goals | task.static:
            parts = get_parts(fact)
            if parts: # Ensure parts is not empty
                predicate = parts[0]
                if predicate in ['lift-at', 'origin', 'destin']:
                    if len(parts) > 1: # Ensure there's an argument
                        all_potential_floors.add(parts[-1]) # Last argument is the floor
                elif predicate == 'above':
                    if len(parts) > 2: # Ensure there are two floor arguments
                        all_potential_floors.update(parts[1:]) # Both arguments are floors

        # If no 'above' facts define an order, assign level 0 to all potential floors.
        if not floors_directly_above and all_potential_floors:
             for f in all_potential_floors:
                 self.floor_levels[f] = 0
        else:
            # Assign levels using BFS starting from identified lowest floors
            queue = [(f, 1) for f in lowest_floors]
            visited = set(lowest_floors)

            while queue:
                current_floor, level = queue.pop(0)
                self.floor_levels[current_floor] = level

                # Find the floor directly above the current floor
                next_floor = floors_directly_above.get(current_floor)
                if next_floor and next_floor not in visited:
                     visited.add(next_floor)
                     queue.append((next_floor, level + 1))

            # Assign level 0 to any potential floor not reached by BFS (e.g., isolated floors)
            for f in all_potential_floors:
                 if f not in self.floor_levels:
                     self.floor_levels[f] = 0


        # Store destination floor for each passenger from the goal state.
        self.passenger_destinations = {}
        # Get all passengers mentioned in init/goals/static
        all_passengers = set()
        for fact in task.initial_state | task.goals | task.static:
             parts = get_parts(fact)
             if parts:
                 predicate = parts[0]
                 if predicate in ['origin', 'destin', 'served', 'boarded']:
                     if len(parts) > 1: # Ensure there's an argument for the passenger
                        all_passengers.add(parts[1])

        # Find destination for each passenger
        for passenger_name in all_passengers:
             for fact in task.initial_state | self.static: # Destin facts are usually static or in initial state
                 if match(fact, "destin", passenger_name, "*"):
                     self.passenger_destinations[passenger_name] = get_parts(fact)[2]
                     break


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Check if goal is reached
        if self.goals <= state:
            return 0

        # 2. Identify the current floor of the lift.
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # If lift location is unknown or not in our floor levels map,
        # we can't calculate floor distance. Fallback to just counting actions.
        lift_location_known = current_lift_floor is not None and current_lift_floor in self.floor_levels

        # 3. Identify unserved passengers.
        all_passengers_in_state = set()
        for fact in state:
             parts = get_parts(fact)
             if parts:
                 predicate = parts[0]
                 if predicate in ['origin', 'destin', 'boarded', 'served']:
                     if len(parts) > 1:
                        all_passengers_in_state.add(parts[1])

        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_passengers = all_passengers_in_state - served_passengers

        # 4, 5, 6, 7. Count actions needed and identify required floors.
        num_board_needed = 0
        num_depart_needed = 0
        current_required_floors = set()

        for passenger in unserved_passengers:
            is_boarded = False
            # Check if passenger is boarded
            for fact in state:
                if match(fact, "boarded", passenger):
                    is_boarded = True
                    break

            if is_boarded:
                # Passenger is boarded, needs to go to destination
                num_depart_needed += 1
                destin_floor = self.passenger_destinations.get(passenger)
                if destin_floor:
                    current_required_floors.add(destin_floor)
            else:
                # Passenger is not boarded, must be at origin
                num_board_needed += 1
                # Find their origin floor in the current state
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                        origin_floor = get_parts(fact)[2]
                        current_required_floors.add(origin_floor)
                        break
                # If origin_floor is not found, the passenger is in an unexpected state.
                # We added 1 for board_needed but cannot add a required floor.

        # 8. Calculate minimum floor distance.
        min_floor_distance = 0
        if lift_location_known and current_required_floors:
            current_level = self.floor_levels[current_lift_floor]
            distances = []
            for floor in current_required_floors:
                if floor in self.floor_levels: # Only consider required floors we have levels for
                    distances.append(abs(self.floor_levels[floor] - current_level))
            if distances: # Ensure distances list is not empty
                min_floor_distance = min(distances)
            # If current_required_floors is not empty but none of the floors are in self.floor_levels,
            # min_floor_distance remains 0. This might happen if floor_levels parsing failed for some floors.

        # 9. Heuristic value = actions needed + lift movement cost
        heuristic_value = num_board_needed + num_depart_needed + min_floor_distance

        return heuristic_value
