import re
from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Assume Heuristic base class is available and imported elsewhere
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the cost to serve all passengers by summing:
    1. The number of 'board' and 'depart' actions needed for unserved passengers.
    2. The estimated lift movement cost to visit necessary floors.

    # Heuristic Initialization
    - Extracts floor ordering and maps floor names to levels based on numerical suffix.
    - Extracts origin and destination floors for each passenger from static facts and initial state.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift.
    2. Identify all passengers who are not yet served.
    3. For each unserved passenger:
       - If waiting at origin: needs 'board' (1 action) and 'depart' (1 action). Add 2 to passenger action cost. Add origin floor and destination floor to required floors.
       - If boarded: needs 'depart' (1 action). Add 1 to passenger action cost. Add destination floor to required floors.
    4. Calculate total passenger action cost (sum from step 3).
    5. Calculate lift movement cost:
       - Find the minimum and maximum floor levels among the required floors.
       - If there are required floors, estimate movement as the distance from the current lift floor to the closer of the min/max required floors, plus the distance between the min and max required floors. This estimates the travel needed to reach and traverse the range of required floors.
    6. The total heuristic is the sum of passenger action cost and lift movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels, passenger origins,
        and passenger destinations from static facts and initial state.
        """
        super().__init__(task)

        # Extract floor objects and map them to levels based on numerical suffix
        self.floor_to_level = {}
        all_objects = set()
        # Collect all terms from initial state, goals, and static facts
        for fact in task.initial_state | task.goals | task.static:
             parts = get_parts(fact)
             all_objects.update(parts) # Add all parts as potential objects

        # Filter for floors (assuming 'f' followed by digits) and sort numerically
        floor_names = sorted([obj for obj in all_objects if re.fullmatch(r'f\d+', obj)],
                             key=lambda f: int(f[1:])) # Sort by the number after 'f'

        for i, floor_name in enumerate(floor_names):
            self.floor_to_level[floor_name] = i + 1 # Assign levels starting from 1

        # Extract passenger origins and destinations
        self.passenger_to_origin_floor = {}
        self.passenger_to_destin_floor = {}
        self.all_passengers = set()

        # Origin facts are in the initial state
        for fact in task.initial_state:
            if match(fact, "origin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.passenger_to_origin_floor[passenger] = floor
                self.all_passengers.add(passenger)

        # Destin facts are static
        for fact in task.static:
             if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.passenger_to_destin_floor[passenger] = floor
                self.all_passengers.add(passenger)

        # Also collect passengers mentioned in goals (served) just in case
        for goal in self.goals:
             if match(goal, "served", "*"):
                 _, passenger = get_parts(goal)
                 self.all_passengers.add(passenger)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify the current floor of the lift.
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # If lift location is unknown, the state is likely invalid or unsolvable.
        # Return a large value if there are unserved passengers, else 0.
        if current_lift_floor is None:
             unserved_passengers_exist = any(f'(served {p})' not in state for p in self.all_passengers)
             return float('inf') if unserved_passengers_exist else 0

        current_level = self.floor_to_level.get(current_lift_floor, 0) # Default to 0 if floor not found (error case)


        # 2. Identify all passengers who are not yet served.
        unserved_passengers = set()
        for passenger in self.all_passengers:
            if f'(served {passenger})' not in state:
                unserved_passengers.add(passenger)

        if not unserved_passengers:
            # All passengers are served, goal reached. Heuristic is 0.
            return 0

        # 3. and 4. Calculate passenger-specific actions and identify required floors.
        passenger_action_cost = 0
        required_floors = set() # Floors the lift must visit

        # Determine which unserved passengers are boarded
        boarded_unserved_passengers = set()
        for fact in state:
            if match(fact, "boarded", "*"):
                passenger = get_parts(fact)[1]
                if passenger in unserved_passengers:
                    boarded_unserved_passengers.add(passenger)

        # Passengers who are unserved and not boarded must be waiting at their origin
        waiting_unserved_passengers = unserved_passengers - boarded_unserved_passengers

        # Process boarded unserved passengers
        for passenger in boarded_unserved_passengers:
            passenger_action_cost += 1 # Cost for 'depart' action
            destin_floor = self.passenger_to_destin_floor.get(passenger)
            if destin_floor:
                required_floors.add(destin_floor)

        # Process waiting unserved passengers
        for passenger in waiting_unserved_passengers:
            passenger_action_cost += 2 # Cost for 'board' + 'depart' actions
            origin_floor = self.passenger_to_origin_floor.get(passenger)
            destin_floor = self.passenger_to_destin_floor.get(passenger)
            if origin_floor:
                required_floors.add(origin_floor)
            if destin_floor:
                 required_floors.add(destin_floor)


        # 5. Calculate lift movement cost.
        lift_movement_cost = 0
        if required_floors:
            # Filter out any required floors that weren't mapped to a level (shouldn't happen if parsing is correct)
            valid_required_levels = [self.floor_to_level[f] for f in required_floors if f in self.floor_to_level]

            if valid_required_levels: # Ensure there are valid levels
                min_level = min(valid_required_levels)
                max_level = max(valid_required_levels)

                # Estimate movement to cover the range [min_level, max_level] starting from current_level
                # Cost to reach the closer extreme + cost to traverse the range
                dist_to_min = abs(current_level - min_level)
                dist_to_max = abs(current_level - max_level)
                range_dist = max_level - min_level

                # Estimate is minimum travel to reach one end and then traverse to the other
                # This is a non-admissible estimate aiming to capture necessary movement.
                lift_movement_cost = min(dist_to_min, dist_to_max) + range_dist

        # 6. Total heuristic is the sum.
        total_cost = passenger_action_cost + lift_movement_cost

        return total_cost
