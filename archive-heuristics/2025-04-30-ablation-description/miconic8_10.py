# Assuming Heuristic base class is available in the environment
from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
from collections import defaultdict, deque

# Helper function to parse facts
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the cost to serve all unserved passengers required by the goal.
    For each such passenger, it estimates the minimum number of actions needed based on their current state:
    - If not boarded: Estimate includes moving the lift to their origin floor, boarding them, moving the lift to their destination floor, and departing them.
    - If boarded: Estimate includes moving the lift to their destination floor and departing them.
    The total heuristic value is the sum of these estimated costs for all unserved goal passengers.
    Floor movement cost is estimated by the absolute difference in floor levels,
    where floor levels are determined by parsing the 'above' predicates assuming a linear floor structure.

    # Assumptions
    - Each action (board, depart, up, down) has a cost of 1.
    - The lift has unlimited capacity.
    - The 'above' predicates define a strict total ordering of floors, forming a linear chain (e.g., f1 < f2 < f3 ...).
    - All passengers required by the goal have a 'destin' fact in the static information.
    - Unboarded, unserved goal passengers have an 'origin' fact in the current state.

    # Heuristic Initialization
    - Extract goal conditions to identify all passengers that need to be served.
    - Parse 'above' predicates from static facts to determine the floor ordering and map floor names to levels. This assumes a linear chain structure (f_min, f_min+1, ... f_max).
    - Extract destination floors for each passenger from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of passengers that must be served according to the goal state.
    2. Determine the current floor of the lift from the state. If the lift location is unknown, the state is likely invalid or unsolvable, return infinity.
    3. Identify which of the goal passengers are currently 'served' in the state. The heuristic will only consider those who are not yet served.
    4. For each unserved goal passenger:
        a. Get their destination floor (pre-calculated during initialization from static facts). If the destination is missing, the problem is ill-defined, return infinity.
        b. Check if the passenger is currently 'boarded' in the state.
        c. If the passenger is 'boarded':
            - Calculate the vertical distance between the current lift floor and the passenger's destination floor using the pre-calculated floor levels.
            - Add this distance + 1 (for the 'depart' action) to the total heuristic for this passenger.
        d. If the passenger is not 'boarded':
            - Find the passenger's origin floor from the current state. If the origin is missing for an unboarded, unserved passenger, the state is likely invalid, return infinity.
            - Calculate the vertical distance between the current lift floor and the passenger's origin floor.
            - Calculate the vertical distance between the passenger's origin floor and their destination floor.
            - Add the first distance + 1 (for 'board') + the second distance + 1 (for 'depart') to the total heuristic for this passenger.
    5. Sum the costs calculated for each unserved goal passenger to get the total heuristic value.
    6. If the goal state is already reached (all goal conditions are met), the heuristic is 0. This check is done first.
    7. If floor levels could not be built from static facts (e.g., non-linear structure, no floors), return a high value if there are unserved goal passengers, indicating a potentially unsolvable or malformed problem for this heuristic.

    """

    def __init__(self, task):
        """Initialize the heuristic."""
        self.goals = task.goals
        self.static_facts = task.static

        # Extract all passengers from the goals (they are the objects of 'served' predicates)
        self.all_goal_passengers = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

        # Extract destination floors for each passenger from static facts
        self.passenger_destinations = {}
        for fact in self.static_facts:
            if match(fact, "destin", "*", "*"):
                passenger, destination_floor = get_parts(fact)[1:]
                self.passenger_destinations[passenger] = destination_floor

        # Build floor ordering and map floor names to levels assuming a linear chain
        self.floor_levels = self._build_floor_levels()

    def _build_floor_levels(self):
        """
        Parses 'above' facts to determine floor ordering and assign levels.
        Assumes 'above' facts define a simple linear chain (f1 < f2 < ...).
        Finds the lowest floor (in-degree 0 in the 'above' graph) and builds upwards.
        Returns a dictionary mapping floor names to integer levels, or an empty dictionary if building fails.
        """
        above_graph = defaultdict(list)
        in_degree = defaultdict(int)
        all_floors = set()

        for fact in self.static_facts:
            if match(fact, "above", "*", "*"):
                f1, f2 = get_parts(fact)[1:]
                above_graph[f1].append(f2)
                in_degree[f2] += 1
                all_floors.add(f1)
                all_floors.add(f2)

        if not all_floors:
             return {} # No floors defined

        # Find the unique lowest floor (in-degree 0)
        lowest_floors = [f for f in all_floors if in_degree[f] == 0]

        if len(lowest_floors) != 1:
             # Expected exactly one lowest floor for a simple chain
             return {} # Indicate failure

        floor_levels = {}
        current_floor = lowest_floors[0]
        level = 1
        floor_levels[current_floor] = level
        processed_count = 1

        # Build upwards following the unique 'above' link
        while processed_count < len(all_floors):
             # Find the unique floor f_next such that (above current_floor f_next)
             next_floors = [f2 for f1, f2 in above_graph.items() if f1 == current_floor]

             if len(next_floors) != 1:
                  # Expected exactly one floor directly above the current one for a simple chain
                  return {} # Indicate failure

             current_floor = next_floors[0]
             level += 1
             floor_levels[current_floor] = level
             processed_count += 1

        # Final check: ensure all floors were assigned a level
        if processed_count != len(all_floors):
             return {} # Indicate failure

        return floor_levels

    def _get_floor_level(self, floor_name):
        """Get the integer level for a floor name."""
        # If floor levels couldn't be built, return a large value
        if not self.floor_levels:
            # Fallback: try parsing f<number> (less robust)
            try:
                # Assumes floor names are like 'f1', 'f10', etc.
                # This fallback is less reliable than the graph method
                return int(floor_name[1:])
            except (ValueError, IndexError):
                # Cannot parse, return a large value indicating an issue
                return 1000000 # Arbitrary large value

        # Return the pre-calculated level, or a large value if floor name is unknown
        return self.floor_levels.get(floor_name, 1000000) # Arbitrary large value for unknown floors

    def _get_floor_distance(self, floor1, floor2):
        """Calculate the absolute distance between two floors based on levels."""
        level1 = self._get_floor_level(floor1)
        level2 = self._get_floor_level(floor2)
        # If either level is the large error value, the distance is effectively infinite
        if level1 >= 1000000 or level2 >= 1000000:
             return 1000000 # Return large value indicating inability to calculate distance
        return abs(level1 - level2)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Check if the goal is reached first
        if self.goals <= state:
            return 0

        # Identify which passengers are required by the goal and are not yet served
        served_passengers_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_goal_passengers = self.all_goal_passengers - served_passengers_in_state

        # If floor levels couldn't be built and there are goal passengers to serve,
        # return a high value indicating a potential problem with the domain/instance structure
        if not self.floor_levels and unserved_goal_passengers:
             return len(unserved_goal_passengers) * 1000000 # Arbitrary high cost per passenger

        total_heuristic = 0

        # Find the current lift location
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        if current_lift_floor is None:
             # This shouldn't happen in a valid state, but handle defensively
             return float('inf') # State is likely invalid or unsolvable

        # Check if the current lift floor is valid (has a level)
        if self._get_floor_level(current_lift_floor) >= 1000000:
             return float('inf') # State is likely invalid or unsolvable


        # Identify current origin floors for passengers not yet boarded
        # Note: 'origin' facts are removed after boarding, so we only find origins for non-boarded passengers
        passenger_origins_in_state = {}
        for fact in state:
            if match(fact, "origin", "*", "*"):
                 passenger, origin_floor = get_parts(fact)[1:]
                 passenger_origins_in_state[passenger] = origin_floor

        # Identify which passengers are currently boarded
        boarded_passengers_in_state = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}


        for passenger in unserved_goal_passengers:
            destination_floor = self.passenger_destinations.get(passenger)
            if destination_floor is None:
                 # This passenger is in the goal but has no destination? Problematic.
                 # This indicates an invalid problem definition. Return infinity.
                 return float('inf')

            # Check if destination floor is valid
            if self._get_floor_level(destination_floor) >= 1000000:
                 return float('inf') # Invalid problem definition


            if passenger in boarded_passengers_in_state:
                # Passenger is boarded, needs to go to destination and depart
                # Cost = distance(current_lift_floor, destination) + depart_cost
                distance_to_dest = self._get_floor_distance(current_lift_floor, destination_floor)
                if distance_to_dest >= 1000000: return float('inf') # Distance calculation failed
                cost = distance_to_dest + 1 # 1 for depart action
                total_heuristic += cost
            else:
                # Passenger is not boarded, needs to go to origin, board, go to destination, depart
                origin_floor = passenger_origins_in_state.get(passenger)
                if origin_floor is None:
                     # Passenger is unserved, not boarded, but has no origin fact in the current state?
                     # This could happen if the origin fact was removed without boarding (invalid state)
                     # or if the passenger was never at an origin (invalid problem).
                     # For a robust heuristic, return infinity.
                     return float('inf')

                # Check if origin floor is valid
                if self._get_floor_level(origin_floor) >= 1000000:
                     return float('inf') # Invalid state or problem definition


                # Cost = distance(current_lift_floor, origin) + board_cost + distance(origin, destination) + depart_cost
                distance_to_origin = self._get_floor_distance(current_lift_floor, origin_floor)
                distance_origin_to_dest = self._get_floor_distance(origin_floor, destination_floor)

                if distance_to_origin >= 1000000 or distance_origin_to_dest >= 1000000:
                     return float('inf') # Distance calculation failed

                cost = distance_to_origin + 1 + \
                       distance_origin_to_dest + 1
                total_heuristic += cost

        return total_heuristic
