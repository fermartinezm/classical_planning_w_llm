from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or fact[0] != '(' or fact[-1] != ')':
        # Return empty list for malformed facts, or handle as error
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
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
    This heuristic estimates the number of actions required to serve all passengers.
    It sums the number of 'board' actions needed, the number of 'depart' actions
    needed, and an estimate of the minimum number of lift 'move' actions
    required to visit all necessary floors (passenger origins and destinations).

    # Assumptions
    - The floor structure is linear, defined by 'above' predicates.
    - All passengers must be served to reach the goal.
    - The cost of each action (board, depart, up, down) is 1.
    - The lift has unlimited capacity.

    # Heuristic Initialization
    - Extracts passenger destination floors from static facts.
    - Builds a mapping from floor names to numerical order based on 'above' facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift from the state.
    2. Identify all passengers that are not yet served.
    3. For each unserved passenger:
       - If the passenger is waiting at their origin floor (i.e., the state contains `(origin passenger floor)`):
         - Increment the count of needed 'board' actions.
         - Add their origin floor to the set of floors the lift must visit (pickup floors).
       - If the passenger is boarded (i.e., the state contains `(boarded passenger)` and not `(served passenger)`):
         - Increment the count of needed 'depart' actions.
         - Add their destination floor (looked up from static facts) to the set of floors the lift must visit (dropoff floors).
    4. The total number of needed 'board' actions is the count of unboarded unserved passengers.
    5. The total number of needed 'depart' actions is the count of boarded unserved passengers.
    6. Calculate the set of all floors the lift must visit: the union of pickup floors and dropoff floors.
    7. Estimate the minimum number of 'move' actions (up/down) required for the lift to travel from its current floor and visit all floors in the required set.
       - If the set of required floors is empty, the move cost is 0.
       - Otherwise, find the lowest and highest floor numbers among the required floors using the floor-to-number mapping.
       - The estimated move cost is the span of the required floors (highest number - lowest number) plus the minimum distance from the current lift floor's number to either the lowest or highest required floor number. This formula calculates the minimum moves to traverse the range containing all required floors, starting from the current position.
    8. The total heuristic value is the sum of the total needed 'board' actions, the total needed 'depart' actions, and the estimated 'move' actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal conditions (implicitly, all passengers served).
        - Static facts (`destin` relationships and `above` relationships).
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store passenger destination floors from static facts.
        self.passenger_destinations = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.passenger_destinations[passenger] = floor

        # Build floor mapping from 'above' facts.
        self.floor_to_number = {}
        all_floors = set()
        above_map = {} # Maps floor -> floor immediately above it

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, f1, f2 = get_parts(fact)
                all_floors.add(f1)
                all_floors.add(f2)
                above_map[f1] = f2

        # Find the lowest floor (a floor that is "above" nothing else)
        # This assumes a single linear chain of floors.
        lowest_floor = None
        floors_that_are_above = set(above_map.values())
        # Find a floor that is mentioned but is not the target of any 'above' relation
        potential_lowest = [f for f in all_floors if f not in floors_that_are_above]

        if len(potential_lowest) == 1:
             lowest_floor = potential_lowest[0]
        elif len(potential_lowest) > 1:
             # Handle cases with multiple disjoint floor chains or complex structures
             # For standard miconic, this shouldn't happen.
             # As a fallback, try to find the one with the smallest number if floors are f<num>
             try:
                 lowest_floor = min(potential_lowest, key=lambda f: int(f[1:]))
             except (ValueError, IndexError):
                 # Cannot parse floor names as numbers, pick one arbitrarily
                 lowest_floor = potential_lowest[0] # Arbitrary pick


        # Traverse the 'above' chain to build the number mapping
        current_floor = lowest_floor
        number = 1
        while current_floor is not None:
            self.floor_to_number[current_floor] = number
            current_floor = above_map.get(current_floor)
            number += 1

        # Note: If there are no above facts but floors exist (e.g., single floor),
        # all_floors will be populated, above_map will be empty, potential_lowest
        # will contain all floors, lowest_floor will be one of them, and the loop
        # will run only once, mapping that floor to 1. This is correct for a single floor.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break
        # If lift-at fact is missing, the state is likely invalid.
        # Returning infinity signals this state is likely unreachable or problematic.
        if current_lift_floor is None:
             return float('inf')

        # Identify unserved passengers and required floors
        unboarded_unserved_count = 0
        boarded_unserved_count = 0
        pickup_floors = set()
        dropoff_floors = set()

        # Collect all passengers mentioned in relevant state/static facts
        all_passengers = set()
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] in ["origin", "boarded", "served"]:
                  if len(parts) > 1:
                      all_passengers.add(parts[1])
        # Add passengers from destinations, in case they aren't mentioned in initial state facts
        all_passengers.update(self.passenger_destinations.keys())


        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}
        origin_facts = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "origin", "*", "*")}


        for passenger in all_passengers:
            if passenger in served_passengers:
                continue # This passenger is already served

            # Passenger is unserved
            if passenger in boarded_passengers:
                # Unserved and boarded
                boarded_unserved_count += 1
                # Add destination floor to dropoff floors
                if passenger in self.passenger_destinations:
                    dropoff_floors.add(self.passenger_destinations[passenger])
                # else: Passenger boarded but no destination? Invalid problem/state.
            elif passenger in origin_facts:
                # Unserved and unboarded (waiting at origin)
                unboarded_unserved_count += 1
                # Add origin floor to pickup floors
                pickup_floors.add(origin_facts[passenger])
            # else: Unserved, not boarded, not at origin? Invalid state.


        # Calculate move cost
        required_floors = pickup_floors | dropoff_floors
        move_cost = 0

        if required_floors:
            # Map required floors to numbers, filtering out any floors not in our mapping
            required_nums = {self.floor_to_number[f] for f in required_floors if f in self.floor_to_number}

            # If after filtering, there are still required floors with valid mappings
            if required_nums:
                min_req_num = min(required_nums)
                max_req_num = max(required_nums)
                current_num = self.floor_to_number.get(current_lift_floor, None) # Get number, None if floor not mapped

                # If current lift floor is not mapped, we can't calculate move cost meaningfully.
                # This implies an invalid state or problem definition.
                if current_num is None:
                     return float('inf')

                # Calculate minimum moves to visit all floors in the range [min_req_num, max_req_num]
                # starting from current_num.
                # This is the span + min distance from current to either end.
                span = max_req_num - min_req_num
                dist_to_min = abs(current_num - min_req_num)
                dist_to_max = abs(current_num - max_req_num)

                move_cost = span + min(dist_to_min, dist_to_max)
            # else: required_floors was not empty, but none of them were in the floor mapping.
            # This suggests an issue with the problem definition or state. Move cost remains 0.


        # Total heuristic is sum of actions needed
        total_cost = unboarded_unserved_count + boarded_unserved_count + move_cost

        return total_cost
