from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    # Use zip to compare corresponding elements up to the length of the shorter sequence
    # fnmatch handles wildcards like '*'
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    passengers by summing the estimated actions needed for each unserved
    passenger independently. It considers the travel distance for the lift
    and the board/depart actions for each passenger.

    # Assumptions
    - The cost of each action (move, board, depart) is 1.
    - The heuristic calculates the cost for each unserved passenger as if they
      were the only passenger being transported, ignoring potential efficiencies
      from picking up/dropping off multiple passengers on a single trip or stop.
    - The 'above' predicates define a linear order of floors, allowing assignment
      of numerical levels to floors.
    - The PDDL instance is well-formed, meaning all floors mentioned in 'above',
      'origin', 'destin', and 'lift-at' are part of a single linear floor structure.

    # Heuristic Initialization
    - Parses the static facts to build a mapping from floor objects to numerical
      floor levels based on the 'above' predicates.
    - Stores the destination floor for each passenger from the static 'destin' facts.
    - Identifies all passengers from the 'destin' facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current floor of the lift.
    2. Initialize the total heuristic cost to 0.
    3. For each passenger:
       a. Check if the passenger is already 'served'. If yes, this passenger
          contributes 0 to the heuristic; continue to the next passenger.
       b. If the passenger is not served, determine if they are 'boarded' or
          waiting at their 'origin' floor.
       c. Get the passenger's destination floor from the pre-calculated map.
       d. If the passenger is 'boarded':
          - Calculate the travel distance for the lift from its current floor
            to the passenger's destination floor (absolute difference in floor levels).
          - Add this travel distance and 1 (for the 'depart' action) to the
            total heuristic cost.
       e. If the passenger is waiting at their 'origin' floor:
          - Find the passenger's origin floor from the current state facts.
          - Calculate the travel distance for the lift from its current floor
            to the passenger's origin floor.
          - Add this travel distance and 1 (for the 'board' action) to a
            temporary cost for this passenger.
          - Calculate the travel distance for the lift from the passenger's
            origin floor to their destination floor.
          - Add this travel distance and 1 (for the 'depart' action) to the
            temporary cost for this passenger.
          - Add the temporary cost for this passenger to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        self.goals = task.goals # Goal conditions (used implicitly by checking served status)
        static_facts = task.static # Facts that are not affected by actions.

        # Map floor objects to numerical levels based on 'above' predicates.
        self.floor_levels = {}
        above_map = {} # f_lower -> f_higher
        all_floors = set()

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, f_higher, f_lower = get_parts(fact)
                above_map[f_lower] = f_higher
                all_floors.add(f_higher)
                all_floors.add(f_lower)

        # Find the lowest floor (a floor that is a 'lower' floor but never a 'higher' floor)
        # Assuming there is exactly one lowest floor and the 'above' facts form a chain.
        lowest_floor = None
        potential_lowest = set(above_map.keys()) - set(above_map.values())
        if len(potential_lowest) == 1:
             lowest_floor = next(iter(potential_lowest))
        elif len(all_floors) == 1: # Case with only one floor
             lowest_floor = next(iter(all_floors))
        elif all_floors: # Fallback if structure is unexpected or disconnected
             # Try to find a floor that is not a 'higher' floor for any 'above' fact
             higher_floors = set(above_map.values())
             potential_lowest_fallback = [f for f in all_floors if f not in higher_floors]
             if len(potential_lowest_fallback) == 1:
                 lowest_floor = potential_lowest_fallback[0]
             elif potential_lowest_fallback:
                 # Multiple potential lowest floors or none found by this method, pick one arbitrarily
                 lowest_floor = potential_lowest_fallback[0]
                 # print(f"Warning: Multiple potential lowest floors found. Assuming {lowest_floor} is level 1.")
             else:
                 # No floor found that isn't a 'higher' floor. Could be a cycle or empty.
                 # Pick any floor to start if floors exist.
                 lowest_floor = next(iter(all_floors))
                 # print(f"Warning: Could not identify a clear lowest floor. Assuming {lowest_floor} is level 1.")
        # else: # No floors defined, lowest_floor remains None


        # Build floor levels starting from the lowest floor
        current_floor = lowest_floor
        level = 1
        while current_floor is not None:
            self.floor_levels[current_floor] = level
            current_floor = above_map.get(current_floor)
            level += 1

        # Store destination floor for each passenger from static facts
        self.destinations = {}
        for fact in static_facts:
             if match(fact, "destin", "*", "*"):
                 _, passenger, floor = get_parts(fact)
                 self.destinations[passenger] = floor

        # Identify all passengers
        self.passengers = set(self.destinations.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find the current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        if current_lift_floor is None:
             # This should not happen in a valid miconic state where lift exists
             # If lift location is unknown, goal is unreachable.
             return float('inf') # Should not be reachable

        # If floor_levels wasn't built (e.g., no floors or no above facts),
        # we can't calculate travel cost. Return a simple count of unserved passengers.
        if not self.floor_levels:
             unserved_count = 0
             for passenger in self.passengers:
                 if f"(served {passenger})" not in state:
                     unserved_count += 1
             # Each unserved needs at least board + depart (2 actions) if lift is at correct floors
             # This is a weak fallback heuristic.
             return unserved_count * 2


        current_lift_level = self.floor_levels.get(current_lift_floor)
        if current_lift_level is None:
             # Current lift floor not in our floor map - indicates issue with floor parsing or state
             # This floor should have been part of the 'all_floors' set if parsing was correct.
             return float('inf') # Should not happen in valid problems


        total_cost = 0  # Initialize action cost counter.

        # Iterate through all passengers and sum their individual costs if not served
        for passenger in self.passengers:
            # Check if the passenger is served
            if f"(served {passenger})" in state:
                continue # Passenger is served, contributes 0 cost

            # Passenger is not served. Check if boarded or waiting at origin.
            destination_floor = self.destinations.get(passenger)
            if destination_floor is None:
                 # Passenger has no destination defined - problem is likely ill-formed
                 return float('inf') # Should not happen in valid problems

            destination_level = self.floor_levels.get(destination_floor)
            if destination_level is None:
                 # Destination floor not in our floor map - indicates issue
                 return float('inf') # Should not happen in valid problems


            if f"(boarded {passenger})" in state:
                # Passenger is boarded, needs to travel to destination and depart
                travel_cost = abs(current_lift_level - destination_level)
                depart_cost = 1
                total_cost += travel_cost + depart_cost
            else:
                # Passenger is unboarded, waiting at origin. Needs pickup and dropoff.
                # Find the origin floor from the current state
                origin_floor = None
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                        origin_floor = get_parts(fact)[2]
                        break

                if origin_floor is None:
                    # This passenger is not served, not boarded, and not at an origin.
                    # This implies an invalid state or the passenger was somehow removed.
                    # For a well-formed problem, this shouldn't happen for unserved passengers.
                    # Return infinity or a very high cost to prune this path.
                    return float('inf') # Invalid state encountered

                origin_level = self.floor_levels.get(origin_floor)
                if origin_level is None:
                     # Origin floor not in our floor map - indicates issue
                     return float('inf') # Should not happen in valid problems


                # Cost to pick up: travel from current lift to origin + board action
                travel_to_origin_cost = abs(current_lift_level - origin_level)
                board_cost = 1

                # Cost to drop off: travel from origin to destination + depart action
                travel_origin_to_destin_cost = abs(origin_level - destination_level)
                depart_cost = 1

                total_cost += travel_to_origin_cost + board_cost + travel_origin_to_destin_cost + depart_cost

        return total_cost
