from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
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
    # Ensure the number of parts matches the number of args, unless args has a wildcard at the end
    if len(parts) != len(args) and '*' not in args:
         return False
    # Use zip to handle cases where parts might be longer than args (if args ends with *)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components:
    1. A base cost for each unserved passenger (representing board and depart actions).
    2. An estimated movement cost based on the range of floors the lift needs to visit.

    # Assumptions
    - Each unserved passenger requires at least one 'board' and one 'depart' action.
    - The lift must visit the origin floor of any unboarded passenger and the
      destination floor of any unserved passenger (boarded or not).
    - The movement cost is estimated by the total vertical distance spanning
      the current lift floor and all necessary origin/destination floors.

    # Heuristic Initialization
    - Parses static facts to determine passenger origins and destinations.
    - Parses static facts to establish the floor order and create a floor-to-index mapping.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Identify all passengers who are not yet 'served'.
    2. Count the number of unserved passengers. If zero, the heuristic is 0.
    3. For each unserved passenger:
       - If the passenger is not 'boarded', their origin floor is a necessary stop.
       - Their destination floor is a necessary stop (whether boarded or not).
    4. Collect the set of all necessary floors: the current lift floor, plus
       the origin floors of unboarded passengers, plus the destination floors
       of all unserved passengers.
    5. Map these necessary floors to their numerical indices using the pre-computed mapping.
    6. Calculate the movement cost as the difference between the maximum and minimum
       floor index among the necessary floors. This estimates the vertical range the
       lift must cover.
    7. Calculate the total heuristic: (number of unserved passengers * 2) + movement cost.
       (The '* 2' accounts for the board and depart action for each passenger).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Passenger origins and destinations.
        - Floor ordering and mapping floor names to indices.
        """
        self.goals = task.goals  # Goal conditions (e.g., (served p1))
        static_facts = task.static  # Facts that are not affected by actions.

        # Store passenger origins and destinations
        self.passenger_origins = {}
        self.passenger_destins = {}
        for fact in static_facts:
            if match(fact, "origin", "*", "*"):
                p, f = get_parts(fact)[1:]
                self.passenger_origins[p] = f
            elif match(fact, "destin", "*", "*"):
                p, f = get_parts(fact)[1:]
                self.passenger_destins[p] = f

        # Build floor ordering and floor-to-index mapping
        above_map = {} # maps lower floor to upper floor
        all_floors = set()
        lower_floors = set()
        upper_floors = set()

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f_upper, f_lower = get_parts(fact)[1:]
                above_map[f_lower] = f_upper
                all_floors.add(f_upper)
                all_floors.add(f_lower)
                lower_floors.add(f_lower)
                upper_floors.add(f_upper)

        # Find the lowest floor (a floor that is a lower_floor but not an upper_floor)
        lowest_floor = (lower_floors - upper_floors).pop() # Assumes a single lowest floor

        # Build the ordered list of floors
        ordered_floors = []
        current_floor = lowest_floor
        while current_floor in above_map:
            ordered_floors.append(current_floor)
            current_floor = above_map[current_floor]
        ordered_floors.append(current_floor) # Add the highest floor

        # Create the floor-to-index mapping
        self.floor_to_index = {floor: i for i, floor in enumerate(ordered_floors)}

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
             # This should not happen in a valid miconic state, but handle defensively
             return float('inf') # Or some large value indicating an invalid state

        unserved_count = 0
        origin_floors_needed = set()
        destin_floors_needed = set()

        # Iterate through all passengers defined in the problem
        for passenger in self.passenger_destins.keys():
            served_fact = f"(served {passenger})"
            boarded_fact = f"(boarded {passenger})"
            origin_fact = f"(origin {passenger} {self.passenger_origins[passenger]})"

            # Check if the passenger is served
            if served_fact not in state:
                unserved_count += 1
                destin_floors_needed.add(self.passenger_destins[passenger])

                # Check if the passenger is boarded
                if boarded_fact not in state:
                    # Passenger is waiting at origin
                    origin_floors_needed.add(self.passenger_origins[passenger])
                # else: passenger is boarded, origin is not needed as a pickup point anymore

        # If all passengers are served, the goal is reached
        if unserved_count == 0:
            return 0

        # Calculate movement cost based on the range of necessary floors
        relevant_floor_indices = {self.floor_to_index[current_lift_floor]}

        for floor in origin_floors_needed:
             relevant_floor_indices.add(self.floor_to_index[floor])

        for floor in destin_floors_needed:
             relevant_floor_indices.add(self.floor_to_index[floor])

        min_index = min(relevant_floor_indices)
        max_index = max(relevant_floor_indices)

        movement_cost = max_index - min_index

        # Total heuristic: 2 actions per unserved passenger (board + depart) + movement cost
        # This is a non-admissible estimate as movement is shared and board/depart happen at specific floors.
        total_cost = unserved_count * 2 + movement_cost

        return total_cost

