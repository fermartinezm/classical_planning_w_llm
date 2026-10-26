# Assume Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or fact[0] != '(' or fact[-1] != ')':
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
    It sums the minimum actions needed for each passenger (board + depart, or just depart if already boarded)
    and adds an estimate of the lift movement cost to visit all necessary floors (origins for waiting,
    destinations for boarded).

    # Assumptions
    - Floors are ordered linearly (e.g., f1 < f2 < f3 ...). The heuristic determines this order
      from the `above` predicates in the static facts by sorting floor names numerically.
    - Each passenger needs to be boarded once (if waiting) and departed once (if waiting or boarded).
    - The lift can carry multiple passengers. The movement cost estimates the travel needed
      to visit all required floors.

    # Heuristic Initialization
    - Extracts the goal conditions (`(served ?p)` for all passengers).
    - Extracts static facts (`above` relationships and `destin` for passengers).
    - Builds a mapping from floor names to numerical indices based on the `above` predicates.
    - Stores the destination floor for each passenger.
    - Stores a list of all passengers.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check for Goal State: If all goal conditions (`(served ?p)`) are met in the current state, the heuristic is 0.

    2. Calculate Passenger Action Cost:
       - Initialize passenger action cost to 0.
       - Identify passengers who are waiting (`(origin ?p ?f)`) and those who are boarded (`(boarded ?p)`).
       - For every passenger found to be waiting, add 2 to the cost (representing the future `board` and `depart` actions needed).
       - For every passenger found to be boarded, add 1 to the cost (representing the future `depart` action needed).
       - Passengers who are `served` do not contribute to this cost.

    3. Identify Required Floors for Lift Travel:
       - Determine the current floor of the lift from the state (`(lift-at ?f)`).
       - Create a set of floor indices that the lift must visit.
       - For each passenger identified as waiting, add the index of their origin floor to the set.
       - For each passenger identified as boarded, add the index of their destination floor (retrieved from static facts) to the set.

    4. Calculate Lift Movement Cost:
       - Initialize lift movement cost to 0.
       - If the set of required floor indices is empty, no movement is needed for currently waiting/boarded passengers, so the cost is 0. (This case should ideally coincide with the goal state check).
       - If the set is not empty:
         - Find the minimum and maximum floor indices among the required floors.
         - Find the index of the lift's current floor.
         - Estimate the minimum number of move actions needed to travel from the current floor to visit all floors within the range defined by the minimum and maximum required indices. This is calculated as the distance from the current floor to the closer end of the required range, plus the total span of the required range: `min(abs(current_idx - min_req_idx), abs(current_idx - max_req_idx)) + (max_req_idx - min_req_idx)`. Add this value to the total cost.

    5. Sum Costs: The total heuristic value is the sum of the Passenger Action Cost and the Lift Movement Cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # 1. Determine floor order and create index mapping
        # Find all floors mentioned in 'above' predicates
        floors = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'above':
                # Add both floors from the 'above' predicate
                if len(parts) > 1: floors.add(parts[1])
                if len(parts) > 2: floors.add(parts[2])


        # Sort floors based on the number in their name (assuming f1, f2, f10, f20 format)
        # This assumes a linear floor structure like f1 < f2 < ... < fn
        def floor_sort_key(floor_name):
            # Extract the number from the floor name (e.g., 'f1' -> 1, 'f10' -> 10)
            try:
                # Handle potential non-numeric parts or empty strings after 'f'
                num_str = floor_name[1:]
                if not num_str: return float('inf') # Malformed name
                return int(num_str)
            except ValueError:
                # Handle unexpected floor names if necessary, or put them last
                return float('inf') # Put malformed names at the end

        sorted_floors = sorted(list(floors), key=floor_sort_key)

        self.floor_to_index = {floor: i for i, floor in enumerate(sorted_floors)}
        self.index_to_floor = {i: floor for i, floor in enumerate(sorted_floors)}

        # 2. Extract passenger destinations (from static facts)
        self.destinations = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'destin':
                if len(parts) > 2:
                    passenger, destination_floor = parts[1], parts[2]
                    self.destinations[passenger] = destination_floor

        # 3. Get list of all passengers (from destinations)
        self.all_passengers = list(self.destinations.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Check for Goal State
        # The goal is typically (and (served p1) (served p2) ...).
        # We can check if all goals from self.goals are in the state.
        if all(goal in state for goal in self.goals):
            return 0

        total_cost = 0

        # 2. Calculate Passenger Action Cost
        waiting_passengers = {} # {p: origin_floor}
        boarded_passengers = set()
        # served_passengers = set() # Not strictly needed for cost calculation

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == 'origin':
                if len(parts) > 2:
                    p, f = parts[1], parts[2]
                    waiting_passengers[p] = f
            elif parts[0] == 'boarded':
                if len(parts) > 1:
                    p = parts[1]
                    boarded_passengers.add(p)
            # elif parts[0] == 'served':
            #     if len(parts) > 1:
            #         p = parts[1]
            #         served_passengers.add(p)

        # Add cost for board/depart actions
        # Passengers who are waiting need board (1) + depart (1) = 2 actions
        total_cost += len(waiting_passengers) * 2
        # Passengers who are boarded need depart (1) action
        total_cost += len(boarded_passengers) * 1

        # 3. Identify Required Floors for Lift Travel
        required_floor_indices = set()

        # Add origin floors for waiting passengers
        for p, f_origin in waiting_passengers.items():
             if f_origin in self.floor_to_index: # Ensure floor is known
                 required_floor_indices.add(self.floor_to_index[f_origin])

        # Add destination floors for boarded passengers
        for p in boarded_passengers:
            if p in self.destinations: # Ensure passenger destination is known
                f_destin = self.destinations[p]
                if f_destin in self.floor_to_index: # Ensure floor is known
                    required_floor_indices.add(self.floor_to_index[f_destin])

        # 4. Calculate Lift Movement Cost
        if not required_floor_indices:
             # If no required floors, movement cost is 0.
             # This should only happen if all unserved passengers are neither waiting nor boarded,
             # which implies a state inconsistency or that they are served (handled by goal check).
             pass # total_cost already includes passenger actions if any unserved exist

        else:
            # Find current lift floor index
            current_floor = None
            for fact in state:
                parts = get_parts(fact)
                if parts and parts[0] == 'lift-at':
                    if len(parts) > 1:
                        current_floor = parts[1]
                    break

            # Calculate movement cost only if lift location is known and floor is valid
            if current_floor and current_floor in self.floor_to_index:
                current_floor_idx = self.floor_to_index[current_floor]

                min_required_idx = min(required_floor_indices)
                max_required_idx = max(required_floor_indices)

                # Estimate moves: distance to closer end + span of required floors
                # This estimates the minimum travel to reach one end of the required range
                # and then sweep across the entire range.
                lift_move_cost = min(abs(current_floor_idx - min_required_idx), abs(current_floor_idx - max_required_idx)) + (max_required_idx - min_required_idx)
                total_cost += lift_move_cost
            else:
                 # This case indicates an unexpected state (lift location unknown or invalid floor).
                 # Assuming valid states based on problem description.
                 pass # Movement cost remains 0

        # 5. Return total estimated cost
        return total_cost
