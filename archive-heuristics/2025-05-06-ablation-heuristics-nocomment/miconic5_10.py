# Assuming heuristics.heuristic_base exists and contains the Heuristic class
from heuristics.heuristic_base import Heuristic

# Helper function
def get_parts(fact):
    """Helper to parse a fact string into predicate and arguments."""
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    return fact[1:-1].split()

# The main heuristic class
class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        This heuristic estimates the remaining cost to solve a Miconic planning
        problem instance by summing the estimated costs for each passenger
        who has not yet been served. The cost for each unserved passenger is
        calculated independently, assuming they are transported from their
        current location (origin floor or inside the lift) to their destination.
        This calculation includes the estimated travel distance for the lift
        and the costs of the 'board' and 'depart' actions.

    Assumptions:
        - Floors are named using a convention like 'f1', 'f2', ..., 'fn',
          and these names imply a linear order based on the numerical part.
        - The 'above' predicates in the domain confirm this linear, numerical
          ordering of floors.
        - The cost of each action ('up', 'down', 'board', 'depart') is 1.
        - The heuristic does not account for the possibility of transporting
          multiple passengers simultaneously or optimizing the lift's path
          to serve multiple passengers efficiently. It provides a lower bound
          on the actions needed if passengers were served one by one, but is
          not strictly admissible due to ignoring potential batching.

    Heuristic Initialization:
        The constructor processes the static facts of the planning task:
        - It extracts the destination floor for each passenger from the
          '(destin ?person ?floor)' facts and stores this mapping.
        - It identifies all floor objects mentioned in the '(above ?floor1 ?floor2)'
          facts. It then sorts these floor names numerically based on the number
          following 'f' (e.g., 'f1' < 'f2'). A mapping from each floor name to its
          numerical index in this sorted list is created. This mapping is used
          to calculate the distance between any two floors as the absolute
          difference of their indices.
        - It identifies the set of passengers that need to be served to reach
          the goal state, based on the '(served ?person)' goals.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state (represented as a frozenset of facts):
        1.  Check if the state is the goal state. If all goal conditions
            (all required '(served ?person)' facts) are present in the state,
            the heuristic value is 0.
        2.  Identify the current floor of the lift by finding the fact
            '(lift-at ?floor)' in the state. If the lift location cannot be
            determined or refers to an unknown floor, return infinity
            (indicating a potentially invalid or unreachable state).
        3.  Initialize a total heuristic cost to 0.
        4.  Identify all passengers who are currently boarded in the lift
            by finding the facts '(boarded ?person)' in the state.
        5.  Identify the origin floor for all passengers waiting at a floor
            by finding the facts '(origin ?person ?floor)' in the state.
        6.  Iterate through the set of passengers that need to be served
            (identified during initialization from the goal).
        7.  For each passenger in this set:
            a.  If the fact '(served ' + passenger + ')' is already in the current state,
                this passenger is served, and contributes 0 to the heuristic.
                Continue to the next passenger.
            b.  Retrieve the passenger's destination floor 'D' using the mapping
                created during initialization. If the destination is unknown or
                refers to an unknown floor, return infinity.
            c.  Check if the passenger is in the set of boarded passengers.
                i.  If the passenger is boarded:
                    - Calculate the distance between the current lift floor 'L'
                      and the destination floor 'D' using the floor index mapping:
                      `distance = abs(self.floor_to_index[L] - self.floor_to_index[D])`.
                    - The estimated cost for this passenger is `distance + 1`
                      (travel to destination + 'depart' action). Add this to the
                      total heuristic cost.
                ii. If the passenger is not boarded (they must be waiting at their origin):
                    - Find the passenger's origin floor 'O' from the
                      `unboarded_origins` mapping derived from the state. If the
                      origin is not found or refers to an unknown floor, return infinity.
                    - Calculate the distance from the current lift floor 'L' to the
                      origin floor 'O': `dist_L_O = abs(self.floor_to_index[L] - self.floor_to_index[O])`.
                    - Calculate the distance from the origin floor 'O' to the
                      destination floor 'D': `dist_O_D = abs(self.floor_to_index[O] - self.floor_to_index[D])`.
                    - The estimated cost for this passenger is `dist_L_O + 1 + dist_O_D + 1`
                      (travel to origin + 'board' action + travel to destination + 'depart' action).
                      Add this to the total heuristic cost.
            d.  If a passenger is not served, not boarded, and not at an origin
                (which should not happen in a valid state), return infinity.
        8.  Return the calculated total heuristic cost.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # Extract passenger destinations from static facts
        self.passenger_destinations = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "destin" and len(parts) == 3:
                _, passenger, destination = parts
                self.passenger_destinations[passenger] = destination

        # Determine floor order and create floor_to_index map
        # Collect all floor names from 'above' predicates
        floor_names = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above" and len(parts) == 3:
                _, f1, f2 = parts
                floor_names.add(f1)
                floor_names.add(f2)

        # Sort floor names based on the number part (assuming f<number> format)
        # This relies on the naming convention shown in examples.
        try:
            # Extract number from floor name (e.g., 'f1' -> 1) and sort numerically
            sorted_floors = sorted(list(floor_names), key=lambda f: int(f[1:]))
        except (ValueError, IndexError):
             # Fallback if floor names are not strictly f<number> or empty
             # This might happen in malformed problems, but for standard miconic it should work.
             # A more robust parser would be needed for arbitrary floor names and 'above' relations.
             print(f"Warning: Floor names might not be in 'f<number>' format or are malformed. Sorting alphabetically as fallback. Floors: {floor_names}")
             sorted_floors = sorted(list(floor_names)) # Simple alphabetical sort as fallback

        self.floor_to_index = {floor: index for index, floor in enumerate(sorted_floors)}

        # Identify passengers that need to be served from the goal
        self.goal_passengers = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "served" and len(parts) == 2:
                _, passenger = parts
                self.goal_passengers.add(passenger)


    def __call__(self, node):
        state = node.state

        # Check if goal is reached (heuristic is 0)
        if self.goals <= state:
             return 0

        # Find current lift location
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at" and len(parts) == 2:
                _, current_lift_floor = parts
                break

        # If lift location is not found or refers to an unknown floor, state is invalid
        if current_lift_floor is None or current_lift_floor not in self.floor_to_index:
             return float('inf')

        total_cost = 0

        # Track which passengers are boarded
        boarded_passengers = set()
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "boarded" and len(parts) == 2:
                _, passenger = parts
                boarded_passengers.add(passenger)

        # Track origins for unboarded passengers
        unboarded_origins = {}
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == "origin" and len(parts) == 3:
                  _, passenger, origin_floor = parts
                  unboarded_origins[passenger] = origin_floor

        # Calculate cost for each unserved passenger
        for passenger in self.goal_passengers:
            # If already served, cost is 0 for this passenger, skip
            if '(served ' + passenger + ')' in state:
                continue

            # Get destination floor for the passenger
            destination_floor = self.passenger_destinations.get(passenger)
            # If a goal passenger has no destination defined or destination floor is unknown, state is invalid
            if destination_floor is None or destination_floor not in self.floor_to_index:
                 return float('inf')

            if passenger in boarded_passengers:
                # Passenger is boarded, needs to go from current lift floor to destination and depart
                dist_L_D = abs(self.floor_to_index[current_lift_floor] - self.floor_to_index[destination_floor])
                cost_p = dist_L_D + 1 # travel + depart action
                total_cost += cost_p
            elif passenger in unboarded_origins:
                # Passenger is waiting at origin, needs lift to come, board, travel to destin, and depart
                origin_floor = unboarded_origins[passenger]

                # Ensure origin floor is valid
                if origin_floor not in self.floor_to_index:
                     return float('inf')

                dist_L_O = abs(self.floor_to_index[current_lift_floor] - self.floor_to_index[origin_floor])
                dist_O_D = abs(self.floor_to_index[origin_floor] - self.floor_to_index[destination_floor])
                cost_p = dist_L_O + 1 + dist_O_D + 1 # travel to origin + board + travel to destin + depart
                total_cost += cost_p
            else:
                 # Passenger is not served, not boarded, and not at an origin. Invalid state.
                 # This case should ideally not be reachable in a valid problem instance
                 # and valid state transitions.
                 return float('inf')

        return total_cost
