from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    passengers who have not yet reached their destination. It calculates the
    cost for each unserved passenger independently and sums these costs.

    # Assumptions
    - Actions (board, depart, up, down) have a unit cost of 1.
    - Floors are arranged linearly, and the 'above' predicate defines this order.
    - A passenger is either waiting at their origin, boarded in the lift, or served.
    - The lift is always at exactly one floor.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the static 'destin' facts.
    - Builds a mapping from floor names (e.g., 'f1', 'f2') to numerical floor levels
      based on the static 'above' facts. This allows calculating the distance
      (number of moves) between floors.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current floor of the lift by finding the 'lift-at' fact.
    2. Initialize the total heuristic cost to 0.
    3. Iterate through all passengers whose destinations are known (from initialization).
    4. For each passenger:
       a. Check if the passenger is already 'served'. If yes, this passenger
          contributes 0 to the heuristic; continue to the next passenger.
       b. If the passenger is not served, check if they are 'boarded'.
       c. If the passenger is 'boarded':
          - They are in the lift and need to reach their destination floor.
          - Calculate the number of floor moves required for the lift to go
            from its current floor to the passenger's destination floor. This is
            the absolute difference between the numerical levels of the floors.
          - Add 1 for the 'depart' action needed at the destination.
          - Add this cost to the total heuristic.
       d. If the passenger is not 'boarded' (and not served), they must be
          waiting at their 'origin' floor.
          - Find the passenger's origin floor by searching for the 'origin' fact
            in the current state.
          - Calculate the number of floor moves required for the lift to go
            from its current floor to the passenger's origin floor.
          - Add 1 for the 'board' action needed at the origin.
          - Calculate the number of floor moves required for the lift to go
            from the origin floor to the passenger's destination floor.
          - Add 1 for the 'depart' action needed at the destination.
          - Add this total cost (moves to origin + board + moves to destin + depart)
            to the total heuristic.
    5. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting passenger destinations and
        building the floor number mapping.
        """
        self.goals = task.goals  # Goal conditions (used implicitly by checking 'served')
        static_facts = task.static  # Facts that are not affected by actions.

        # Extract passenger destinations from static facts
        self.destin_map = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'destin':
                self.destin_map[parts[1]] = parts[2]

        # Build floor number mapping from static 'above' facts
        # 'above f_above f_below' means f_above is directly above f_below
        floor_above_to_floor_below = {}
        all_floors = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'above':
                f_above, f_below = parts[1], parts[2]
                floor_above_to_floor_below[f_above] = f_below
                all_floors.add(f_above)
                all_floors.add(f_below)

        # Find the lowest floor: it's a floor that appears as f_below but not as f_above
        floors_below = set(floor_above_to_floor_below.values())
        floors_above = set(floor_above_to_floor_below.keys())
        # The lowest floor is the one in floors_below that is not in floors_above
        # Assumes there is exactly one lowest floor
        lowest_floor = (floors_below - floors_above).pop()

        self.floor_to_num = {}
        self.floor_to_num[lowest_floor] = 1

        # Build map from floor below to floor above for easy traversal upwards
        floor_below_to_floor_above = {v: k for k, v in floor_above_to_floor_below.items()}

        current_floor = lowest_floor
        current_num = 1
        # Traverse upwards, assigning increasing numbers
        while current_floor in floor_below_to_floor_above:
            current_floor = floor_below_to_floor_above[current_floor]
            current_num += 1
            self.floor_to_num[current_floor] = current_num

        # Optional: Sanity check that all floors were mapped
        # assert len(self.floor_to_num) == len(all_floors)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to serve all unserved passengers.
        """
        state = node.state  # Current world state (frozenset of facts)

        # Find current lift floor
        current_lift_f = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at':
                current_lift_f = parts[1]
                break
        # Assumes a 'lift-at' fact is always present in a valid state

        total_cost = 0  # Initialize action cost counter.

        # Iterate through all passengers whose destinations are known
        for passenger, destin_f in self.destin_map.items():
            # Check if passenger is already served
            if f'(served {passenger})' in state:
                continue # This passenger is done

            # Check if passenger is boarded
            if f'(boarded {passenger})' in state:
                # Passenger is in the lift, needs to go to destination and depart
                current_lift_f_num = self.floor_to_num[current_lift_f]
                destin_f_num = self.floor_to_num[destin_f]

                # Cost = moves to destination + depart action
                total_cost += abs(current_lift_f_num - destin_f_num) + 1
            else:
                # Passenger is waiting at their origin floor
                # Find origin floor from the current state
                origin_f = None
                for fact in state:
                    parts = get_parts(fact)
                    if parts[0] == 'origin' and parts[1] == passenger:
                        origin_f = parts[2]
                        break
                # Assumes that if a passenger is not served and not boarded,
                # they must have an 'origin' fact in the state.

                current_lift_f_num = self.floor_to_num[current_lift_f]
                origin_f_num = self.floor_to_num[origin_f]
                destin_f_num = self.floor_to_num[destin_f]

                # Cost = moves to origin + board action + moves to destination + depart action
                total_cost += abs(current_lift_f_num - origin_f_num) + 1 + abs(origin_f_num - destin_f_num) + 1

        return total_cost
