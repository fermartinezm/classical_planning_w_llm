from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    passengers by summing the estimated cost for each unserved passenger
    independently. The cost for a passenger includes the lift travel to their
    origin (if waiting), the board action, the lift travel from their origin
    to their destination, and the depart action. This heuristic is non-admissible
    as it ignores synergies (e.g., multiple passengers sharing lift travel).

    # Assumptions
    - The domain uses standard miconic actions: lift movement (up/down), board, and depart.
    - Floor names like f1, f2, f3, etc., represent ordered floors, and their
      lexicographical order corresponds to their physical level (f1 is the lowest,
      f2 is above f1, etc., or vice versa, as long as the ordering is consistent).
      This heuristic assumes lexicographical order corresponds to increasing level.
    - The cost of each action (move between adjacent floors, board, depart) is 1.

    # Heuristic Initialization
    - Parses static facts to identify all floor objects and determine their
      relative levels based on lexicographical sorting of names.
    - Stores the origin and destination floor for each passenger from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Get the current location of the lift from the state.
    2. Initialize the total heuristic cost to 0.
    3. Identify the set of passengers who are not yet served based on the state.
    4. For each unserved passenger:
       a. Check if the passenger is currently boarded in the lift.
       b. If the passenger is boarded:
          - The estimated cost for this passenger is the number of moves needed
            for the lift to travel from its current floor to the passenger's
            destination floor, plus 1 action for the 'depart' operation.
          - Cost = `abs(level(current_lift_floor) - level(destination_floor)) + 1`.
       c. If the passenger is not boarded (meaning they are waiting at their origin):
          - The estimated cost for this passenger is the number of moves needed
            for the lift to travel from its current floor to the passenger's
            origin floor, plus 1 action for 'board', plus the number of moves
            needed for the lift to travel from the origin floor to the
            destination floor, plus 1 action for 'depart'.
          - Cost = `abs(level(current_lift_floor) - level(origin_floor)) + 1 + abs(level(origin_floor) - level(destination_floor)) + 1`.
       d. Add the estimated cost for this passenger to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels, passenger origins,
        and destinations from the task's static facts.
        """
        # We don't strictly need to store goals or static facts directly
        # but we process static facts to build necessary maps.
        # self.goals = task.goals

        # Extract floors and create floor-to-level mapping
        floor_names = set()
        self.origin_map = {}
        self.destin_map = {}

        # Iterate through static facts to find floors, origins, and destinations
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == "above" and len(parts) == 3:
                # (above f1 f2) implies f1 and f2 are floors
                floor_names.add(parts[1])
                floor_names.add(parts[2])
            elif predicate == "origin" and len(parts) == 3:
                # (origin p1 f1)
                passenger, floor = parts[1], parts[2]
                self.origin_map[passenger] = floor
                floor_names.add(floor) # Ensure origin floor is included
            elif predicate == "destin" and len(parts) == 3:
                # (destin p1 f2)
                passenger, floor = parts[1], parts[2]
                self.destin_map[passenger] = floor
                floor_names.add(floor) # Ensure destination floor is included

        # Sort floor names lexicographically to assign levels.
        # This assumes floor names like f1, f2, f10 sort correctly.
        # E.g., ['f1', 'f10', 'f2'] sorted is ['f1', 'f10', 'f2'].
        # A more robust approach would parse numerical suffixes if names are 'f<num>'.
        # However, lexicographical sort works for f1, f2, f3...
        # If names are f1, f2, ..., f10, f11, ..., f20, lexicographical sort is correct.
        sorted_floor_names = sorted(list(floor_names))
        self.floor_to_level = {floor: i + 1 for i, floor in enumerate(sorted_floor_names)}

        # Get all passenger names from origin/destin maps
        self.passengers = set(self.origin_map.keys()).union(set(self.destin_map.keys()))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find current lift location
        lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at" and len(parts) == 2:
                lift_floor = parts[1]
                break

        # If lift location is not found, the state is likely malformed or terminal
        # in an unexpected way. Return a high value or handle appropriately.
        # Assuming valid states have lift-at, get its level.
        lift_level = self.floor_to_level.get(lift_floor, 0) # Default to 0 if floor not in map (error case)

        total_cost = 0

        # Identify passengers who are served or boarded in the current state
        served_passengers = set()
        boarded_passengers = set()
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == "served" and len(parts) == 2:
                served_passengers.add(parts[1])
            elif parts[0] == "boarded" and len(parts) == 2:
                boarded_passengers.add(parts[1])

        # Calculate cost for each unserved passenger
        for passenger in self.passengers:
            if passenger in served_passengers:
                continue # Passenger is already served, cost is 0 for this passenger

            origin_floor = self.origin_map.get(passenger)
            destin_floor = self.destin_map.get(passenger)

            # Basic validation: passenger must have origin and destination defined
            if origin_floor is None or destin_floor is None:
                 # Cannot calculate heuristic for this passenger, skip or handle error
                 # This shouldn't happen in valid miconic problems
                 continue

            origin_level = self.floor_to_level.get(origin_floor, 0)
            destin_level = self.floor_to_level.get(destin_floor, 0)

            if passenger in boarded_passengers:
                # Passenger is in the lift, needs to go to destination and depart
                # Cost = Moves from current lift floor to destination + Depart action
                cost = abs(lift_level - destin_level) + 1
                total_cost += cost
            else:
                # Passenger is waiting at origin, needs pickup and then trip to destination
                # Cost = Moves from current lift floor to origin + Board action +
                #        Moves from origin to destination + Depart action
                cost = abs(lift_level - origin_level) + 1 + abs(origin_level - destin_level) + 1
                total_cost += cost

        return total_cost
