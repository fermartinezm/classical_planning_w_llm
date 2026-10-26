import math  # Using math for clarity, standard abs() also works.

# Assuming the Heuristic base class is available at this path
# Make sure this import path matches your project structure.
from heuristics.heuristic_base import Heuristic

# Helper function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extracts the components of a PDDL fact string.
    Removes parentheses and splits by space.
    Example: "(lift-at f1)" -> ["lift-at", "f1"]
    """
    # Ensure fact is a string and has content
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove parentheses and split
    return fact[1:-1].split()

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic (elevator) domain.

    # Summary
    This heuristic estimates the remaining cost (number of actions) to serve all
    passengers by summing the estimated costs for each unserved passenger individually.
    The cost for a passenger includes the lift movement required to pick them up
    (if waiting at their origin) and drop them off at their destination, plus the
    actions for boarding (if waiting) and departing.

    # Assumptions
    - Floor names are consistently formatted as 'fN' where N is an integer
      representing the floor level (e.g., f1 is level 1, f10 is level 10).
    - The 'above' predicate in the domain definition is consistent with this
      numbering, implying floors have a linear order.
    - The lift moves between adjacent floors using 'up' and 'down' actions,
      each costing 1 action. The travel distance between floor fI and fJ is
      therefore calculated as abs(I - J).
    - The heuristic calculates the cost for each passenger independently, assuming
      the lift starts from its current position for each passenger's task. It
      ignores potential optimizations from carrying multiple passengers
      simultaneously or optimizing the lift's route. This makes the heuristic
      non-admissible (it can overestimate the true cost) but aims to be
      informative for guiding a greedy best-first search.

    # Heuristic Initialization
    - The constructor (`__init__`) processes the static facts of the task.
    - It extracts and stores the destination floor for each passenger from the
      static 'destin' predicates in a dictionary `self.destinations`.
    - It identifies the set of all passengers (`self.passengers`) involved in
      the problem based on these 'destin' facts. This assumes all passengers
      that need to be served have a corresponding 'destin' fact.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Get the current state (`node.state`).
    2.  Check if the current state is a goal state using `self.task.goal_reached(state)`.
        If yes, the heuristic value is 0.
    3.  Identify the current location of the lift (`lift_floor`) by finding the
        `(lift-at ?f)` fact in the state. Determine the lift's level (`lift_level`)
        using the `_get_floor_level` helper. If the lift location is missing in a
        non-goal state, return infinity (error/invalid state).
    4.  Determine the status of each passenger:
        - Identify passengers currently served (`served_passengers`).
        - Identify passengers currently inside the lift (`boarded_passengers`).
        - Identify passengers waiting at their origin (`waiting_passengers`), storing
          their origin floor.
    5.  Initialize `total_heuristic_value = 0`.
    6.  Iterate through all passengers `p` identified during initialization (`self.passengers`):
        a. If `p` is in `served_passengers`, their contribution is 0. Continue.
        b. Retrieve the destination floor `f_dest` for `p` from `self.destinations`.
           Calculate the destination level `level_dest`. Handle potential errors if
           destination or floor levels are invalid (return infinity).
        c. If `p` is in `boarded_passengers`:
           - Estimate cost = `abs(lift_level - level_dest)` (actions to move lift to destination)
                           + 1 (action to depart).
           - Add this cost to `total_heuristic_value`.
        d. If `p` is in `waiting_passengers`:
           - Get the origin floor `f_orig` for `p`. Calculate `level_orig`.
           - Estimate cost = `abs(lift_level - level_orig)` (actions to move lift to origin)
                           + 1 (action to board)
                           + `abs(level_orig - level_dest)` (actions to move lift from origin to destination)
                           + 1 (action to depart).
           - Add this cost to `total_heuristic_value`.
        e. If passenger `p` needs service (is in `self.passengers` but not `served_passengers`)
           but is neither boarded nor waiting, this indicates an inconsistent or
           unexpected state. Return infinity.
    7.  After iterating through all passengers, return `total_heuristic_value`. If the
        calculated value is 0 but the state is not a goal state (checked at the start),
        return 1 to ensure the search progresses (this case should ideally not occur
        in well-defined problems).
    """

    def __init__(self, task):
        """
        Initializes the heuristic.
        - Stores the task for goal checking.
        - Extracts destination information for all passengers from static facts.
        - Identifies all passengers based on 'destin' facts.
        """
        super().__init__(task)
        self.task = task # Store task for goal checking later
        static_facts = task.static

        # Store destination floor for each passenger {passenger_name: floor_name}
        self.destinations = {}
        # Store all passengers mentioned in destin predicates
        self.passengers = set()

        for fact in static_facts:
            parts = get_parts(fact)
            # Ensure fact is correctly parsed and represents a 'destin' predicate
            if parts and parts[0] == 'destin' and len(parts) == 3:
                passenger = parts[1]
                floor = parts[2]
                self.destinations[passenger] = floor
                self.passengers.add(passenger)
            # Could potentially parse 'floor' or 'passenger' type declarations
            # if needed, but relying on 'destin' covers passengers needing service.

        # Basic validation: Check if all passengers in goals have destinations defined.
        # This assumes goals are primarily '(served p)'.
        for goal_fact in self.task.goals:
            parts = get_parts(goal_fact)
            if parts and parts[0] == 'served' and len(parts) == 2:
                passenger = parts[1]
                if passenger not in self.destinations:
                    # This might indicate an issue with the PDDL instance.
                    # For robustness, we could add passengers from goals if missing,
                    # but without a destination, calculation is impossible.
                    # We will rely on the main loop to handle missing destinations.
                    pass


    def _get_floor_level(self, floor_name):
        """
        Extracts the integer level from a floor name string like 'f10'.
        Returns the integer level (e.g., 10).
        Raises ValueError if the floor name format is invalid.
        """
        if not isinstance(floor_name, str) or not floor_name.startswith('f') or len(floor_name) < 2:
            raise ValueError(f"Invalid floor name format: '{floor_name}'. Expected 'f' followed by digits.")
        try:
            # Extract substring after 'f' and convert to integer
            return int(floor_name[1:])
        except ValueError:
            # Handle cases like 'f' or 'f<non-digit>'
            raise ValueError(f"Invalid floor name format: '{floor_name}'. Expected 'f' followed by digits.")


    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        Estimates the total actions needed to serve all remaining passengers.
        """
        state = node.state

        # Goal check: If the current state is the goal, heuristic value is 0.
        if self.task.goal_reached(state):
            return 0

        heuristic_value = 0

        # Find current lift location and level
        lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'lift-at' and len(parts) == 2:
                lift_floor = parts[1]
                break

        # If lift location is not found in a non-goal state, it's problematic.
        if lift_floor is None:
             # Returning infinity signals this state is likely invalid or problematic.
             return float('inf')

        try:
            lift_level = self._get_floor_level(lift_floor)
        except ValueError as e:
             # Invalid lift floor name format, treat as error state.
             # print(f"Heuristic error: {e}") # Optional logging
             return float('inf')

        # Determine the status of all passengers in the current state
        served_passengers = set()
        boarded_passengers = set()
        waiting_passengers = {} # Map: passenger_name -> origin_floor_name

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts

            predicate = parts[0]
            if predicate == 'served' and len(parts) == 2:
                served_passengers.add(parts[1])
            elif predicate == 'boarded' and len(parts) == 2:
                boarded_passengers.add(parts[1])
            elif predicate == 'origin' and len(parts) == 3:
                # Passenger parts[1] is waiting at floor parts[2]
                waiting_passengers[parts[1]] = parts[2]

        # Calculate estimated cost for each passenger needing service
        for p in self.passengers:
            # Skip passengers already served
            if p in served_passengers:
                continue

            # Get destination floor and level for passenger p
            if p not in self.destinations:
                 # If a passenger needs service (implied by being in self.passengers
                 # and not served) but lacks a destination, the problem is ill-defined.
                 # print(f"Heuristic error: Passenger {p} needs service but has no destination defined.")
                 return float('inf') # Indicate error/inconsistency

            try:
                dest_floor = self.destinations[p]
                dest_level = self._get_floor_level(dest_floor)

                # Case 1: Passenger is boarded
                if p in boarded_passengers:
                    # Cost = move lift to destination + depart action
                    move_cost = abs(lift_level - dest_level)
                    depart_cost = 1
                    heuristic_value += move_cost + depart_cost

                # Case 2: Passenger is waiting at origin
                elif p in waiting_passengers:
                    origin_floor = waiting_passengers[p]
                    origin_level = self._get_floor_level(origin_floor)

                    # Cost = move lift to origin + board + move lift to destination + depart
                    move_to_origin_cost = abs(lift_level - origin_level)
                    board_cost = 1
                    move_origin_to_dest_cost = abs(origin_level - dest_level)
                    depart_cost = 1
                    heuristic_value += move_to_origin_cost + board_cost + move_origin_to_dest_cost + depart_cost

                # Case 3: Passenger needs service but is not boarded and not waiting
                else:
                    # This state should typically not be reachable for an unserved passenger
                    # in the standard Miconic domain. Indicates an inconsistency.
                    # print(f"Heuristic error: Passenger {p} in inconsistent state (needs service but not waiting or boarded).")
                    return float('inf')

            except ValueError as e:
                 # Invalid floor name encountered for origin or destination of passenger p.
                 # print(f"Heuristic error processing passenger {p}: {e}") # Optional logging
                 return float('inf')


        # Final safety check: If heuristic is 0 but goal not met, return 1.
        # This prevents the search from stopping prematurely if the heuristic incorrectly
        # evaluates a non-goal state as 0. This scenario is unlikely if the logic
        # correctly identifies all work needed for unserved passengers.
        if heuristic_value == 0 and not self.task.goal_reached(state):
             # This could occur if self.passengers is empty but the goal is non-trivial,
             # or if there's a flaw in state representation or goal definition.
             return 1

        return heuristic_value

