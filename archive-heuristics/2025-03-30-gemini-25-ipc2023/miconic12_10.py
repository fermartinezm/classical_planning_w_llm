from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

# Helper function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(predicate obj1 obj2)" -> ["predicate", "obj1", "obj2"]
    return fact.strip()[1:-1].split()

# Helper function to match a fact against a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern. Wildcards (*) are allowed.

    Args:
        fact (str): The PDDL fact string (e.g., "(at obj loc)").
        *args: A sequence of strings representing the pattern (e.g., "at", "*", "loc*").

    Returns:
        bool: True if the fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    # Ensure the number of parts in the fact matches the pattern length
    if len(parts) != len(args):
        return False
    # Check each part against the corresponding pattern argument using fnmatch for wildcard support
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic (elevator) domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It calculates the cost for each passenger individually based on their current state
    (waiting, boarded, or served) and sums these costs. The cost for a passenger
    includes the necessary board/depart actions and the estimated lift movement
    required to pick them up and drop them off from the lift's current position.

    # Assumptions
    - Floors are linearly ordered, and floor names like 'f1', 'f2' indicate their level (1, 2, ...).
      The heuristic relies on extracting this number to calculate distances between floors.
    - The `(above f_i f_j)` predicate, if present, is assumed to be consistent with this numbering
      (e.g., implies floor i is below floor j), but the heuristic primarily uses the floor number
      extracted from the name, not the `above` predicate itself for distance calculation.
    - The heuristic calculates costs per passenger independently and sums them. This means
      lift movements might be overestimated, as potential shared travel between passengers
      (e.g., picking up multiple passengers before moving, dropping off multiple passengers)
      is not explicitly modeled by simply summing individual travel segments starting from
      the current lift position. This overestimation is acceptable as the heuristic
      does not need to be admissible and aims to guide a greedy best-first search effectively.
    - The primary goal is to have all passengers in the 'served' state, as defined by the
      `(served ?p)` predicate.

    # Heuristic Initialization
    - Extracts and stores the destination floor for each passenger from the static facts (`destin`).
    - Identifies all unique passengers based on the `destin` facts found in the static information.
    - Pre-calculates the integer level for each known floor name encountered in static facts
      and the initial state, assuming the 'fN' naming convention (e.g., 'f10' corresponds to level 10).
      This allows for efficient distance calculation later.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify the current floor of the lift (`f_lift`) by finding the `(lift-at ?f)` predicate in the current state. If not found, the state is considered invalid or problematic, returning infinity.
    2.  Determine the current status of each passenger by checking for `(origin ?p ?f)`, `(boarded ?p)`, and `(served ?p)` predicates in the state.
    3.  Initialize the total heuristic cost `h = 0`.
    4.  Iterate through each passenger `p` identified during initialization:
        a. If the passenger is already served (`(served p)` is true), their contribution to the cost is 0.
        b. If the passenger is currently boarded (`(boarded p)` is true):
           - Retrieve their destination floor `d_p` (stored during initialization).
           - Calculate the distance from the current lift floor to the destination: `dist = distance(f_lift, d_p)`.
           - The cost estimated for this passenger is `dist + 1` (representing lift movement actions + 1 `depart` action).
           - Add this cost to the total heuristic value `h`.
        c. If the passenger is waiting at their origin (`(origin p o_p)` is true):
           - Retrieve their origin floor `o_p` (from the current state) and destination `d_p` (from initialization).
           - Calculate the distance from the lift's current floor to the passenger's origin: `dist1 = distance(f_lift, o_p)`.
           - Calculate the distance from the origin floor to the destination floor: `dist2 = distance(o_p, d_p)`.
           - The cost estimated for this passenger is `dist1 + 1 + dist2 + 1` (representing moves to origin + 1 `board` action + moves to destination + 1 `depart` action).
           - Add this cost to the total heuristic value `h`.
        d. If a passenger is unserved but neither boarded nor at their origin, this indicates an unexpected state. A warning is logged, and a minimal cost (1) is added.
    5.  The final heuristic value is the total sum `h`.
    6.  The distance between two floors `f1` and `f2` is calculated as the absolute difference of their levels: `abs(level(f1) - level(f2))`. Floor levels are determined by parsing the number from the floor name (e.g., `f10` -> level 10). If a floor level cannot be determined, an error occurs, and the heuristic returns infinity for that state.
    7.  If all passengers are determined to be served (`num_unserved == 0`), the heuristic checks if the state satisfies the task's goal conditions. If yes, it returns 0. If not (e.g., a terminal state where passengers are served but other goals aren't met), it returns 1 to indicate it's not the true goal. For any non-goal state with unserved passengers, the heuristic returns a positive value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by processing static information from the task.
        Stores passenger destinations, identifies all passengers, and pre-calculates floor levels.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store passenger destinations and identify all passengers
        self.destinations = {}
        self.passengers = set()
        potential_floors = set() # Keep track of all floor names encountered

        # Process static facts to find destinations, passengers, and floors
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) > 1:
                predicate = parts[0]
                if predicate == "destin":
                    # Fact format: (destin passenger floor)
                    _, p, f = parts
                    self.destinations[p] = f
                    self.passengers.add(p)
                    potential_floors.add(f)
                elif predicate == "above":
                     # Fact format: (above floor1 floor2) - add both floors
                     _, f1, f2 = parts
                     potential_floors.add(f1)
                     potential_floors.add(f2)

        # Add floors mentioned in the initial state as well
        for fact in task.initial_state:
             parts = get_parts(fact)
             if len(parts) > 1:
                 predicate = parts[0]
                 if predicate == 'lift-at':
                     # Fact format: (lift-at floor)
                     potential_floors.add(parts[1])
                 elif predicate == 'origin':
                     # Fact format: (origin passenger floor)
                     potential_floors.add(parts[2]) # Add origin floor

        # Determine floor levels assuming format fN (e.g., f1, f2, f10)
        self.floor_levels = {}
        for floor in potential_floors:
            level = self._parse_floor_level(floor)
            if level is not None:
                self.floor_levels[floor] = level
            else:
                # Log a warning if a floor name doesn't match the expected pattern.
                # The _get_floor_level method will handle errors robustly if the level is needed later.
                print(f"Warning: Floor name '{floor}' does not match expected pattern 'fN'. Level could not be determined during initialization.")


    def _parse_floor_level(self, floor_name):
        """
        Parses the integer level from a floor name string like 'fN'.
        Returns the integer level if the pattern matches, otherwise returns None.
        """
        match_num = re.match(r'f(\d+)', floor_name)
        if match_num:
            return int(match_num.group(1))
        return None

    def _get_floor_level(self, floor_name):
        """
        Returns the integer level of a given floor name.
        It uses pre-calculated levels if available, otherwise attempts to parse the level dynamically.
        Raises ValueError if the level cannot be determined.
        """
        if floor_name in self.floor_levels:
            return self.floor_levels[floor_name]
        else:
            # Attempt to parse the level now if it wasn't encountered during initialization
            level = self._parse_floor_level(floor_name)
            if level is not None:
                self.floor_levels[floor_name] = level # Cache the newly parsed level
                return level
            else:
                # If the level is still unknown after trying to parse, raise an error.
                # Distance calculation is impossible without knowing the floor level.
                raise ValueError(f"Level for floor '{floor_name}' is unknown and could not be parsed. Cannot calculate distance.")

    def _distance(self, floor1, floor2):
        """
        Calculates the distance (number of required move actions) between two floors
        based on their levels.
        """
        # Distance is 0 if the floors are the same
        if floor1 == floor2:
            return 0
        # Get levels for both floors (raises ValueError if unknown)
        level1 = self._get_floor_level(floor1)
        level2 = self._get_floor_level(floor2)
        # Distance is the absolute difference in levels
        return abs(level1 - level2)

    def __call__(self, node):
        """
        Calculate the heuristic value for the given state node.
        Estimates the total number of actions (moves, boards, departs) needed to serve all passengers.
        """
        state = node.state
        total_cost = 0

        # 1. Find the current location of the lift
        lift_floor = None
        for fact in state:
            # Check if the fact matches the pattern (lift-at *)
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break # Found the lift location, no need to check further facts

        if lift_floor is None:
            # If lift location predicate is missing, the state might be invalid or terminal
            # in a way not expected by the standard domain model.
            print("Warning: Lift location predicate 'lift-at' not found in state.")
            return float('inf') # Return infinity to strongly discourage exploring this state

        # 2. Determine the current state of all passengers
        passenger_origins = {} # Maps waiting passengers to their origin floor
        boarded_passengers = set() # Set of passengers currently in the lift
        served_passengers = set() # Set of passengers who have reached their destination

        for fact in state:
            parts = get_parts(fact)
            if len(parts) > 1:
                predicate = parts[0]
                # Check if the second part is a passenger we are tracking
                passenger = parts[1]
                if passenger in self.passengers:
                    if predicate == "origin":
                        # Format: (origin passenger floor)
                        passenger_origins[passenger] = parts[2]
                    elif predicate == "boarded":
                        # Format: (boarded passenger)
                        boarded_passengers.add(passenger)
                    elif predicate == "served":
                        # Format: (served passenger)
                        served_passengers.add(passenger)

        # 3. Calculate the estimated cost for each unserved passenger
        num_unserved = 0
        for p in self.passengers:
            # Skip passengers who are already served
            if p in served_passengers:
                continue

            num_unserved += 1 # Count this passenger as needing service
            dest_floor = self.destinations.get(p) # Get the passenger's destination floor

            if dest_floor is None:
                 # This should not happen if initialization correctly processed all passengers/destinations
                 print(f"Error: Destination for passenger {p} not found during heuristic calculation.")
                 return float('inf') # Cannot compute heuristic without destination information

            try:
                # Calculate cost based on whether the passenger is boarded or waiting
                if p in boarded_passengers:
                    # Passenger is boarded: cost = move actions to destination + 1 depart action
                    cost_p = self._distance(lift_floor, dest_floor) + 1
                    total_cost += cost_p
                elif p in passenger_origins:
                    # Passenger is waiting at origin: cost = moves to origin + 1 board + moves to destination + 1 depart
                    origin_floor = passenger_origins[p]
                    dist_to_origin = self._distance(lift_floor, origin_floor)
                    dist_origin_to_dest = self._distance(origin_floor, dest_floor)
                    cost_p = dist_to_origin + 1 + dist_origin_to_dest + 1
                    total_cost += cost_p
                else:
                    # Unserved passenger 'p' is not boarded and not at origin.
                    # This state configuration is unexpected in the standard Miconic model.
                    print(f"Warning: Unserved passenger {p} is neither boarded nor at origin. Assigning base cost 1. State: {state}")
                    # Assign a minimal cost of 1, assuming at least one action (e.g., depart) is needed.
                    total_cost += 1
            except ValueError as e:
                # This exception occurs if _distance fails (e.g., unknown floor level)
                print(f"Error calculating distance for passenger {p}: {e}")
                # Return infinity for this state, as the heuristic cannot be reliably computed.
                return float('inf')

        # 4. Final adjustments for goal states
        if num_unserved == 0:
            # If all passengers we track are served, check if this state matches the actual goal conditions
            is_goal = self.goals <= state
            # Return 0 if it's a true goal state. Return 1 if passengers are served but other goal conditions might be missing.
            return 0 if is_goal else 1

        # For any non-goal state with unserved passengers, the calculated cost should be positive.
        # The calculation ensures cost >= 1 if num_unserved > 0 because each unserved passenger adds at least 1 (for depart).
        return total_cost
