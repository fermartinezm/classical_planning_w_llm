from fnmatch import fnmatch
# Assuming Heuristic base class is available in a module named heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts represented as strings
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at ball1 room1)" -> ["at", "ball1", "room1"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Wildcards (*) are allowed in the pattern arguments.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (e.g., "at", "*", "room1").
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern argument (with wildcard support)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers.
    It sums the minimum actions required per unserved passenger (board + depart or just depart)
    and adds an estimate of the movement cost for the lift to visit all necessary floors.

    # Assumptions
    - Floors are ordered numerically based on their names (e.g., f1 < f2 < ...).
    - Each unserved passenger requires at least one board and one depart action (if waiting)
      or one depart action (if boarded).
    - The movement cost is estimated by the minimum travel distance to cover the range
      of floors that must be visited for pickups and dropoffs.

    # Heuristic Initialization
    - Extract the floor ordering from `above` facts to create a floor-to-index mapping.
    - Extract destination floors for all passengers from `destin` facts.
    - Identify all passenger names involved in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the lift from the state using the `lift-at` predicate.
    2. Identify the state of each passenger from the current state: waiting at origin (`origin`), boarded (`boarded`), or served (`served`).
    3. Collect all passengers who are not yet served (i.e., not currently in the `served` predicate). If there are no unserved passengers, the current state is a goal state, and the heuristic is 0.
    4. Calculate the minimum action cost:
       - Initialize `action_cost` to 0.
       - For each unserved passenger waiting at their origin, add 2 to the cost (representing the minimum `board` and `depart` actions required for that passenger).
       - For each unserved passenger who is currently boarded, add 1 to the cost (representing the minimum `depart` action required for that passenger).
    5. Identify the set of floors the lift must visit to serve the unserved passengers:
       - Initialize an empty set `required_floors`.
       - For each unserved passenger waiting at their origin, add their origin floor and their destination floor (looked up from the precomputed `passenger_to_destin` mapping) to the `required_floors` set.
       - For each unserved passenger who is currently boarded, add their destination floor (looked up from the precomputed `passenger_to_destin` mapping) to the `required_floors` set.
    6. If the set of `required_floors` is empty (which should only happen if there are no unserved passengers, already handled in step 3), the movement cost is 0.
    7. If the set of `required_floors` is not empty:
       - Map the current lift floor and all floors in `required_floors` to their numerical indices using the precomputed `floor_to_index` mapping. Handle potential errors if a floor is not found in the mapping (e.g., return infinity).
       - Find the minimum and maximum index among the required floors (`min_required_index` and `max_required_index`).
       - Estimate the movement cost as the minimum of two simple travel scenarios:
         - Scenario A: Travel from the current floor index (`current_index`) down to the minimum required floor index (`min_required_index`), then sweep upwards to the maximum required floor index (`max_required_index`). Cost = `abs(current_index - min_required_index) + (max_required_index - min_required_index)`.
         - Scenario B: Travel from the current floor index (`current_index`) up to the maximum required floor index (`max_required_index`), then sweep downwards to the minimum required floor index (`min_required_index`). Cost = `abs(current_index - max_required_index) + (max_required_index - min_required_index)`.
         The movement cost estimate is the minimum of the costs from Scenario A and Scenario B.
    8. The total heuristic value is the sum of the calculated `action_cost` and `movement_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information from the task.
        This includes the floor ordering and passenger destinations.
        """
        # Assuming task object has 'goals' and 'static' attributes
        self.goals = task.goals
        self.static = task.static

        # Extract floor ordering and create floor-to-index mapping
        self.floor_to_index = {}
        floor_names = set()
        # Collect all floor names mentioned in static facts (above, destin, origin)
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "above":
                # (above f1 f2) means f2 is above f1
                floor_names.add(parts[1])
                floor_names.add(parts[2])
            elif parts[0] in ["destin", "origin"]:
                 # Floor is the third part of (predicate passenger floor)
                 if len(parts) > 2:
                    floor_names.add(parts[2])

        # Sort floor names based on the number suffix (assuming format f<number>)
        # This is a domain-specific assumption based on the provided examples.
        try:
            # Sort by the integer part after 'f'
            sorted_floors = sorted(list(floor_names), key=lambda f: int(f[1:]))
        except (ValueError, IndexError):
             # Fallback if floor names are not in the expected f<number> format
             # A more general approach would involve topological sort based on 'above' relations.
             # For this domain, f<number> is typical. If it fails, alphabetical sort is a simple fallback.
             # print("Warning: Floor names not in f<number> format. Using alphabetical sort.")
             sorted_floors = sorted(list(floor_names))

        # Create the mapping from floor name to a numerical index (1-based)
        for i, floor in enumerate(sorted_floors):
            self.floor_to_index[floor] = i + 1

        # Extract passenger destinations and collect all passenger names involved in the problem
        self.passenger_to_destin = {}
        self.all_passengers = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "destin":
                # (destin passenger floor)
                if len(parts) > 2:
                    passenger, floor = parts[1], parts[2]
                    self.passenger_to_destin[passenger] = floor
                    self.all_passengers.add(passenger)
            elif parts[0] == "origin":
                 # (origin passenger floor) - Passengers can also be introduced via origin in static
                 if len(parts) > 2:
                    passenger = parts[1]
                    self.all_passengers.add(passenger)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions from the current state.

        Args:
            node: The current state node in the search tree. Assumes node.state is a frozenset of fact strings.

        Returns:
            An estimated cost (integer) to reach a goal state, or float('inf') if the state is likely unsolvable
            or contains unmappable floors.
        """
        state = node.state

        # 1. Identify current lift location
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break
        # Assuming lift-at is always present in a valid state. If not, problem with state representation.
        if current_lift_floor is None:
             # print("Error: Lift location not found in state.")
             return float('inf') # Should not happen in valid states

        # 2. Identify passenger states from the current state
        waiting_passenger_origins = {} # passenger -> origin_floor
        boarded_passengers = set()
        served_passengers = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "origin":
                # (origin passenger floor)
                if len(parts) > 2:
                    passenger, floor = parts[1], parts[2]
                    waiting_passenger_origins[passenger] = floor
            elif parts[0] == "boarded":
                # (boarded passenger)
                if len(parts) > 1:
                    passenger = parts[1]
                    boarded_passengers.add(passenger)
            elif parts[0] == "served":
                # (served passenger)
                if len(parts) > 1:
                    passenger = parts[1]
                    served_passengers.add(passenger)

        # 3. Collect unserved passengers
        # Assuming goal is (served p) for all p in self.all_passengers defined in static
        unserved_passengers = self.all_passengers - served_passengers

        if not unserved_passengers:
            return 0 # Goal state reached (all relevant passengers served)

        # 4. Calculate action cost and identify required floors
        action_cost = 0
        required_floors = set() # Collect floors the lift must visit

        for passenger in unserved_passengers:
            # Check if the passenger is currently waiting at an origin
            if passenger in waiting_passenger_origins:
                # Needs board (1) + depart (1) actions
                action_cost += 2
                origin_floor = waiting_passenger_origins[passenger]
                destin_floor = self.passenger_to_destin.get(passenger)
                if destin_floor: # Ensure destination exists (should for unserved passengers in a solvable problem)
                    required_floors.add(origin_floor)
                    required_floors.add(destin_floor)
                else:
                     # Unserved passenger with no destination in static - problematic problem definition
                     # print(f"Warning: Unserved passenger {passenger} has no destination defined in static facts.")
                     return float('inf') # Indicate potential unsolvability

            # Check if the passenger is currently boarded
            elif passenger in boarded_passengers:
                # Needs depart (1) action
                action_cost += 1
                destin_floor = self.passenger_to_destin.get(passenger)
                if destin_floor: # Ensure destination exists
                    required_floors.add(destin_floor)
                else:
                     # Unserved boarded passenger with no destination in static - problematic problem definition
                     # print(f"Warning: Unserved boarded passenger {passenger} has no destination defined in static facts.")
                     return float('inf') # Indicate potential unsolvability

            # Note: Passengers who are unserved but neither waiting nor boarded
            # (e.g., maybe dropped off at the wrong floor, though domain doesn't allow this)
            # are implicitly handled: they contribute to unserved_passengers,
            # but don't add to action_cost here. Their destination might be in required_floors
            # if they were previously waiting/boarded. This seems consistent with the domain.


        # 6. Calculate movement cost
        movement_cost = 0
        # Movement is only needed if there are floors to visit
        if required_floors:
            # Ensure current lift floor is in the mapping (should be if parsing was successful)
            if current_lift_floor not in self.floor_to_index:
                 # This indicates an issue with floor parsing or state
                 # print(f"Error: Current lift floor '{current_lift_floor}' not found in floor index map.")
                 return float('inf') # Indicate unsolvable or problematic state

            i_current = self.floor_to_index[current_lift_floor]

            # Map required floors to indices, skipping any unknown floors
            # If a required floor is not in the map, it's a problem.
            required_indices = set()
            for f in required_floors:
                if f in self.floor_to_index:
                    required_indices.add(self.floor_to_index[f])
                else:
                    # print(f"Error: Required floor '{f}' not found in floor index map.")
                    return float('inf') # Indicate unsolvable or problematic state


            # Movement is only needed if there are required floors with known indices
            if required_indices:
                min_i = min(required_indices)
                max_i = max(required_indices)

                # Estimate moves to cover the range [min_i, max_i] starting from i_current
                # This is the distance to one end of the range plus the size of the range.
                # Option 1: Go towards min_i first, then sweep up to max_i
                cost1 = abs(i_current - min_i) + (max_i - min_i)
                # Option 2: Go towards max_i first, then sweep down to min_i
                cost2 = abs(i_current - max_i) + (max_i - min_i)

                movement_cost = min(cost1, cost2)
            # else: This case implies required_floors was not empty, but none of its floors
            # could be mapped to indices, which is handled by the loop above returning inf.
            # So, if we reach here and required_indices is empty, required_floors must have been empty.


        # 8. Total heuristic
        return action_cost + movement_cost

