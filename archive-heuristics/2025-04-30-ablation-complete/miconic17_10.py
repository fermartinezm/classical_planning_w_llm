from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
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
    This heuristic estimates the number of actions needed to serve all passengers.
    It models the problem as a two-phase process: first, the lift picks up
    all waiting passengers; second, it drops off all boarded passengers.
    The heuristic sums the estimated movement costs for these two phases and
    the number of necessary board and depart actions.

    # Assumptions
    - Floors are ordered numerically based on their names (e.g., f1 < f2 < ...).
    - The lift can carry all passengers simultaneously.
    - The lift must visit all required pickup floors in the first phase and
      all required dropoff floors in the second phase.
    - The movement cost to visit a set of floors on a line starting from a
      given floor is estimated by traveling to the closer extreme of the
      required floor range and then sweeping across the entire range.

    # Heuristic Initialization
    - Extracts all floor objects and creates a mapping from floor name to
      a numerical index (1-based) based on the assumed numerical ordering.
    - Extracts the origin and destination floors for each passenger from
      the static facts, and identifies all passengers.

    # Step-By-Step Thinking for Computing Heuristic
    1. Find the current floor of the lift.
    2. Identify all unserved passengers by checking against the 'served' facts
       in the current state. If no unserved passengers, the heuristic is 0.
    3. Partition the unserved passengers into two groups:
       - Those waiting at their origin floor (not currently 'boarded').
       - Those already 'boarded'.
    4. Determine the set of floors the lift must visit in the first phase
       (pickup phase): These are the origin floors of all passengers waiting
       at their origin.
    5. Determine the set of floors the lift must visit in the second phase
       (dropoff phase): These are the destination floors of all passengers
       who are currently boarded (both those initially boarded and those
       picked up in phase 1).
    6. Calculate the estimated movement cost for Phase 1: Use the
       `calculate_movement_cost` helper function, starting from the current
       lift floor and targeting the set of Phase 1 required floors.
    7. Determine the estimated starting floor for Phase 2. This is the floor
       at the end of the estimated Phase 1 movement path. Based on the
       `calculate_movement_cost` logic, this is the extreme floor (min or max)
       of the Phase 1 required range that is *further* from the Phase 1 start
       floor.
    8. Calculate the estimated movement cost for Phase 2: Use the
       `calculate_movement_cost` helper function, starting from the estimated
       Phase 2 start floor and targeting the set of Phase 2 required floors.
    9. The total estimated movement cost is the sum of the costs for Phase 1
       and Phase 2.
    10. Calculate the action cost:
        - Each passenger waiting at their origin needs one 'board' action.
        - Each unserved passenger eventually needs one 'depart' action.
        - The total action cost is the number of passengers needing pickup
          plus the total number of unserved passengers.
    11. The final heuristic value is the sum of the total estimated movement
        cost and the total action cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger routes.
        """
        self.goals = task.goals
        self.static = task.static

        # 1. Extract and order floors
        # Find all floor objects by looking at objects mentioned in 'above' predicates
        floor_names_set = set()
        for fact in self.static:
            if match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3: # Ensure correct predicate structure
                    floor_names_set.add(parts[1])
                    floor_names_set.add(parts[2])

        # Sort floor names numerically (assuming format 'f' followed by digits)
        try:
            floor_objects = sorted(list(floor_names_set), key=lambda f: int(f[1:]))
        except (ValueError, IndexError):
             # Fallback if floor names are not in expected format, sort alphabetically
             floor_objects = sorted(list(floor_names_set))


        # Create floor name to index mapping (1-based index)
        self.floor_map = {floor_name: i + 1 for i, floor_name in enumerate(floor_objects)}
        self.index_to_floor = {i + 1: floor_name for i, floor_name in enumerate(floor_objects)}

        # 2. Extract passenger origins and destinations from static facts
        self.origins = {}
        self.destins = {}
        self.all_passengers = set()

        for fact in self.static:
            parts = get_parts(fact)
            if parts and len(parts) == 3:
                if parts[0] == 'origin':
                    p, f = parts[1], parts[2]
                    self.origins[p] = f
                    self.all_passengers.add(p)
                elif parts[0] == 'destin':
                    p, f = parts[1], parts[2]
                    self.destins[p] = f
                    self.all_passengers.add(p)

    def calculate_movement_cost(self, start_floor_idx, required_floors):
        """
        Estimates the minimum movement cost to visit a set of floors
        starting from a given floor index. Assumes linear floor structure.
        Cost is min distance to reach the range + distance to sweep the range.
        """
        # Filter out any required floors that weren't successfully mapped
        required_floors = {f for f in required_floors if f in self.floor_map}

        if not required_floors:
            return 0

        required_indices = {self.floor_map[f] for f in required_floors}

        min_req_idx = min(required_indices)
        max_req_idx = max(required_indices)

        dist_to_min = abs(start_floor_idx - min_req_idx)
        dist_to_max = abs(start_floor_idx - max_req_idx)
        range_dist = max_req_idx - min_req_idx

        # Cost to get to the closer end of the range + cost to sweep the range
        movement_cost = min(dist_to_min, dist_to_max) + range_dist
        return movement_cost

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                parts = get_parts(fact)
                if len(parts) == 2:
                    current_lift_floor = parts[1]
                    break

        if current_lift_floor is None or current_lift_floor not in self.floor_map:
             # Should not happen in a valid state, return a high cost
             return 1000000

        current_lift_floor_idx = self.floor_map[current_lift_floor]

        # Identify unserved passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*") and len(get_parts(fact)) == 2}
        unserved_passengers = self.all_passengers - served_passengers

        if not unserved_passengers:
            return 0 # Goal state

        # Partition unserved passengers
        passengers_to_pickup = set()
        passengers_to_dropoff = set()

        boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*") and len(get_parts(fact)) == 2}

        for p in unserved_passengers:
            if p not in boarded_passengers:
                 # Passenger is unserved and not boarded, assume they are at origin
                 passengers_to_pickup.add(p)
            else:
                 # Passenger is unserved and boarded
                 passengers_to_dropoff.add(p)

        # Determine required floors for each phase
        floors_to_visit_first_leg = {self.origins[p] for p in passengers_to_pickup if p in self.origins}
        floors_to_visit_second_leg = {self.destins[p] for p in passengers_to_pickup if p in self.destins} | \
                                     {self.destins[p] for p in passengers_to_dropoff if p in self.destins}

        # Calculate movement cost for Phase 1
        movement_cost_phase1 = self.calculate_movement_cost(current_lift_floor_idx, floors_to_visit_first_leg)

        # Determine estimated start floor for Phase 2
        start_floor_phase2_idx = current_lift_floor_idx
        if floors_to_visit_first_leg:
             # Filter out unmapped floors before getting indices
             f1_indices = {self.floor_map[f] for f in floors_to_visit_first_leg if f in self.floor_map}
             if f1_indices: # Ensure f1_indices is not empty
                 min_f1_idx = min(f1_indices)
                 max_f1_idx = max(f1_indices)
                 dist_to_min = abs(current_lift_floor_idx - min_f1_idx)
                 dist_to_max = abs(current_lift_floor_idx - max_f1_idx)

                 # The path ends at the extreme floor (min or max) that is *further*
                 # from the start floor, based on the sweep direction chosen by min(dist_to_min, dist_to_max).
                 # If dist_to_min <= dist_to_max, the path goes towards min first, sweeping up to max. End is max.
                 # If dist_to_max < dist_to_min, the path goes towards max first, sweeping down to min. End is min.
                 if dist_to_min <= dist_to_max:
                      start_floor_phase2_idx = max_f1_idx
                 else:
                      start_floor_phase2_idx = min_f1_idx
             # else: f1_indices was empty, start_floor_phase2_idx remains current_lift_floor_idx

        # Calculate movement cost for Phase 2
        movement_cost_phase2 = self.calculate_movement_cost(start_floor_phase2_idx, floors_to_visit_second_leg)

        # Total movement cost
        total_movement_cost = movement_cost_phase1 + movement_cost_phase2

        # Calculate action cost
        # Each passenger needing pickup requires 1 board action.
        # Each unserved passenger requires 1 depart action eventually.
        action_cost = len(passengers_to_pickup) + len(unserved_passengers)

        # Total heuristic is sum of movement and action costs
        total_cost = total_movement_cost + action_cost

        return total_cost
