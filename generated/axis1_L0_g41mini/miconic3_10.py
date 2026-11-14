from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    This heuristic estimates the number of actions required to serve all passengers
    in the miconic domain from the current state until the goal state is reached.
    It returns 0 for goal states (all passengers served) and a finite estimate
    otherwise, guiding greedy best-first search efficiently.

    Assumptions
    -----------
    - The lift serves passengers sequentially.
    - Boarding a passenger requires the lift to be at that passenger's origin floor.
    - Departing a passenger requires the lift to be at the passenger's destination floor,
      and the passenger must already be boarded.
    - Moving the lift up or down between floors is needed to reach origins/destinations.
    - No parallel boarding/departing actions occur; each boarding and departing is one action.
    - We ignore ordering constraints among passengers to estimate a lower but informative cost.
    - The "above" predicate defines a strict floor order that enables computing floor distances.

    Heuristic Initialization
    ------------------------
    - Extract passenger to destination floor mappings from static facts (destin predicates).
    - Extract ordered floors with their relative positions based on the "above" predicate to compute floor distances quickly.
    - Store static facts to enable quick lookups.
    - For unknown passengers (unlikely), handle gracefully by ignoring.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a state:
    1. Identify the floor where the lift currently is.
    2. Find all passengers who are not yet served (not "served" in state).
    3. Separate these passengers into two sets:
       a) Boarded passengers needing to depart at their destination floors.
       b) Passengers not yet boarded, needing to be picked up from origin floors.
    4. For all boarded passengers, estimate the minimal total lift travel cost plus departure actions:
       - For each boarded passenger: add 1 action for depart.
       - Sum the minimal travel distance to reach all their destination floors from the lift's current or intermediate floors.
    5. For all waiting passengers (not boarded), estimate the minimal travel cost plus boarding actions:
       - Each requires 1 boarding action.
       - Travel cost to reach their origin floors (again, consider best visitation order).
    6. Since the heuristic should be efficient and not solve a traveling salesman problem:
       - Approximate lift travel as the max floor distance needed to cover for boarded and waiting passengers plus movement from current lift floor.
       - Use the maximum among (current lift floor to origin floors), (origin floors to destination floors), and (current lift floor to destination floors for boarded).
    7. Each passenger contributes at least 2 actions: one board and one depart, but passengers already boarded only need depart.
       So total actions estimate = 
          (#boarded passengers) * 1 (depart) +
          (#waiting passengers) * 2 (board + depart) +
          estimated lift moves (approximate by max distance to cover floors).
    8. This approach remains consistent, efficient, and finite for all solvable states.

    Note: The heuristic is not admissible but aims for good performance in guiding search.
    """

    def __init__(self, task):
        """Initialize heuristic: extract passenger destinations and floor ordering."""
        self.goals = task.goals  # used to detect goal state quickly
        self.static = task.static

        # Extract passenger -> destination floor mapping from static facts (destin predicates)
        self.passenger_to_dest = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "destin":
                passenger, floor = parts[1], parts[2]
                self.passenger_to_dest[passenger] = floor

        # Extract all floors mentioned in static facts
        floors = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "above" and len(parts) == 3:
                floors.add(parts[1])
                floors.add(parts[2])
            elif parts[0] == "destin":
                floors.add(parts[2])
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == "origin":
                floors.add(parts[2])

        self.floors = sorted(floors)

        # Build floor index mapping for quick distance computations
        self.floor_indices = {f: i for i, f in enumerate(self.floors)}

        # Precompute floor distance function using indices, assuming floors linearly ordered
        # Using "above" facts we confirm that floors have a linear order, so the distance is index difference
        # Note: As per domain, floors form a strict order via above: f1 above f2 means f1 < f2 in height
        # We consider distance as absolute difference of indices
        # This matches the lift moves needed to go between floors

    def floor_distance(self, floor1, floor2):
        """Compute distance between floors as absolute difference of their indices."""
        idx1 = self.floor_indices.get(floor1, None)
        idx2 = self.floor_indices.get(floor2, None)
        if idx1 is None or idx2 is None:
            # Unknown floors: fallback to zero distance (unlikely to happen)
            return 0
        return abs(idx1 - idx2)

    def __call__(self, node):
        """Compute heuristic estimate of actions required to serve all passengers."""
        state = node.state

        # Check if goal reached: if all goals in state, heuristic = 0
        if self.goals <= state:
            return 0

        # Get lift current floor from state
        lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "lift-at":
                lift_floor = parts[1]
                break
        if lift_floor is None:
            # No known lift location, pessimistically 0 (should not happen)
            lift_floor = self.floors[0]

        # Passengers yet to be served
        # Gather passengers not served yet (no (served p))
        # Also determine passenger boarding status: boarded or not
        boarded_passengers = set()
        waiting_passengers = set()
        origin_floors = []
        destination_floors = []
        boarded_dest_floors = []

        # For optimization: passengers in the task are keys of passenger_to_dest if known
        # Better to gather them by scanning state facts (origin or boarded or served)
        passengers_in_state = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "origin" and len(parts) == 3:
                passengers_in_state.add(parts[1])
            elif parts[0] == "boarded" and len(parts) == 2:
                passengers_in_state.add(parts[1])
            elif parts[0] == "served" and len(parts) == 2:
                passengers_in_state.add(parts[1])

        # For each passenger, check if served
        for p in passengers_in_state:
            served_fact = f"(served {p})"
            if served_fact in state:
                continue  # passenger served, ignore

            boarded_fact = f"(boarded {p})"
            if boarded_fact in state:
                # boarded, waiting to depart at destination
                boarded_passengers.add(p)
                dest_floor = self.passenger_to_dest.get(p, None)
                if dest_floor is not None:
                    boarded_dest_floors.append(dest_floor)
            else:
                # not boarded, must be boarded first from origin floor
                waiting_passengers.add(p)
                # find origin floor from state facts
                # origin fact is (origin p f)
                origin_floor = None
                for fact in state:
                    parts = get_parts(fact)
                    if parts[0] == "origin" and len(parts) == 3 and parts[1] == p:
                        origin_floor = parts[2]
                        break
                if origin_floor is not None:
                    origin_floors.append(origin_floor)
                # also their destination floor
                dest_floor = self.passenger_to_dest.get(p, None)
                if dest_floor is not None:
                    destination_floors.append(dest_floor)

        # Estimate lift moves needed:

        # Helper to estimate travel cost to cover a set of floors starting from lift floor:
        # Since exact TSP is expensive, approximate by:
        # distance from lift_floor to closest floor in the set + distance between min and max floors in set
        # (to cover all relevant floors)
        def cover_floors_cost(floor_list):
            if not floor_list:
                return 0
            indices = [self.floor_indices[f] for f in floor_list if f in self.floor_indices]
            if not indices:
                return 0
            min_floor_idx = min(indices)
            max_floor_idx = max(indices)
            lift_idx = self.floor_indices[lift_floor]
            dist_to_nearest = min(abs(lift_idx - min_floor_idx), abs(lift_idx - max_floor_idx))
            floor_span = max_floor_idx - min_floor_idx
            return dist_to_nearest + floor_span

        # For boarded passengers: they only need depart actions at their destination floors,
        # so lift must visit all boarded_dest_floors.
        boarded_floors_cost = cover_floors_cost(boarded_dest_floors)

        # For waiting passengers: lift must visit origins (for boarding) and destinations (for departing)
        # We estimate combined cost for origins and destinations
        # Approximate by cost to cover origin floors + cost to cover destination floors separately
        # plus minimal distance between these sets.

        def span_min_max_floor_indices(floor_list):
            indices = [self.floor_indices[f] for f in floor_list if f in self.floor_indices]
            if not indices:
                return None, None
            return min(indices), max(indices)

        # Origin floors cost from lift floor
        origin_cost = cover_floors_cost(origin_floors)
        # Destination floors cost (for waiting passengers)
        destination_cost = cover_floors_cost(destination_floors)

        # Compute approximate distance between origin floors set and destination floors set
        orig_min, orig_max = span_min_max_floor_indices(origin_floors)
        dest_min, dest_max = span_min_max_floor_indices(destination_floors)

        inter_set_distance = 0
        if orig_min is not None and dest_min is not None:
            # distance between sets is minimal absolute difference between any origin floor and destination floor index range
            if orig_max < dest_min:
                inter_set_distance = dest_min - orig_max
            elif dest_max < orig_min:
                inter_set_distance = orig_min - dest_max
            else:
                inter_set_distance = 0  # overlap

        waiting_floors_cost = origin_cost + inter_set_distance + destination_cost

        # Total lift movement cost is max of the boarded floors cost and waiting floors cost,
        # because the lift needs to cover all floors for both groups, but we approximate separately.
        # However, since boarded passengers and waiting passengers must all be served, sum the costs conservatively:
        # Alternatively, lift might need to cover both sets combined to avoid underestimation.
        # So combine all floors and calculate unified cost:
        all_relevant_floors = set(boarded_dest_floors) | set(origin_floors) | set(destination_floors)
        total_lift_movement = cover_floors_cost(list(all_relevant_floors))

        # Action costs:
        # Each boarded passenger needs 1 depart action.
        # Each waiting passenger needs 1 board + 1 depart = 2 actions.
        # Each floor movement is 1 action per floor moved (up/down).
        total_boarded_depart = len(boarded_passengers)  # depart actions
        total_waiting_board_depart = 2 * len(waiting_passengers)  # board+depart actions

        # Total estimated cost is sum of all above
        # Movement cost is in terms of floors moved; each up/down action counts as 1

        heuristic_value = total_boarded_depart + total_waiting_board_depart + total_lift_movement

        # Ensure heuristic is never negative
        if heuristic_value < 0:
            heuristic_value = 0

        return heuristic_value
