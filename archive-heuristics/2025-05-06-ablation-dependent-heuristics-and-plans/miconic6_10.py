from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    This heuristic estimates the remaining effort by summing the individual costs
    for each unserved passenger. The cost for a passenger depends on whether they
    are waiting at their origin or are already boarded, and involves lift movement
    to pickup/dropoff floors and the board/depart actions. It ignores the possibility
    of serving multiple passengers at the same floor visit.

    # Assumptions
    - The floor structure is defined by `(above f_lower f_higher)` facts, where
      `f_higher` is immediately above `f_lower`. These facts are expected to form
      a single linear chain of floors.
    - Passenger destinations are static and available in the initial state facts.
    - Action costs are uniform (cost 1).

    # Heuristic Initialization
    - Parses `(above f_lower f_higher)` facts from static information to build
      a mapping from floor names to numerical levels, allowing distance calculation.
      If the floor structure is ambiguous (e.g., disconnected or no `above` facts),
      all floors are assigned level 0, resulting in zero distance between any two floors.
    - Parses `(destin ?person ?floor)` facts from the initial state to store
      the destination floor for each passenger. Also collects all passenger names
      mentioned in relevant initial state facts (`origin`, `destin`, `served`, `boarded`).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value for a given state is computed as follows:

    1. Identify the current floor of the lift by finding the fact `(lift-at ?floor)` in the state.
    2. Initialize the total estimated cost to 0.
    3. For each passenger identified during initialization:
       a. Check if the passenger is already served (i.e., `(served passenger)` is true in the state). If yes, this passenger requires no further actions; continue to the next passenger.
       b. If the passenger is not served, check if they are currently boarded in the lift (i.e., `(boarded passenger)` is true in the state).
          - If boarded: The estimated cost for this passenger is the number of floors the lift needs to move from its current location to the passenger's destination floor (calculated using the floor level map), plus 1 action for the `depart` action. Add this cost to the total.
       c. If the passenger is not served and not boarded, they must be waiting at their origin floor (i.e., `(origin passenger origin_floor)` is true in the state).
          - If waiting: Find their origin floor by searching for `(origin passenger ?floor)` in the state. The estimated cost for this passenger is the number of floors the lift needs to move from its current location to the origin floor, plus 1 action for the `board` action, plus the number of floors the lift needs to move from the origin floor to the destination floor, plus 1 action for the `depart` action. Add this cost to the total.
    4. The total estimated cost accumulated from all unserved passengers is the heuristic value for the state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        self.goals = task.goals # Goal conditions (used implicitly by checking served status)
        static_facts = task.static
        initial_state_facts = task.initial_state # Destinations are typically in initial state

        # 1. Build floor level mapping from (above f_lower f_higher) facts
        above_pairs = []
        all_floor_names = set()

        # Collect all floor names mentioned in relevant static/initial facts
        for fact in static_facts | initial_state_facts:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'above' and len(parts) == 3:
                 f1, f2 = parts[1], parts[2]
                 # Interpretation: (above f_lower f_higher) means f_higher is immediately above f_lower
                 above_pairs.append((f1, f2))
                 all_floor_names.add(f1)
                 all_floor_names.add(f2)
            elif predicate in ['lift-at', 'origin', 'destin'] and len(parts) > 1:
                 # Assuming the last argument is always a floor for these predicates
                 all_floor_names.add(parts[-1])

        self.floor_to_level = {}

        if not all_floor_names:
            # No floors found at all
            print("Warning: No floors found in task facts.")
        elif not above_pairs:
             # If floors exist but no above facts, assume they are all on the same level or disconnected.
             # Assign level 0 to all floors. Distance is always 0.
             self.floor_to_level = {f: 0 for f in all_floor_names}
             print("Warning: No 'above' facts found. Assuming all floors are at the same level.")
        else:
            # Build map: f_lower -> f_higher
            floor_above_map = {f_lower: f_higher for f_lower, f_higher in above_pairs}

            # Find the lowest floor: a floor that appears as f_lower but never as f_higher
            lower_floors_in_pairs = set(f_lower for f_lower, f_higher in above_pairs)
            higher_floors_in_pairs = set(f_higher for f_lower, f_higher in above_pairs)
            lowest_floors = lower_floors_in_pairs - higher_floors_in_pairs

            # If the structure is a single chain, there should be exactly one lowest floor.
            # If not, the floor structure is ambiguous or disconnected. Fallback to level 0.
            if len(lowest_floors) != 1:
                 print(f"Warning: Could not determine unique lowest floor from above facts. Found: {lowest_floors}. All floors: {all_floor_names}. Assigning level 0 to all.")
                 self.floor_to_level = {f: 0 for f in all_floor_names}
            else:
                f_lowest = lowest_floors.pop()
                Floors_ordered = []
                floor_to_level = {}
                current = f_lowest
                level = 0

                # Traverse upwards from the lowest floor
                while current is not None:
                    Floors_ordered.append(current)
                    floor_to_level[current] = level
                    level += 1
                    current = floor_above_map.get(current)

                # Check if all identified floors were included in the chain traversal
                # If not, assign level 0 to any missing floors (disconnected)
                if set(Floors_ordered) != all_floor_names:
                    print(f"Warning: Not all floors ({all_floor_names}) included in the 'above' chain traversal ({set(Floors_ordered)}). Assigning level 0 to missing floors.")
                    for f in all_floor_names:
                        if f not in floor_to_level:
                            floor_to_level[f] = 0
                self.floor_to_level = floor_to_level


        # 2. Store passenger destinations from initial state and collect all passenger names
        self.passenger_destinations = {}
        self.all_passengers = set()
        for fact in initial_state_facts:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == "destin" and len(parts) == 3:
                passenger, destination_floor = parts[1], parts[2]
                self.passenger_destinations[passenger] = destination_floor
                self.all_passengers.add(passenger)
            elif predicate in ["origin", "served", "boarded"] and len(parts) > 1:
                 passenger = parts[1]
                 self.all_passengers.add(passenger)


    def dist(self, f1, f2):
        """Calculate the distance (number of floors to traverse) between two floors."""
        # If a floor is not in the map (e.g., due to parsing issues or invalid instance),
        # return a large value to penalize states involving unknown floors.
        if f1 not in self.floor_to_level or f2 not in self.floor_to_level:
            # This indicates a problem with floor parsing or instance definition
            # print(f"Error: Floor {f1} or {f2} not found in floor level map.") # Avoid excessive printing in search
            return 1000000 # Arbitrarily large value

        return abs(self.floor_to_level[f1] - self.floor_to_level[f2])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached (all passengers served)
        # The goal is a set of (served p) facts.
        # We check if the set of goal facts is a subset of the state facts.
        if self.goals <= state:
            return 0

        # 1. Identify current lift floor
        current_lift_f = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_f = get_parts(fact)[1]
                break

        if current_lift_f is None:
            # This should not happen in a valid miconic state, but handle defensively.
            # If lift location is unknown, we can't move. Assume unsolvable or return a large cost.
            # print("Error: Lift location not found in state.") # Avoid excessive printing in search
            return 1000000 # Arbitrarily large value

        total_cost = 0  # Initialize action cost counter.

        # 2. Iterate through all passengers identified during initialization
        for passenger in self.all_passengers:
            # Check if passenger is served
            if f"(served {passenger})" in state:
                continue # Passenger is served, no cost

            # Passenger is not served. Find their state (boarded or waiting)
            dest_f = self.passenger_destinations.get(passenger)
            if dest_f is None:
                # This passenger doesn't have a destination defined. Problematic instance?
                # print(f"Warning: Destination not found for passenger {passenger}.") # Avoid excessive printing in search
                # Cannot estimate cost for this passenger, add a penalty?
                total_cost += 100 # Arbitrary penalty
                continue

            if f"(boarded {passenger})" in state:
                # Passenger is boarded, needs to reach destination and depart
                # Cost = moves to destination + depart action
                total_cost += self.dist(current_lift_f, dest_f) + 1
            else:
                # Passenger is not served and not boarded, must be waiting at origin
                # Find origin floor
                origin_f = None
                # Efficiently check for origin fact using set membership first
                origin_fact_str = None
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                         origin_fact_str = fact
                         break

                if origin_fact_str:
                    origin_f = get_parts(origin_fact_str)[2]
                else:
                    # Passenger is not served, not boarded, and not at an origin. Invalid state?
                    # Or maybe they were picked up but the origin fact wasn't deleted?
                    # Assuming valid states, this shouldn't happen for an unserved, unboarded passenger.
                    # print(f"Error: Unserved, unboarded passenger {passenger} not found at any origin.") # Avoid excessive printing in search
                    # Cannot estimate cost for this passenger reliably, add a base cost.
                    total_cost += 100 # Arbitrary penalty
                    continue

                # Passenger is waiting at origin, needs pickup and dropoff
                # Cost = moves to origin + board action + moves to destination + depart action
                total_cost += self.dist(current_lift_f, origin_f) + 1 + self.dist(origin_f, dest_f) + 1

        return total_cost
