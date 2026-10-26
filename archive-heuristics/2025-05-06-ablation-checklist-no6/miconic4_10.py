from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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

    Estimates the number of actions needed to reach the goal state (all passengers served).
    The heuristic is the sum of:
    1. A base cost: 2 for each unboarded, unserved passenger (board + depart),
       and 1 for each boarded, unserved passenger (depart).
    2. A movement cost: Estimated travel distance for the lift to visit all
       required pickup and dropoff floors. This is estimated as the span of
       required floors plus the distance from the current lift floor to the
       farthest required floor.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Passenger destinations.
        - Floor ordering and mapping to integers.
        - Set of all passenger names involved in the problem.
        """
        super().__init__(task)

        # Extract passenger destinations from static facts
        self.destinations = {}
        self.all_passengers = set()
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, destination = get_parts(fact)
                self.destinations[passenger] = destination
                self.all_passengers.add(passenger)

        # Collect all passenger names mentioned in the initial state and goal state
        # to ensure we consider all passengers relevant to the problem.
        for fact in self.initial_state:
             if match(fact, "origin", "*", "*"):
                _, passenger, _ = get_parts(fact)
                self.all_passengers.add(passenger)
             elif match(fact, "boarded", "*"):
                _, passenger = get_parts(fact)
                self.all_passengers.add(passenger)
             elif match(fact, "served", "*"):
                _, passenger = get_parts(fact)
                self.all_passengers.add(passenger)

        for fact in self.goals:
             if match(fact, "served", "*"):
                _, passenger = get_parts(fact)
                self.all_passengers.add(passenger)


        # Build floor ordering and mapping to integers
        all_floors = set()
        # Collect all floors mentioned in static 'above' facts
        for fact in self.static:
            if match(fact, "above", "*", "*"):
                _, floor1, floor2 = get_parts(fact)
                all_floors.add(floor1)
                all_floors.add(floor2)

        # If no 'above' facts (unlikely for miconic but defensive),
        # try to find floors in initial state facts like (lift-at) or (origin)
        if not all_floors:
             for fact in self.initial_state:
                 if match(fact, "lift-at", "*"):
                     _, floor = get_parts(fact)
                     all_floors.add(floor)
                 elif match(fact, "origin", "*", "*"):
                     _, _, floor = get_parts(fact)
                     all_floors.add(floor)
                 elif match(fact, "destin", "*", "*"): # Also check destin in initial state if present
                     _, _, floor = get_parts(fact)
                     all_floors.add(floor)


        # Sort floors based on the integer suffix (assuming f1, f2, f3... naming convention)
        # This is a domain-dependent assumption based on typical miconic instances.
        try:
            # Filter out any non-floor objects that might have been added by mistake
            floor_names = [f for f in all_floors if f.startswith('f') and f[1:].isdigit()]
            sorted_floors = sorted(floor_names, key=lambda f: int(f[1:]))
            self.floor_name_to_int = {f: i for i, f in enumerate(sorted_floors)}
        except (ValueError, IndexError):
            # Fallback if floor names are not in f<number> format.
            # Sort alphabetically as a robust fallback, though potentially less accurate.
            sorted_floors = sorted(list(all_floors))
            self.floor_name_to_int = {f: i for i, f in enumerate(sorted_floors)}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break

        # If lift location is unknown, the state is likely invalid or unhandleable
        if current_lift_floor is None:
             return float('inf') # Indicate invalid state


        # Identify unserved passengers and their status
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_passengers = self.all_passengers - served_passengers

        # If all passengers are served, the goal is reached
        if not unserved_passengers:
            return 0 # Goal state

        # Pre-process state to quickly check passenger status and locations
        boarded_in_state = set()
        origin_in_state = {} # Map passenger to origin floor
        for fact in state:
             parts = get_parts(fact)
             if not parts: continue # Skip malformed facts

             predicate = parts[0]
             if predicate == "boarded" and len(parts) == 2:
                 boarded_in_state.add(parts[1])
             elif predicate == "origin" and len(parts) == 3:
                 origin_in_state[parts[1]] = parts[2]


        pickup_floors = set()
        dropoff_floors = set()
        base_actions = 0 # Sum of board/depart actions

        for passenger in unserved_passengers:
            is_boarded = passenger in boarded_in_state
            origin_floor = origin_in_state.get(passenger)
            is_origin = origin_floor is not None

            if is_boarded:
                # Passenger is boarded, needs to be dropped off at destination
                dest_floor = self.destinations.get(passenger)
                if dest_floor: # Ensure destination is known from static facts
                    dropoff_floors.add(dest_floor)
                    base_actions += 1 # Needs 1 depart action
                # else: Destination unknown for this passenger. Heuristic might be inaccurate.

            elif is_origin:
                # Passenger is at origin, needs pickup and dropoff
                # origin_floor is already known from origin_in_state lookup
                if origin_floor: # Should always be true if is_origin is true
                    pickup_floors.add(origin_floor)
                    dest_floor = self.destinations.get(passenger)
                    if dest_floor: # Ensure destination is known from static facts
                        dropoff_floors.add(dest_floor)
                        base_actions += 2 # Needs 1 board + 1 depart action
                    # else: Destination unknown for this passenger. Heuristic might be inaccurate.

            # Passengers not in state as (origin) or (boarded) but not (served)
            # are not explicitly handled here, assuming valid miconic states
            # transition correctly between these predicates.

        # Calculate movement cost
        required_floors = pickup_floors.union(dropoff_floors)

        movement_cost = 0
        if required_floors:
            # Ensure all required floors are in our mapping
            valid_required_floors = {f for f in required_floors if f in self.floor_name_to_int}

            if valid_required_floors:
                required_indices = {self.floor_name_to_int[f] for f in valid_required_floors}
                min_idx = min(required_indices)
                max_idx = max(required_indices)

                # Get current lift index, handle case where lift floor isn't in mapping (shouldn't happen)
                current_lift_idx = self.floor_name_to_int.get(current_lift_floor)
                if current_lift_idx is None:
                     # Lift floor not in mapping - indicates problem setup issue.
                     # Fallback: estimate movement based only on required floors span.
                     movement_cost = max_idx - min_idx
                else:
                    # Estimate movement: Go to one extreme of required floors, sweep to the other.
                    # Add distance from current floor to the farthest required floor.
                    dist_to_min = abs(current_lift_idx - min_idx)
                    dist_to_max = abs(current_lift_idx - max_idx)
                    span = max_idx - min_idx
                    movement_cost = max(dist_to_min, dist_to_max) + span
            # else: No valid required floors found in mapping. movement_cost remains 0.

        # Total heuristic is the sum of base actions and estimated movement
        return base_actions + movement_cost
