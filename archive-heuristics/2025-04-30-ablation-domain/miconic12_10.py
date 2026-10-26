from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    # Split by spaces, ignoring spaces within quoted strings if any (not typical in miconic)
    # A simple split is sufficient for miconic predicates like (pred obj1 obj2)
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It sums the number of pending 'board' actions, pending 'depart' actions,
    and an estimate of the lift's travel cost.

    # Assumptions
    - Each waiting passenger needs one 'board' action.
    - Each unserved passenger needs one 'depart' action.
    - The lift must travel to the origin floor of waiting passengers and the
      destination floor of all unserved passengers.
    - The travel cost is estimated as the distance from the current lift floor
      to the furthest floor that is either an origin of a waiting passenger
      or a destination of an unserved passenger.

    # Heuristic Initialization
    - Parses static facts to determine the floor order and create a mapping
      from floor names (e.g., 'f1', 'f2') to numerical indices (0, 1, ...).
    - Parses static facts to store the destination floor for each passenger.

    # Step-By-Step Thinking for Computing Heuristic
    1. Determine the current floor of the lift.
    2. Identify all passengers who are not yet served.
    3. Partition unserved passengers into two groups: waiting (at their origin)
       and boarded (inside the lift).
    4. Count the number of waiting passengers (`num_waiting`). Each needs a 'board' action.
    5. Count the number of unserved passengers (`num_unserved`). Each needs a 'depart' action.
    6. Identify the set of "relevant" floors:
       - The origin floor for every waiting passenger.
       - The destination floor for every unserved passenger (both waiting and boarded).
    7. If there are no relevant floors (meaning all passengers are served), the heuristic is 0.
    8. If there are relevant floors, find the minimum and maximum floor indices among them.
    9. Calculate the estimated travel cost: This is the maximum of the distance
       from the current lift floor index to the minimum relevant floor index,
       and the distance from the current lift floor index to the maximum relevant
       floor index. This estimates the minimum travel needed to reach the extremes
       of the required stops.
    10. The total heuristic value is `num_waiting + num_unserved + estimated_travel_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals # Keep goals reference if needed, though not directly used in this h
        static_facts = task.static

        # 1. Build floor index mapping
        # Use 'above' predicates to determine floor order.
        # Build a map from floor_lower -> floor_upper
        is_above_map = {}
        # Collect all floor names
        all_floors = set()
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, f_upper, f_lower = get_parts(fact)
                is_above_map[f_lower] = f_upper
                all_floors.add(f_lower)
                all_floors.add(f_upper)

        if not all_floors:
             # Handle case with no floors (shouldn't happen in valid miconic)
             self.floor_indices = {}
        else:
            # Find the lowest floor (a floor that is not the upper floor in any 'above' fact)
            upper_floors = set(is_above_map.values())
            lowest_floor = None
            for floor in all_floors:
                if floor not in upper_floors:
                    lowest_floor = floor
                    break

            # Build the ordered list of floors starting from the lowest
            ordered_floors = []
            current = lowest_floor
            while current is not None:
                ordered_floors.append(current)
                current = is_above_map.get(current) # Get the floor immediately above

            # Create the floor name to index mapping
            self.floor_indices = {floor: i for i, floor in enumerate(ordered_floors)}

        # 2. Store passenger destinations
        self.passenger_destinations = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, destination_floor = get_parts(fact)
                self.passenger_destinations[passenger] = destination_floor

        # Get all passenger names from destinations (assuming all passengers have a destination)
        self.all_passengers = set(self.passenger_destinations.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Determine current lift floor
        lift_at_fact = next((fact for fact in state if match(fact, "lift-at", "*")), None)
        if lift_at_fact is None:
             # Should not happen in a valid miconic state, but handle defensively
             return float('inf') # Or some large value indicating invalid state

        f_lift = get_parts(lift_at_fact)[1]
        idx_lift = self.floor_indices.get(f_lift)
        if idx_lift is None:
             # Should not happen if floor_indices was built correctly
             return float('inf')

        # 2. Identify passenger states
        served_passengers = {p for fact in state if match(fact, "served", "*") for _, p in [get_parts(fact)]}
        boarded_passengers = {p for fact in state if match(fact, "boarded", "*") for _, p in [get_parts(fact)]}
        
        # Waiting passengers are those not served and not boarded
        unserved_passengers = self.all_passengers - served_passengers
        waiting_passengers = unserved_passengers - boarded_passengers

        # 3. Count pending actions
        num_waiting = len(waiting_passengers)
        num_unserved = len(unserved_passengers)

        # 4. Identify relevant floors
        F_pickup = {o for fact in state if match(fact, "origin", "*", "*") for _, p, o in [get_parts(fact)] if p in waiting_passengers}
        F_dropoff = {self.passenger_destinations[p] for p in unserved_passengers}

        F_all_relevant = F_pickup.union(F_dropoff)

        # 5. If no relevant floors, all unserved passengers must be at their destination (and served)
        # This check is slightly redundant if num_unserved is 0, but handles cases where
        # F_all_relevant might be empty for other reasons (e.g., initial state is goal)
        if not F_all_relevant:
            return 0 # Goal state

        # 6. Calculate estimated travel cost
        relevant_indices = [self.floor_indices[f] for f in F_all_relevant if f in self.floor_indices]
        if not relevant_indices:
             # Should not happen if F_all_relevant is not empty and floor_indices is correct
             return float('inf')

        idx_min_all = min(relevant_indices)
        idx_max_all = max(relevant_indices)

        # Travel cost is the distance from current floor to the furthest relevant floor
        travel_cost = max(abs(idx_lift - idx_min_all), abs(idx_lift - idx_max_all))

        # 7. Total heuristic value
        # Each waiting passenger needs 1 board action.
        # Each unserved passenger needs 1 depart action.
        # Add estimated travel cost.
        heuristic_value = num_waiting + num_unserved + travel_cost

        return heuristic_value

