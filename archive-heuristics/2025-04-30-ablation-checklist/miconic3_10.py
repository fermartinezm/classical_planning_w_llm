from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    passengers. It considers the number of boarding and departing actions
    needed at distinct floors and estimates the lift movement cost based
    on the range of floors that need service (pickup or dropoff).

    # Assumptions
    - Floors are linearly ordered.
    - Each passenger needs one board and one depart action.
    - Multiple passengers can be boarded/departed at the same floor visit.
    - Movement cost is estimated based on traversing the range of floors
      requiring service.

    # Heuristic Initialization
    - Extracts the floor ordering from the static `above` facts to create
      a floor-to-index mapping.
    - Extracts passenger destinations from the static `destin` facts.

    # Step-by-Step Thinking for Computing the Heuristic Value
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Unserved Passengers:** Determine which passengers have not
        yet reached their destination (`(served ?p)` is not in the state).

    2.  **Identify Service Floors:** For the unserved passengers:
        *   If a passenger `p` is `(origin p o)`, floor `o` needs a pickup service.
        *   If a passenger `p` is `(boarded p)`, their destination floor `d`
            (obtained from static facts) needs a dropoff service.
        *   Collect all unique floors that need either pickup or dropoff service
            into a set `ServiceFloors`.

    3.  **Count Actions:**
        *   Estimate the number of board actions needed: This is the number of
            distinct floors in `ServiceFloors` that are origin floors for
            waiting passengers (`PickupFloors`).
        *   Estimate the number of depart actions needed: This is the number of
            distinct floors in `ServiceFloors` that are destination floors for
            boarded passengers (`DropoffFloors`).
        *   Total non-movement actions = `len(PickupFloors) + len(DropoffFloors)`.

    4.  **Estimate Movement Cost:**
        *   Find the current floor of the lift (`CurrentFloor`).
        *   If `ServiceFloors` is empty, movement cost is 0.
        *   If `ServiceFloors` is not empty:
            *   Find the lowest floor (`f_min_service`) and the highest floor
                (`f_max_service`) among the `ServiceFloors` using the floor index mapping.
            *   Estimate the movement cost as the minimum distance required to
                travel from the `CurrentFloor` to cover the range from
                `f_min_service` to `f_max_service`. A simple estimate is the
                distance from the current floor to one end of the service range,
                plus the distance to traverse the entire service range.
                `movement_cost = min(dist(CurrentFloor, f_min_service) + dist(f_min_service, f_max_service),
                                    dist(CurrentFloor, f_max_service) + dist(f_max_service, f_min_service))`

    5.  **Sum Costs:** The total heuristic value is the sum of the estimated
        non-movement actions and the estimated movement cost.
        `h = len(PickupFloors) + len(DropoffFloors) + movement_cost`

    This heuristic is non-admissible because it sums costs for distinct floors
    and uses a simplified movement model, but it aims to be informative by
    considering the necessary stops and the travel distance.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor ordering and passenger destinations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # 1. Build floor ordering and index mapping
        # We assume floors are linearly ordered based on (above f_i f_{i+1}) facts.
        # Find the floor immediately below each floor.
        floor_below_map = {}
        all_floors = set()
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f_above, f_below = get_parts(fact)[1:]
                floor_below_map[f_above] = f_below
                all_floors.add(f_above)
                all_floors.add(f_below)

        # Find the highest floor (appears as f_above but never as f_below)
        highest_floor = None
        floors_appearing_as_below = set(floor_below_map.values())
        for floor in all_floors:
             if floor not in floors_appearing_as_below:
                 # Check if it appears as f_above at all (to exclude floors not connected)
                 if floor in floor_below_map:
                    highest_floor = floor
                    break # Found the highest floor

        if highest_floor is None and all_floors:
             # Handle case with only one floor or circular 'above' (shouldn't happen in valid PDDL)
             # Or if the highest floor doesn't appear as f_above (e.g., only f2 in (above f1 f2))
             # In a linear structure f1 above f2 above f3 ... fN, f1 is highest.
             # f1 appears as f_above, but never as f_below.
             # Let's re-find highest: floor f such that no (above ?any f) exists.
             floors_appearing_as_above = set(floor_below_map.keys())
             floors_appearing_as_below = set(floor_below_map.values())
             # Highest floor is in keys but not values
             potential_highest = floors_appearing_as_above - floors_appearing_as_below
             if len(potential_highest) == 1:
                 highest_floor = potential_highest.pop()
             elif len(all_floors) == 1:
                 highest_floor = list(all_floors)[0] # Single floor case
             else:
                 # Fallback: Assume floors are f1, f2, ... fN and f1 is highest
                 # This is a strong assumption but common in miconic benchmarks
                 sorted_floors = sorted(list(all_floors), key=lambda f: int(f[1:]))
                 highest_floor = sorted_floors[-1] # Assume fN is highest
                 # Rebuild map based on assumed order f1 < f2 < ...
                 floor_below_map = {}
                 for i in range(len(sorted_floors) - 1):
                     floor_below_map[sorted_floors[i+1]] = sorted_floors[i]
                 highest_floor = sorted_floors[-1] # fN is highest
                 # Let's reverse the index logic: f1 is lowest (index 0), fN is highest (index N-1)
                 floor_above_map = {}
                 for i in range(len(sorted_floors) - 1):
                     floor_above_map[sorted_floors[i]] = sorted_floors[i+1]
                 lowest_floor = sorted_floors[0] # f1 is lowest

                 ordered_floors = []
                 current = lowest_floor
                 while current is not None:
                     ordered_floors.append(current)
                     current = floor_above_map.get(current)

                 self.floor_indices = {floor: i for i, floor in enumerate(ordered_floors)}
                 # Use this floor_indices and skip the original highest_floor logic
                 highest_floor = ordered_floors[-1] # Highest floor by index
                 lowest_floor = ordered_floors[0] # Lowest floor by index

        if not hasattr(self, 'floor_indices'):
             # Build the ordered list starting from the highest floor found
             ordered_floors = []
             current = highest_floor
             # Follow the 'below' links
             while current is not None:
                 ordered_floors.append(current)
                 current = floor_below_map.get(current)

             # Reverse the list to have lowest floor at index 0
             ordered_floors.reverse()
             self.floor_indices = {floor: i for i, floor in enumerate(ordered_floors)}


        # 2. Store passenger destinations
        self.destinations = {}
        for goal in self.goals:
            # Goal is typically (served ?p)
            # We need destinations from static facts
            pass # Destinations are not in goals, they are static facts

        # Extract destinations from static facts
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                passenger, destination = get_parts(fact)[1:]
                self.destinations[passenger] = destination

    def get_floor_index(self, floor_name):
        """Get the numerical index for a given floor name."""
        return self.floor_indices.get(floor_name, -1) # Return -1 or raise error if floor not found

    def dist(self, floor1, floor2):
        """Calculate the distance (number of floors to traverse) between two floors."""
        idx1 = self.get_floor_index(floor1)
        idx2 = self.get_floor_index(floor2)
        if idx1 == -1 or idx2 == -1:
            # This should not happen with valid inputs based on domain facts
            return float('inf')
        return abs(idx1 - idx2)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find current lift location
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        if current_lift_floor is None:
             # Should not happen in a valid state, but handle defensively
             return float('inf') # Cannot proceed without lift location

        # Identify unserved passengers and their status (origin or boarded)
        unserved_passengers = {} # {passenger: 'origin' or 'boarded'}
        served_passengers = set()

        for fact in state:
            if match(fact, "served", "*"):
                served_passengers.add(get_parts(fact)[1])

        # Collect all passengers mentioned in origin or boarded facts
        all_passengers_in_state = set()
        for fact in state:
            if match(fact, "origin", "*", "*"):
                 all_passengers_in_state.add(get_parts(fact)[1])
            elif match(fact, "boarded", "*"):
                 all_passengers_in_state.add(get_parts(fact)[1])

        # Determine status for unserved passengers
        for passenger in all_passengers_in_state:
            if passenger not in served_passengers:
                is_origin = any(match(fact, "origin", passenger, "*") for fact in state)
                is_boarded = any(match(fact, "boarded", passenger) for fact in state)

                if is_origin:
                    unserved_passengers[passenger] = 'origin'
                elif is_boarded:
                    unserved_passengers[passenger] = 'boarded'
                # Passengers not in origin/boarded/served facts are likely an issue,
                # but we only care about those needing service.

        # Identify floors needing service (pickup or dropoff)
        pickup_floors = set()
        dropoff_floors = set()

        for passenger, status in unserved_passengers.items():
            if status == 'origin':
                # Find their origin floor
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                        pickup_floors.add(get_parts(fact)[2])
                        break
            elif status == 'boarded':
                # Find their destination floor (from static facts)
                dest_floor = self.destinations.get(passenger)
                if dest_floor: # Ensure destination is known
                    dropoff_floors.add(dest_floor)

        service_floors = pickup_floors | dropoff_floors

        # Calculate heuristic components
        h = 0

        # 1. Add cost for board actions needed at distinct pickup floors
        h += len(pickup_floors)

        # 2. Add cost for depart actions needed at distinct dropoff floors
        h += len(dropoff_floors)

        # 3. Estimate movement cost
        if not service_floors:
            movement_cost = 0
        else:
            # Find min and max floor indices among service floors
            service_floor_indices = [self.get_floor_index(f) for f in service_floors]
            min_idx = min(service_floor_indices)
            max_idx = max(service_floor_indices)

            # Get the actual floor names for min/max indices
            # This requires reversing the floor_indices map, or finding the floor by index
            # Let's just iterate through service_floors to find min/max floor names
            f_min_service = min(service_floors, key=lambda f: self.get_floor_index(f))
            f_max_service = max(service_floors, key=lambda f: self.get_floor_index(f))

            # Estimate movement: go from current floor to one end of the service range,
            # then traverse the range. Take the minimum of going down first vs. up first.
            dist_L_min = self.dist(current_lift_floor, f_min_service)
            dist_L_max = self.dist(current_lift_floor, f_max_service)
            dist_min_max = self.dist(f_min_service, f_max_service)

            # Option 1: Go to lowest service floor, then sweep up to highest
            cost1 = dist_L_min + dist_min_max
            # Option 2: Go to highest service floor, then sweep down to lowest
            cost2 = dist_L_max + dist_min_max

            movement_cost = min(cost1, cost2)

            h += movement_cost

        return h

