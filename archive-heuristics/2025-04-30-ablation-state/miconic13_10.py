# Need to import fnmatch for the match helper function
from fnmatch import fnmatch
import sys # Import sys for float('inf')

# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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

# Assume Heuristic base class is defined elsewhere and imported
# from heuristics.heuristic_base import Heuristic

# Inherit from Heuristic if available, otherwise just define the class
class miconicHeuristic: # (Heuristic)
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers.
    It sums the minimum required board/depart actions for unserved passengers
    and an estimate of the vertical movement cost required to visit all floors
    where passengers need pickup or dropoff.

    # Assumptions
    - Passengers need to be picked up at their origin and dropped off at their destination.
    - The lift can carry multiple passengers.
    - The 'above' facts define a linear order of floors.
    - Movement cost is estimated by the distance needed to traverse the range of floors
      requiring attention (pickup or dropoff for unserved passengers).

    # Heuristic Initialization
    - Build a mapping from floor names to their index based on the 'above' static facts.
    - Store the destination floor for each passenger from the 'destin' static facts.
    - Identify all passenger names involved in the problem based on origin, destin, and goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Get the current state and identify the lift's current floor.
    2. Identify all passengers who are not yet served by checking which goal facts `(served p)`
       are not present in the current state.
    3. If no passengers are unserved, the heuristic is 0.
    4. For each unserved passenger:
       - If the passenger is waiting at their origin floor (fact `(origin p f)` exists in state),
         increment the count of waiting passengers (`N_waiting`). Add their origin
         floor to the set of floors needing pickup (`waiting_origin_floors`). Add their
         destination floor (looked up from static facts) to the set of floors requiring
         a stop for dropoff (`required_dest_floors`).
       - If the passenger is boarded (fact `(boarded p)` exists in state), increment the count
         of boarded passengers (`N_boarded`). Add their destination floor (looked up from
         static facts) to the set of floors requiring a stop for dropoff (`required_dest_floors`).
    5. Combine the sets of pickup floors and dropoff floors for unserved passengers
       to get the set of `Required_stops`.
    6. Map the `Required_stops` floors to their numerical indices using the pre-computed map.
    7. Find the minimum (`min_required_floor_idx`) and maximum (`max_required_floor_idx`)
       floor indices among the `Required_stops`.
    8. Calculate the movement cost:
       - If there is only one floor in `Required_stops`, the cost is the absolute
         distance between the current lift floor index and that required floor index.
       - If there are multiple floors in `Required_stops`, the cost is the sum of
         the absolute distance from the current lift floor index to the minimum required
         floor index plus the absolute distance from the maximum required floor index
         to the current lift floor index. This estimates the travel needed to reach
         and cover the span of required floors.
    9. The total heuristic value is `N_waiting + N_boarded + movement_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, floor indices,
        passenger destinations, and the set of all passengers from static facts.
        """
        self.goals = task.goals # Goal conditions (e.g., {(served p1), ...})
        static_facts = task.static

        # 1. Build floor order and floor-to-index map
        above_map = {} # Maps f_lower -> f_higher
        all_floors = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                f_higher, f_lower = parts[1], parts[2]
                above_map[f_lower] = f_higher
                all_floors.add(f_higher)
                all_floors.add(f_lower)

        # Find the lowest floor: the floor that is never the value (f_higher) in an (above f_higher f_lower) fact
        floors_that_are_above_something = set(above_map.values())
        lowest_floor = None
        # Handle case with only one floor
        if len(all_floors) == 1:
             lowest_floor = list(all_floors)[0]
        else:
            for floor in all_floors:
                if floor not in floors_that_are_above_something:
                     lowest_floor = floor
                     break
        
        # If lowest_floor is still None, it implies an issue with the 'above' facts structure
        # (e.g., not a linear chain, or empty set of floors). For valid miconic, this shouldn't happen.
        # We proceed assuming lowest_floor was found if all_floors is not empty.


        # Build the ordered list from lowest to highest using above_map
        self.floor_list = []
        current = lowest_floor
        while current is not None:
            self.floor_list.append(current)
            current = above_map.get(current) # Get the floor directly above current

        # Build floor-to-index map (0-indexed from lowest)
        self.floor_to_index = {floor: i for i, floor in enumerate(self.floor_list)}

        # 2. Store passenger destinations and identify all passengers
        self.passenger_destinations = {}
        self.all_passengers = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "destin":
                p, destin_f = parts[1], parts[2]
                self.passenger_destinations[p] = destin_f
                self.all_passengers.add(p)
            # Also add passengers from origin facts, as destin facts might not list all passengers
            elif parts and parts[0] == "origin":
                 p, origin_f = parts[1], parts[2]
                 self.all_passengers.add(p)

        # Also add passengers from goal facts, in case they aren't in origin/destin (unlikely but safe)
        for goal_fact in self.goals:
             parts = get_parts(goal_fact)
             if parts and parts[0] == "served":
                  self.all_passengers.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Get current lift floor
        current_f = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_f = get_parts(fact)[1]
                break
        
        # If lift location is not specified, heuristic is infinite (or a large number)
        # This shouldn't happen in valid states, but as a safeguard:
        if current_f is None or current_f not in self.floor_to_index:
             # Cannot compute heuristic without a valid lift location.
             return float('inf') # Return infinity to prune this state

        current_f_idx = self.floor_to_index[current_f]

        # 2. Identify unserved passengers and their needs
        N_waiting = 0
        N_boarded = 0
        waiting_origin_floors = set()
        required_dest_floors = set()

        # Find all passengers that are in the goal but not in the served state
        goal_passengers = {get_parts(goal_fact)[1] for goal_fact in self.goals if match(goal_fact, "served", "*")}
        served_passengers_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_passengers = goal_passengers - served_passengers_in_state

        if not unserved_passengers:
            return 0 # Goal reached

        # Now iterate through the identified unserved passengers
        for p in unserved_passengers:
            destin_f = self.passenger_destinations.get(p) # Get destination from static facts
            if destin_f is None:
                 # This passenger is in the goal but has no destination defined? Should not happen in valid PDDL
                 continue # Skip this passenger

            if f"(boarded {p})" in state:
                N_boarded += 1
                required_dest_floors.add(destin_f)
            else: # Passenger is not served and not boarded, must be waiting at origin
                # Find origin floor for this passenger in the current state
                origin_f = None
                for fact in state:
                    if match(fact, "origin", p, "*"):
                        origin_f = get_parts(fact)[2]
                        break
                # If origin_f is None, the state is likely invalid (unserved, not boarded, not waiting)
                if origin_f:
                    N_waiting += 1
                    waiting_origin_floors.add(origin_f)
                    required_dest_floors.add(destin_f) # Destination is also a required stop


        # 5. Combine required stops
        Required_stops = waiting_origin_floors.union(required_dest_floors)

        # 6. If no required stops (should only happen if unserved_passengers was empty, but double check)
        if not Required_stops:
             # This case should be covered by the check `if not unserved_passengers:`
             # If we reach here, it means there are unserved passengers, but they
             # are neither waiting nor boarded (e.g., problem state error or passenger
             # type not handled). Assuming valid states, this block might be redundant
             # but let's return the board/depart count as a fallback.
             return N_waiting + N_boarded # Should be 0 if Required_stops is empty

        # 7. Get indices for required stops
        required_indices = {self.floor_to_index[f] for f in Required_stops}

        # 8. Find min and max required floor indices
        min_required_floor_idx = min(required_indices)
        max_required_floor_idx = max(required_indices)

        # 9. Calculate movement cost
        if len(required_indices) == 1:
            # Only one floor needs visiting
            movement_cost = abs(current_f_idx - min_required_floor_idx)
        else:
            # Multiple floors need visiting. Estimate movement to cover the span.
            # This is the distance to reach the lowest required floor + distance from lowest to highest.
            # Or distance to reach highest + distance from highest to lowest.
            # The sum of distances from current to min and current to max covers the span
            # and accounts for the current position relative to the span.
            movement_cost = abs(current_f_idx - min_required_floor_idx) + abs(max_required_floor_idx - current_f_idx)


        # 10. Total heuristic
        total_cost = N_waiting + N_boarded + movement_cost

        return total_cost
