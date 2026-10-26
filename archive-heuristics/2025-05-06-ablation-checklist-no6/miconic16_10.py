from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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

    Estimates the cost as the number of necessary board/depart actions
    plus the estimated minimum vertical distance the lift must travel to visit
    all floors where actions (pickup or dropoff) are required for unserved passengers.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and goal passengers.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal passengers
        self.goal_passengers = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'served' and len(parts) == 2:
                self.goal_passengers.add(parts[1])

        # Extract floor order and create floor_to_index map
        self.floor_to_index = {}
        if static_facts:
            above_relations = {} # Map: floor_below -> floor_above (immediate relation)
            all_floors = set()
            floors_above = set() # Floors that appear as f_high
            floors_below = set() # Floors that appear as f_low

            # Collect all floors and immediate above relations
            for fact in static_facts:
                parts = get_parts(fact)
                if parts and parts[0] == 'above' and len(parts) == 3:
                    f_high, f_low = parts[1], parts[2]
                    # Assuming (above f_high f_low) means f_high is immediately above f_low
                    above_relations[f_low] = f_high
                    all_floors.add(f_high)
                    all_floors.add(f_low)
                    floors_above.add(f_high)
                    floors_below.add(f_low)

            # Find the lowest floor (appears as f_low but never as f_high)
            lowest_floor = None
            potential_lowest = floors_below - floors_above # Floors that are below something but nothing is below them
            if len(potential_lowest) == 1:
                 lowest_floor = list(potential_lowest)[0]
            elif len(all_floors) == 1: # Case with only one floor
                 lowest_floor = list(all_floors)[0]
            # If len(potential_lowest) is not 1 and len(all_floors) > 1, the floor structure is ambiguous.
            # The heuristic will likely return inf later if required floors cannot be indexed.

            # Build the ordered list and index map starting from the lowest floor
            if lowest_floor:
                current_floor = lowest_floor
                index = 0
                # Loop while we are on a known floor and haven't processed all floors
                while current_floor in all_floors and len(self.floor_to_index) < len(all_floors):
                    self.floor_to_index[current_floor] = index
                    index += 1
                    # Find the floor immediately above the current one
                    next_floor = above_relations.get(current_floor)
                    if next_floor is None:
                        # Reached the highest floor (a floor that is not 'below' any other floor)
                        break
                    current_floor = next_floor
            # If lowest_floor was not found or chain was broken, self.floor_to_index might be incomplete or empty.
            # This will be handled in __call__ by returning inf if required floors aren't indexed.


        # Store mapping from passenger to destination floor (static)
        self.passenger_destin = {}
        for fact in static_facts:
             parts = get_parts(fact)
             if parts and parts[0] == 'destin' and len(parts) == 3:
                 passenger, floor = parts[1], parts[2]
                 self.passenger_destin[passenger] = floor


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached (all goal passengers served)
        current_served = {parts[1] for fact in state if match(fact, 'served', '*')}
        if self.goal_passengers.issubset(current_served):
             return 0

        # Find current lift location
        current_lift_f = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'lift-at' and len(parts) == 2:
                current_lift_f = parts[1]
                break

        # If lift location isn't found or floor indexing failed for current location, return infinity
        if current_lift_f is None or current_lift_f not in self.floor_to_index:
             return float('inf')

        current_index = self.floor_to_index[current_lift_f]

        # Identify unserved passengers and their status/location
        unserved_passengers = self.goal_passengers - current_served

        # If no unserved goal passengers, but not a goal state, something is wrong.
        # Assuming goal is only serving passengers.
        if not unserved_passengers:
             return 0 # Should be a goal state based on passenger service

        waiting_locations = set() # Floors where unserved passengers are waiting
        onboard_passengers = set() # Unserved passengers currently boarded
        onboard_destinations = set() # Destinations of onboard unserved passengers

        num_waiting_passengers = 0
        num_boarded_passengers = 0

        # Single pass through state facts to find status and locations of unserved passengers
        for fact in state:
             parts = get_parts(fact)
             if parts and len(parts) >= 2:
                  predicate = parts[0]
                  person = parts[1]
                  if person in unserved_passengers:
                       if predicate == 'origin' and len(parts) == 3:
                            floor = parts[2]
                            num_waiting_passengers += 1
                            waiting_locations.add(floor)
                       elif predicate == 'boarded' and len(parts) == 2:
                            num_boarded_passengers += 1
                            onboard_passengers.add(person) # Keep track of who is boarded

        # Add destinations for onboard passengers to required_floors
        for p in onboard_passengers:
             dest_f = self.passenger_destin.get(p)
             if dest_f: # Ensure destination exists
                 onboard_destinations.add(dest_f)
             # else: onboard passenger with no destination in static facts? Unsolvable?


        # Heuristic calculation
        h = 0

        # 1. Add cost for board/depart actions
        # Each waiting passenger needs 1 board + 1 depart action = 2 actions.
        # Each boarded passenger needs 1 depart action = 1 action.
        h += (num_waiting_passengers * 2) + (num_boarded_passengers * 1)


        # 2. Add estimated movement cost
        # Floors the lift must visit: origins of waiting passengers + destinations of boarded passengers
        required_floors = waiting_locations | onboard_destinations

        if not required_floors:
            # If no required floors, all unserved passengers must be boarded
            # and their destination is the current floor. They just need to depart.
            # Movement cost is 0. This case is covered by the initial goal check.
            movement_cost = 0
        else:
            # Ensure all required floors are in our index map
            required_indices = set()
            for f in required_floors:
                 if f in self.floor_to_index:
                      required_indices.add(self.floor_to_index[f])
                 else:
                      # Required floor not in index map - indicates problem with floor parsing or domain
                      return float('inf')


            min_req_index = min(required_indices)
            max_req_index = max(required_indices)

            # Estimated moves to cover the range of required floors starting from current floor
            # Go from current_index to either min_req or max_req, then sweep the range.
            dist_to_min = abs(current_index - min_req_index)
            dist_to_max = abs(current_index - max_req_index)
            range_dist = max_req_index - min_req_index

            movement_cost = min(dist_to_min, dist_to_max) + range_dist

            h += movement_cost

        return h
