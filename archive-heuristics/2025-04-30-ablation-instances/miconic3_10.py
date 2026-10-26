# The required import for the base class. Assuming it's in a file named heuristic_base.py
from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, maybe log a warning or raise an error
        # Assuming valid PDDL facts as strings
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
    This heuristic estimates the number of actions required to transport all
    passengers to their destination floors. It counts the necessary board and
    depart actions and adds an estimate for the lift's movement cost.

    # Assumptions
    - Passengers are initially waiting at their origin floors or already boarded.
    - Passengers must be boarded at their origin and depart at their destination.
    - The lift moves one floor at a time.
    - All passengers have a defined origin and destination in the static facts.
    - The 'above' predicates define a total order on floors.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the static facts.
    - Builds a mapping from floor names (e.g., 'f1', 'f2') to numerical floor
      levels based on the 'above' predicates in the static facts. This allows
      calculating distances between floors.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current floor of the lift.
    2. Categorize all passengers who are not yet 'served':
       - Waiting passengers: Those with an '(origin ?p ?f)' fact.
       - Boarded passengers: Those with a '(boarded ?p)' fact.
    3. Calculate the non-movement cost:
       - Each waiting passenger needs a 'board' action (cost 1) and a 'depart'
         action (cost 1). Total 2 actions per waiting passenger.
       - Each boarded passenger needs a 'depart' action (cost 1). Total 1 action
         per boarded passenger.
       - Total non-movement cost = (2 * number of waiting passengers) +
         (number of boarded passengers).
    4. Identify the set of floors the lift *must* visit to serve the remaining
       passengers:
       - The origin floor for every waiting passenger.
       - The destination floor for every boarded passenger.
    5. Calculate the movement cost:
       - Find the minimum and maximum floor numbers among the required floors.
       - Find the numerical level of the lift's current floor.
       - The minimum movement cost to visit all required floors starting from
         the current floor is estimated as the distance from the current floor
         to the closest extreme required floor (min or max) plus the total span
         of the required floors (max - min).
         Movement cost = min(|current_floor_num - min_req_num|,
                             |current_floor_num - max_req_num|) + (max_req_num - min_req_num).
         If there are no required floors, the movement cost is 0.
    6. The total heuristic value is the sum of the non-movement cost and the
       movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor mapping and destinations.
        """
        self.goals = task.goals # Store goals, though not directly used in calculation
        static_facts = task.static

        # Extract passenger destinations from static facts
        self.destinations = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "destin":
                person, floor = parts[1], parts[2]
                self.destinations[person] = floor

        # Build floor name to number mapping from 'above' facts
        self.floor_to_num = {}
        self.num_to_floor = {}
        above_pairs = []
        floors_that_are_above = set() # Floors that appear as the first argument in (above f_above f_below)
        floors_that_are_below = set() # Floors that appear as the second argument in (above f_above f_below)

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                f_above, f_below = parts[1], parts[2]
                above_pairs.append((f_above, f_below))
                floors_that_are_above.add(f_above)
                floors_that_are_below.add(f_below)

        if not above_pairs:
             # Handle case with no floors or no 'above' facts
             # This heuristic relies on floor ordering, so it might not be meaningful
             # in such cases. Return 0 or handle appropriately.
             # Assuming valid miconic problems have floors and 'above' facts.
             return

        # Build map from floor_below to floor_above
        floor_above_map = {f_below: f_above for f_above, f_below in above_pairs}

        # Find the lowest floor (a floor that is below another, but nothing is below it)
        # It's a floor in floors_that_are_below but not in floors_that_are_above
        lowest_floor_set = floors_that_are_below - floors_that_are_above
        if not lowest_floor_set:
             # Handle case where no unique lowest floor is found (e.g., circular 'above' facts)
             # This indicates an invalid domain definition for this heuristic.
             # Return 0 or raise an error. Returning 0 as a fallback.
             return
        lowest_floor = lowest_floor_set.pop()


        # Build the mapping by iterating upwards from the lowest floor
        current_floor = lowest_floor
        current_num = 1

        # Use a safety break for malformed 'above' chains (e.g., loops)
        visited_floors_in_chain = set()
        while current_floor in floor_above_map:
             if current_floor in visited_floors_in_chain:
                 # Circular definition detected, break loop
                 break
             visited_floors_in_chain.add(current_floor)

             self.floor_to_num[current_floor] = current_num
             self.num_to_floor[current_num] = current_floor
             current_floor = floor_above_map[current_floor]
             current_num += 1

        # Add the highest floor which is the last one in the chain (if not already added)
        if current_floor not in self.floor_to_num:
             self.floor_to_num[current_floor] = current_num
             self.num_to_floor[current_num] = current_floor


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        current_floor = None
        waiting_passengers = {} # {person: origin_floor} based on state
        boarded_passengers = set() # {person} based on state
        served_passengers = set() # {person} based on state

        # Extract current state information
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "lift-at":
                current_floor = parts[1]
            elif predicate == "origin":
                person, floor = parts[1], parts[2]
                waiting_passengers[person] = floor
            elif predicate == "boarded":
                person = parts[1]
                boarded_passengers.add(person)
            elif predicate == "served":
                person = parts[1]
                served_passengers.add(person)

        # Identify unserved passengers
        # A passenger is unserved if they are not in the served_passengers set.
        # We can get all possible passenger names from the keys in self.destinations
        all_possible_passengers = set(self.destinations.keys())
        unserved_passengers = all_possible_passengers - served_passengers

        if not unserved_passengers:
            return 0 # Goal state

        # Filter waiting/boarded based on who is actually unserved
        waiting_passengers_unserved = {p: f for p, f in waiting_passengers.items() if p in unserved_passengers}
        boarded_passengers_unserved = boarded_passengers.intersection(unserved_passengers)


        # Calculate non-movement cost (board and depart actions)
        # Each waiting passenger needs 1 board + 1 depart = 2 actions
        # Each boarded passenger needs 1 depart = 1 action
        num_waiting_unserved = len(waiting_passengers_unserved)
        num_boarded_unserved = len(boarded_passengers_unserved)
        non_movement_cost = (2 * num_waiting_unserved) + num_boarded_unserved

        # Identify required floors to visit
        required_pickup_floors = set(waiting_passengers_unserved.values())
        required_dropoff_floors = {self.destinations[p] for p in boarded_passengers_unserved if p in self.destinations} # Ensure destination exists

        required_floors = required_pickup_floors | required_dropoff_floors

        # Calculate movement cost
        movement_cost = 0
        if required_floors:
            required_floor_nums = {self.floor_to_num[f] for f in required_floors if f in self.floor_to_num} # Ensure floor is mapped
            if required_floor_nums: # Check again in case required_floors had unmapped names
                min_req_num = min(required_floor_nums)
                max_req_num = max(required_floor_nums)
                current_f_num = self.floor_to_num.get(current_floor, None) # Use .get for safety

                if current_f_num is not None: # Ensure current floor is mapped
                     # Minimum travel distance to visit all points in a range [min_req_num, max_req_num] starting from current_f_num
                     movement_cost = min(abs(current_f_num - min_req_num), abs(current_f_num - max_req_num)) + (max_req_num - min_req_num)
                else:
                     # Should not happen in valid states, but handle defensively
                     # Assume lift is at a floor that needs visiting or closest to the range
                     # A simple fallback could be max_req_num - min_req_num
                     movement_cost = max_req_num - min_req_num # Or a large penalty? Let's use the span.
            # else: required_floor_nums was empty, movement_cost remains 0

        # Total heuristic is sum of non-movement and movement costs
        total_cost = non_movement_cost + movement_cost

        return total_cost
