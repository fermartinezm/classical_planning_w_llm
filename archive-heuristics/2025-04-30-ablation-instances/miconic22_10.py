import re
from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
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


# Assume Heuristic base class is available from heuristics.heuristic_base
# If not, replace 'Heuristic' with 'object'
# from heuristics.heuristic_base import Heuristic

class miconicHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers.
    It counts the required board and depart actions for unserved passengers
    and adds an estimate of the vertical travel distance the lift must cover
    to visit all necessary floors (passenger origins for pickup, passenger
    destinations for dropoff).

    # Assumptions
    - Floors are ordered linearly (e.g., f1, f2, f3...). The `above` predicates
      in the static facts define this order implicitly, specifically the
      adjacent `(above fi fj)` facts where j is the next floor number after i.
      The heuristic assumes floor names are 'f' followed by a number and sorts
      them numerically to determine levels.
    - All passengers relevant to the problem have a defined destination in the static facts.
    - The cost of each action (board, depart, up, down) is 1.
    - An unserved, unboarded passenger is always at their origin floor, and this
      fact `(origin p f)` is present in the state.

    # Heuristic Initialization
    - Extract the destination floor for each passenger from the static facts.
    - Determine the linear order of floors and create a mapping from floor
      name to its level (integer index). This is done by finding all floor
      names mentioned in `above` predicates and sorting them numerically.
    - Store the set of all passengers found.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Current State Information:**
        - Get the current state `state`.
        - Find the current floor of the lift (`lift-at ?f`).

    2.  **Identify Unserved Passengers:**
        - Iterate through all known passengers (extracted during initialization).
        - A passenger `p` is unserved if `(served p)` is not in the `state`.

    3.  **Categorize Unserved Passengers:**
        - For each unserved passenger `p`:
            - If `(boarded p)` is in the `state`, categorize as unserved and boarded.
            - Otherwise (if not served and not boarded), categorize as unserved and unboarded.
              (By domain logic, this implies `(origin p f)` is true for some floor `f`).

    4.  **Calculate Action Cost:**
        - Count the number of unserved, unboarded passengers (`N_unboarded_unserved`). Each needs a `board` action.
        - Count the total number of unserved passengers (`N_unserved`). Each needs a `depart` action.
        - The action cost estimate is `N_unboarded_unserved + N_unserved`.

    5.  **Identify Floors Requiring a Visit:**
        - For each unserved, unboarded passenger `p`, their origin floor `origin(p)` must be visited for pickup.
        - For each unserved, boarded passenger `p`, their destination floor `destin(p)` must be visited for dropoff.
        - Collect all such unique origin and destination floors into a set `floors_to_visit`.

    6.  **Calculate Travel Cost:**
        - If `floors_to_visit` is empty (meaning all passengers are served), the travel cost is 0.
        - If `floors_to_visit` is not empty:
            - Find the minimum and maximum floor levels among the floors in `floors_to_visit` using the pre-calculated `floor_levels` map.
            - Get the level of the current lift floor.
            - The travel cost is estimated as the minimum vertical distance the lift must travel to visit all floors in the required range `[min_level_to_visit, max_level_to_visit]` starting from the `current_level`. This is calculated as `min(abs(current_level - min_level_to_visit), abs(current_level - max_level_to_visit)) + (max_level_to_visit - min_level_to_visit)`.

    7.  **Total Heuristic:**
        - The total heuristic value is the sum of the calculated action cost and travel cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information."""
        # super().__init__(task) # Uncomment if inheriting from Heuristic base class
        self.goals = task.goals
        static_facts = task.static

        # Extract passenger destinations and collect all passenger names
        self.pass_destin = {}
        all_passengers_set = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "destin":
                passenger, floor = parts[1], parts[2]
                self.pass_destin[passenger] = floor
                all_passengers_set.add(passenger)

        # Store the set of all passengers found (relying on destin facts defining all relevant passengers)
        self.all_passengers = frozenset(all_passengers_set)

        # Extract floor order and create level mapping
        floor_names = set()
        for fact in static_facts:
             parts = get_parts(fact)
             # Consider floors mentioned in 'above' predicates as they define the structure
             if parts[0] == "above":
                 floor_names.add(parts[1])
                 floor_names.add(parts[2])

        # Sort floors numerically (e.g., f1, f2, ..., f10)
        # Use a regex to extract the number part for sorting
        def sort_key(floor_name):
            match = re.match(r'f(\d+)', floor_name)
            if match:
                return int(match.group(1))
            return floor_name # Fallback for unexpected floor names

        sorted_floors = sorted(list(floor_names), key=sort_key)

        # Create a map from floor name to its level (0-indexed)
        self.floor_levels = {floor: level for level, floor in enumerate(sorted_floors)}

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # If the goal is reached, the heuristic is 0.
        if self.goals <= state:
            return 0

        # Identify current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break
        # Assuming current_lift_floor is always found in a valid state

        # Identify unserved passengers and their status
        unserved_passengers = {p for p in self.all_passengers if f"(served {p})" not in state}

        unboarded_unserved = set()
        boarded_unserved = set()

        # Find origin/boarded status for unserved passengers
        for p in unserved_passengers:
            if f"(boarded {p})" in state:
                boarded_unserved.add(p)
            else:
                 # If not served and not boarded, they must be at their origin floor.
                 # The fact (origin p f) must be in the state.
                 unboarded_unserved.add(p)

        # Calculate action cost (board + depart)
        # Each unboarded passenger needs 1 board action.
        # Each unserved passenger needs 1 depart action.
        action_cost = len(unboarded_unserved) + len(unserved_passengers)

        # Identify floors that need visiting for pickup or dropoff
        required_pickup_floors = set()
        for p in unboarded_unserved:
             # Find the origin floor for this unboarded passenger from the state
             origin_floor = self._get_origin_floor(p, state)
             if origin_floor: # Should always be found for unboarded_unserved in valid states
                 required_pickup_floors.add(origin_floor)

        required_dropoff_floors = {self.pass_destin[p] for p in boarded_unserved}

        floors_to_visit = required_pickup_floors | required_dropoff_floors

        # Calculate travel cost
        travel_cost = 0
        if floors_to_visit:
            min_level_to_visit = min(self.floor_levels[f] for f in floors_to_visit)
            max_level_to_visit = max(self.floor_levels[f] for f in floors_to_visit)
            current_level = self.floor_levels[current_lift_floor]

            # Estimate travel as minimum moves to cover the range [min_level, max_level]
            # starting from current_level.
            # This is distance to closest end + distance between ends.
            dist_to_min = abs(current_level - min_level_to_visit)
            dist_to_max = abs(current_level - max_level_to_visit)
            range_dist = max_level_to_visit - min_level_to_visit

            travel_cost = min(dist_to_min, dist_to_max) + range_dist


        return action_cost + travel_cost

    def _get_origin_floor(self, passenger, state):
        """Helper to find the origin floor for a passenger in the current state."""
        # This is called only for unboarded_unserved passengers.
        # By domain definition, if unserved and unboarded, the fact (origin p f) must be true in the state.
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "origin" and parts[1] == passenger:
                 return parts[2]
        # This point should ideally not be reached in a valid state for unboarded_unserved passengers.
        # Returning None indicates an unexpected state or domain interpretation issue.
        return None # Should not happen for unboarded_unserved passengers
