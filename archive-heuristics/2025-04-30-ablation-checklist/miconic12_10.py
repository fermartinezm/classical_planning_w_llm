import re
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    # A simpler check is just to zip and check all match, fnmatch handles length differences
    # if the pattern is shorter than the fact parts, it will only check the first parts.
    # If the pattern is longer, zip will stop at the shortest.
    # We need to ensure the predicate matches and then the arguments match up to the pattern length.
    if not parts or len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    passengers. It counts the number of board and depart actions needed
    and adds an estimate of the lift movement cost.

    # Assumptions
    - Floors are named 'f' followed by a number (e.g., f1, f2, f10) and
      these numbers correspond to their vertical order (f1 is below f2, etc.).
    - Each unserved passenger needs one board action (if not already boarded)
      and one depart action.
    - The lift movement cost is estimated based on the range of floors
      that need to be visited (origins of unboarded passengers and
      destinations of all unserved passengers).

    # Heuristic Initialization
    - Extract static information: passenger origins, destinations, and
      the mapping of floor names to numerical levels.

    # Step-by-Step Thinking for Computing the Heuristic Value
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Unserved Passengers:** Determine which passengers are not yet
        in the `(served ?p)` state.
    2.  **Handle Goal State:** If all passengers are served, the heuristic is 0.
    3.  **Categorize Unserved Passengers:** For each unserved passenger, determine
        if they are waiting at their origin (`(origin ?p ?f)`) or are boarded
        (`(boarded ?p)`).
    4.  **Count Board/Depart Actions:**
        - Each passenger waiting at their origin needs a `board` action.
        - Each unserved passenger (whether waiting or boarded) needs a `depart` action.
        - The total non-movement cost is the sum of needed `board` and `depart` actions.
    5.  **Identify Required Floors:** Determine the set of floors the lift must visit:
        - The origin floor for every passenger still waiting at their origin.
        - The destination floor for every passenger who is currently boarded.
        - *Correction/Refinement:* The lift must visit the origin for unboarded passengers *and* the destination for *all* unserved passengers (as they eventually need to be dropped off).
        - So, Required Floors = {origin of p | p is unboarded} U {destination of p | p is unserved}.
    6.  **Map Floors to Levels:** Use the pre-calculated mapping from floor names
        (e.g., 'f5') to numerical levels (e.g., 5).
    7.  **Calculate Movement Cost:**
        - Find the lift's current floor and its level.
        - Find the minimum and maximum levels among the required floors.
        - Estimate the movement cost as the distance from the current level to the
          furthest required level, plus the distance between the minimum and
          maximum required levels. This estimates the cost to reach the range
          of required floors and traverse that range.
          Movement = `max(abs(current_level - min_required_level), abs(current_level - max_required_level)) + (max_required_level - min_required_level)`.
    8.  **Sum Costs:** The total heuristic value is the sum of the board actions,
        depart actions, and the estimated movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Floor levels mapping.
        - Passenger origins and destinations.
        - List of all passengers.
        """
        self.goals = task.goals  # Goal conditions (served passengers).
        static_facts = task.static  # Facts that are not affected by actions.

        self.floor_levels = {}
        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.all_passengers = set()

        # Extract floor names and create level mapping (assuming f1 < f2 < ...).
        # We can find floor names from origin/destin/above predicates in static facts.
        floor_names = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["origin", "destin"]:
                # Fact is (origin passenger floor) or (destin passenger floor)
                passenger_name = parts[1]
                floor_name = parts[2]
                self.all_passengers.add(passenger_name)
                floor_names.add(floor_name)
                if parts[0] == "origin":
                    self.passenger_origins[passenger_name] = floor_name
                else: # parts[0] == "destin"
                    self.passenger_destinations[passenger_name] = floor_name
            elif parts[0] == "above":
                 # Fact is (above floor1 floor2)
                 floor_names.add(parts[1])
                 floor_names.add(parts[2])

        # Sort floor names numerically to create the level mapping.
        # Assumes floor names are like 'f1', 'f10', 'f2'.
        sorted_floor_names = sorted(list(floor_names), key=lambda f: int(f[1:]))
        for level, floor_name in enumerate(sorted_floor_names, 1):
            self.floor_levels[floor_name] = level

        # Ensure we have origins and destinations for all passengers mentioned in goals
        # (Sometimes passengers only appear in goals, not init origin/destin in static)
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == "served":
                 self.all_passengers.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        unserved_passengers = set()
        unboarded_passengers = set()
        boarded_passengers = set()

        # Identify unserved passengers and their status (unboarded or boarded)
        served_passengers_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        for passenger in self.all_passengers:
            if passenger not in served_passengers_in_state:
                unserved_passengers.add(passenger)
                if match(f"(origin {passenger} *)", state):
                     unboarded_passengers.add(passenger)
                elif match(f"(boarded {passenger})", state):
                     boarded_passengers.add(passenger)
                # Note: A passenger should be either origin or boarded if unserved.
                # If neither, something is wrong with the state representation or domain.
                # We assume the state is valid.

        # If all passengers are served, the heuristic is 0.
        if not unserved_passengers:
            return 0

        # --- Calculate non-movement costs (board and depart actions) ---
        # Each unboarded passenger needs one board action.
        board_actions_needed = len(unboarded_passengers)
        # Each unserved passenger needs one depart action eventually.
        depart_actions_needed = len(unserved_passengers)

        # --- Identify required floors and calculate movement cost ---
        required_floors = set()

        # Origins of passengers still waiting to be picked up
        for passenger in unboarded_passengers:
             origin_floor = self.passenger_origins.get(passenger)
             if origin_floor: # Should always exist for unboarded passengers
                 required_floors.add(origin_floor)

        # Destinations of all unserved passengers (both unboarded and boarded)
        for passenger in unserved_passengers:
             destin_floor = self.passenger_destinations.get(passenger)
             if destin_floor: # Should always exist for unserved passengers
                 required_floors.add(destin_floor)


        # Find the lift's current floor
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break # Assuming only one lift-at fact

        if current_floor is None:
             # This should not happen in a valid state, but handle defensively
             # If lift location is unknown, we can't estimate movement.
             # Fallback to just board/depart count or a large value.
             # Let's return board+depart count as a lower bound.
             return board_actions_needed + depart_actions_needed


        current_level = self.floor_levels.get(current_floor)
        if current_level is None:
             # Should not happen if floor_levels is built correctly
             return board_actions_needed + depart_actions_needed


        required_levels = {self.floor_levels[f] for f in required_floors if f in self.floor_levels}

        movement_cost = 0
        if required_levels:
            min_req_level = min(required_levels)
            max_req_level = max(required_levels)

            # Estimate movement: distance to the furthest required floor + distance to sweep the range
            dist_to_min = abs(current_level - min_req_level)
            dist_to_max = abs(current_level - max_req_level)
            range_dist = max_req_level - min_req_level

            # The lift must travel from current_level to cover the range [min_req_level, max_req_level].
            # A reasonable estimate is the distance to the nearest end of the range plus the range itself,
            # OR the distance to the furthest end of the range plus the range itself.
            # The latter (distance to furthest + range) seems to capture the necessary travel better
            # for a non-admissible heuristic aiming to reduce node expansions.
            movement_cost = max(dist_to_min, dist_to_max) + range_dist


        # Total heuristic is the sum of actions needed
        total_cost = board_actions_needed + depart_actions_needed + movement_cost

        return total_cost

    # Helper function to check if a fact matches a pattern in a set of facts
    def match(self, pattern_fact_str, fact_set):
        """
        Checks if any fact in the fact_set matches the pattern_fact_str.
        pattern_fact_str: string like "(predicate arg1 *)"
        fact_set: frozenset of fact strings
        """
        pattern_parts = get_parts(pattern_fact_str)
        for fact_str in fact_set:
            if match(fact_str, *pattern_parts):
                return True
        return False

