from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args,
    # and if each part matches the corresponding pattern.
    # fnmatch allows wildcards like '*'
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        This heuristic estimates the remaining cost to serve all passengers
        by summing up the estimated costs for each individual unserved passenger.
        The estimated cost for a passenger depends on whether they are waiting
        at their origin or are already boarded. Movement costs are estimated
        based on the absolute difference in floor indices. This heuristic is
        non-admissible as it sums costs independently, potentially overcounting
        movement costs when multiple passengers share floors.

    Assumptions:
        - The PDDL domain is 'miconic' as provided.
        - Floors are linearly ordered by the 'above' predicate, forming a chain.
        - Floor names are parseable (e.g., 'f1', 'f2', ...).
        - Each passenger has a unique origin and destination defined in static facts.
        - The goal is to serve all passengers specified in the initial state.
        - All floors mentioned in the problem (initial state, goals, static facts)
          are part of the linear order defined by 'above' facts.

    Heuristic Initialization:
        - Parses static facts to determine the linear order of floors and creates
          a mapping from floor names to integer indices (lowest floor gets index 0).
        - Parses static facts to store the destination floor for each passenger.
        - Identifies all passengers present in the problem instance based on
          'origin' and 'destin' facts.

    Step-By-Step Thinking for Computing Heuristic:
        1. Identify the current floor of the lift from the state.
        2. Identify all passengers in the problem instance (pre-computed during init).
        3. Determine which passengers are already 'served' by checking the state.
        4. Initialize the total heuristic cost to 0.
        5. For each passenger who is NOT served:
            a. Find their destination floor using the pre-computed destination map.
            b. Check if the passenger is at their origin floor by looking for
               '(origin ?p ?f)' facts in the current state. If found, get the origin floor.
            c. Check if the passenger is 'boarded' by looking for '(boarded ?p)'
               facts in the current state.
            d. If the passenger is at their origin:
                - Add 2 to the cost (for the 'board' and 'depart' actions).
                - Add the absolute difference in floor indices between the current
                  lift floor and the passenger's origin floor (estimated moves to pickup).
                - Add the absolute difference in floor indices between the passenger's
                  origin floor and their destination floor (estimated moves while boarded).
            e. If the passenger is boarded:
                - Add 1 to the cost (for the 'depart' action).
                - Add the absolute difference in floor indices between the current
                  lift floor and the passenger's destination floor (estimated moves to dropoff).
            f. A passenger should be either at their origin or boarded if not served
               in a valid state sequence for this domain. If a passenger is unserved
               but neither at origin nor boarded, this heuristic assumes they are
               effectively "lost" or the state is invalid, and their contribution
               is implicitly 0 (they won't be found in origin or boarded checks).
               However, in valid problem instances and states, this case should not occur.
        6. Return the total accumulated cost.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # 1. Map floors to indices
        # Find all floors mentioned in 'above' facts
        above_facts = [fact for fact in static_facts if match(fact, "above", "*", "*")]
        all_floors_in_above = set()
        floor_below_of = {} # Maps a floor to the floor immediately below it
        for fact in above_facts:
            _, f_above, f_below = get_parts(fact)
            all_floors_in_above.add(f_above)
            all_floors_in_above.add(f_below)
            floor_below_of[f_above] = f_below

        # Find the highest floor (a floor in all_floors_in_above that is not a value in floor_below_of)
        highest_floor = None
        floors_that_are_below_others = set(floor_below_of.values())
        for floor in all_floors_in_above:
            if floor not in floors_that_are_below_others:
                 # This floor is not below any other floor mentioned in 'above', so it must be the highest
                 highest_floor = floor
                 break

        # Build ordered list of floors from highest to lowest
        floors_ordered_desc = []
        current_floor = highest_floor
        while current_floor is not None:
            floors_ordered_desc.append(current_floor)
            current_floor = floor_below_of.get(current_floor)

        # Reverse to get lowest first (index 0)
        floors_ordered_asc = list(reversed(floors_ordered_desc))
        self.floor_to_idx = {f: i for i, f in enumerate(floors_ordered_asc)}

        # 2. Store passenger destinations and identify all passengers
        self.destin_map = {}
        self.all_passengers = set()
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destin_map[passenger] = floor
                self.all_passengers.add(passenger)
            elif match(fact, "origin", "*", "*"):
                 _, passenger, floor = get_parts(fact)
                 self.all_passengers.add(passenger)

        # Ensure all floors mentioned in destinations are in our floor map
        # (This check is for robustness against malformed problems)
        for passenger, floor in self.destin_map.items():
             if floor not in self.floor_to_idx:
                 # This problem instance has a destination floor not in the 'above' chain.
                 # The heuristic might be inaccurate or fail.
                 # For this implementation, we assume valid problems where all floors are ordered.
                 pass # In a real scenario, might raise an error or handle differently


    def __call__(self, node):
        state = node.state
        h_value = 0

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        if current_lift_floor is None or current_lift_floor not in self.floor_to_idx:
             # Lift location is missing or on an unknown floor - invalid state
             return float('inf') # Return a high value

        current_lift_idx = self.floor_to_idx[current_lift_floor]

        # Identify served passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # Identify passengers at origin and boarded
        passengers_at_origin = {} # {passenger: floor}
        boarded_passengers = set()
        for fact in state:
            if match(fact, "origin", "*", "*"):
                p, f = get_parts(fact)[1:]
                passengers_at_origin[p] = f
            elif match(fact, "boarded", "*"):
                p = get_parts(fact)[1]
                boarded_passengers.add(p)

        # Calculate heuristic for each unserved passenger
        for passenger in self.all_passengers:
            if passenger not in served_passengers:
                dest_floor = self.destin_map.get(passenger)
                if dest_floor is None or dest_floor not in self.floor_to_idx:
                    # Passenger has no destination defined or destination floor is unknown
                    # Treat as unplannable for this passenger, return high heuristic?
                    # Or just skip? Skipping assumes they don't need serving, which contradicts all_passengers set.
                    # Let's return inf if a required destination is on an unknown floor.
                    if dest_floor is None: # Should not happen based on problem structure
                         return float('inf')
                    if dest_floor not in self.floor_to_idx:
                         # Destination floor is not in the ordered list derived from 'above' facts
                         return float('inf')

                dest_idx = self.floor_to_idx[dest_floor]

                if passenger in passengers_at_origin:
                    origin_floor = passengers_at_origin[passenger]
                    if origin_floor not in self.floor_to_idx:
                         # Origin floor is not in the ordered list
                         return float('inf')
                    origin_idx = self.floor_to_idx[origin_floor]
                    # Cost = board (1) + depart (1) + moves (current->origin + origin->dest)
                    h_value += 2 + abs(current_lift_idx - origin_idx) + abs(origin_idx - dest_idx)
                elif passenger in boarded_passengers:
                    # Cost = depart (1) + moves (current->dest)
                    h_value += 1 + abs(current_lift_idx - dest_idx)
                # else: passenger is unserved but not at origin and not boarded.
                # This state should not be reachable in valid problem instances.
                # We add 0 cost for such passengers, effectively ignoring them,
                # but the check for served_passengers ensures h > 0 if any passenger
                # from all_passengers is not served.

        return h_value
