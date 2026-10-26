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
    # Ensure the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It counts the number of 'board' actions needed (for waiting passengers),
    the number of 'depart' actions needed (for all unserved passengers),
    and adds an estimate of the minimum number of 'up'/'down' moves required
    to visit all necessary floors (origins of waiting passengers and destinations
    of boarded passengers).

    # Assumptions
    - The lift has infinite capacity.
    - The floor structure defined by 'above' predicates forms a single linear sequence.
    - Passengers only need to be served once.

    # Heuristic Initialization
    - Build a map from floor names to integer indices based on the 'above' facts.
    - Store the destination floor for each passenger from the 'destin' facts.
    - Identify all passengers that need to be served based on the goal state.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For a given state:
    1. Identify the current floor of the lift.
    2. Identify which passengers are waiting at their origin, which are boarded,
       and which are already served.
    3. Determine the set of unserved passengers (those not yet 'served').
    4. If there are no unserved passengers, the heuristic is 0.
    5. Identify the set of 'required floors':
       - For each unserved passenger who is waiting at their origin, their origin floor is required.
       - For each unserved passenger who is boarded, their destination floor is required.
    6. Count the number of 'board' actions needed: This is the number of unserved
       passengers who are currently waiting at their origin.
    7. Count the number of 'depart' actions needed: This is the number of unserved
       passengers (each unserved passenger will eventually need one 'depart' action).
    8. Estimate the number of 'up'/'down' moves:
       - Find the minimum and maximum floor indices among the required floors.
       - Calculate the minimum moves required to travel from the current lift floor
         to cover the range of required floors. This is the distance from the current
         floor to the nearest end of the required floor range, plus the span of
         the required floor range itself.
         `moves = min(abs(current_idx - min_req_idx), abs(current_idx - max_req_idx)) + (max_req_idx - min_req_idx)`
    9. The total heuristic value is the sum of the estimated 'board' actions,
       estimated 'depart' actions, and estimated 'move' actions.
       `h = (num waiting unserved) + (num unserved) + estimated_moves`
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.

        @param task: The planning task object containing initial state, goals, and static facts.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Build floor order and index map from static 'above' facts
        self.floor_to_index = self._build_floor_map(self.static_facts)
        # self.index_to_floor = {v: k for k, v in self.floor_to_index.items()} # Not strictly needed for this heuristic

        # Store passenger destinations from static 'destin' facts
        self.passenger_destinations = {}
        for fact in self.static_facts:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.passenger_destinations[passenger] = floor

        # Store goal passengers (all passengers mentioned in goals need to be served)
        self.goal_passengers = set()
        for goal in self.goals:
             if match(goal, "served", "*"):
                 _, passenger = get_parts(goal)
                 self.goal_passengers.add(passenger)

    def _build_floor_map(self, static_facts):
        """
        Builds a map from floor name string to integer index (0-based)
        assuming a linear floor structure defined by 'above' facts.
        """
        above_map = {} # Maps floor_below -> floor_above
        all_floors = set()
        floors_above_set = set() # Floors that appear as the first arg (?f1) in (above ?f1 ?f2)
        floors_below_set = set() # Floors that appear as the second arg (?f2) in (above ?f1 ?f2)

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, floor_above, floor_below = get_parts(fact)
                above_map[floor_below] = floor_above
                all_floors.add(floor_above)
                all_floors.add(floor_below)
                floors_above_set.add(floor_above)
                floors_below_set.add(floor_below)

        if not all_floors:
             # Should not happen in valid miconic problems, but handle defensively
             return {}

        # The lowest floor is the one that appears as ?f2 but never as ?f1
        # If there's only one floor, it's the lowest.
        if len(all_floors) == 1:
             lowest_floor = list(all_floors)[0]
        else:
             # Find the floor that is in floors_below_set but not floors_above_set
             # This is the floor with nothing below it, i.e., the lowest.
             lowest_floor_candidates = floors_below_set - floors_above_set
             if not lowest_floor_candidates:
                 # This can happen if the floor structure is circular or disconnected,
                 # or if the lowest floor is not explicitly the second argument of any 'above' (e.g., f1 is lowest, (above f2 f1) exists, but no (above X f1) exists).
                 # A more robust way: find the floor that is never the first argument (highest)
                 # and the floor that is never the second argument (lowest).
                 highest_floor_candidates = floors_above_set - floors_below_set
                 if highest_floor_candidates:
                     # The lowest floor is the one that is not the second argument of any 'above' fact.
                     lowest_floor_candidates = all_floors - floors_below_set
                     if lowest_floor_candidates:
                         lowest_floor = lowest_floor_candidates.pop()
                     else:
                         # Fallback: just pick an arbitrary floor and hope for the best, or raise error
                         # Assuming valid PDDL, one of the above methods should work.
                         # Let's assume the simple linear structure holds and the lowest is in floors_below_set - floors_above_set
                         # or is the only floor.
                         # If that fails, the structure might be f1, f2, f3 where (above f2 f1), (above f3 f2).
                         # f1 is in floors_below_set, not floors_above_set. This logic seems correct.
                         lowest_floor = (floors_below_set - floors_above_set).pop()
                 else:
                      # Handle case where there are above facts but no clear lowest/highest (e.g., single floor with (above f1 f1) or disconnected)
                      # For linear structures, this shouldn't happen. Pick an arbitrary floor.
                      lowest_floor = list(all_floors)[0]


        # Reconstruct the ordered list of floors starting from the lowest
        floor_order = []
        current = lowest_floor
        while current in above_map:
            floor_order.append(current)
            current = above_map[current]
        floor_order.append(current) # Add the highest floor

        # Create the floor name to index map
        return {floor: i for i, floor in enumerate(floor_order)}


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.

        @param node: The search node containing the current state.
        @return: The estimated number of actions to reach a goal state.
        """
        state = node.state  # Current world state as a frozenset of strings

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break

        # If lift location is unknown, heuristic is infinite (invalid state)
        if current_lift_floor is None or current_lift_floor not in self.floor_to_index:
             return float('inf')

        current_lift_index = self.floor_to_index[current_lift_floor]

        # Identify passenger states
        waiting_passengers = set()
        boarded_passengers = set()
        served_passengers = set()

        for fact in state:
            if match(fact, "origin", "*", "*"):
                _, p, f = get_parts(fact)
                waiting_passengers.add(p)
            elif match(fact, "boarded", "*"):
                _, p = get_parts(fact)
                boarded_passengers.add(p)
            elif match(fact, "served", "*"):
                _, p = get_parts(fact)
                served_passengers.add(p)

        # Identify unserved passengers among those we care about (in goal)
        unserved_passengers = self.goal_passengers - served_passengers

        # If all goal passengers are served, heuristic is 0
        if not unserved_passengers:
            return 0

        # Determine required floors to visit
        required_floors = set()
        waiting_unserved = set()
        boarded_unserved = set()

        for p in unserved_passengers:
            if p in waiting_passengers:
                waiting_unserved.add(p)
                # Find origin floor from current state
                origin_floor = None
                for fact in state:
                    if match(fact, "origin", p, "*"):
                        _, _, origin_floor = get_parts(fact)
                        break
                if origin_floor:
                    required_floors.add(origin_floor)
            elif p in boarded_passengers:
                boarded_unserved.add(p)
                # Find destination floor from static facts
                destin_floor = self.passenger_destinations.get(p)
                if destin_floor:
                    required_floors.add(destin_floor)
            # Note: Passengers not in goal_passengers are ignored.
            # Passengers in goal_passengers but neither waiting nor boarded
            # (and not served) indicate an unexpected state based on domain actions.
            # We assume such states are not reachable or are handled by other parts of the planner.

        # If there are unserved passengers but no required floors, it means
        # all unserved passengers are boarded, and the lift is already at ALL their destinations.
        # The only remaining actions are the departs.
        if not required_floors:
             estimated_moves = 0
        else:
            # Calculate estimated moves to visit all required floors
            required_indices = [self.floor_to_index[f] for f in required_floors if f in self.floor_to_index]
            if not required_indices: # Should not happen if required_floors is not empty and floor_to_index is built correctly
                 estimated_moves = 0
            else:
                min_req_index = min(required_indices)
                max_req_index = max(required_indices)

                # Estimated moves = distance to nearest required floor + distance to cover the range
                # This assumes one sweep up or down covering all required floors.
                estimated_moves = min(abs(current_lift_index - min_req_index), abs(current_lift_index - max_req_index)) + (max_req_index - min_req_index)

        # Calculate estimated actions:
        # 1 action for each 'board' needed (one per waiting unserved passenger)
        # 1 action for each 'depart' needed (one per unserved passenger)
        # Estimated moves for lift travel
        num_board_actions = len(waiting_unserved)
        num_depart_actions = len(unserved_passengers) # Each unserved passenger needs a depart eventually

        total_heuristic = num_board_actions + num_depart_actions + estimated_moves

        return total_heuristic

