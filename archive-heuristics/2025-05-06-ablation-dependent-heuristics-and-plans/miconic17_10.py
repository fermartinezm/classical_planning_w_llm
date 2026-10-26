from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers.
    It sums three components:
    1. Minimum lift moves required to visit all floors where pickups or dropoffs are needed.
    2. Number of board actions needed (one for each passenger waiting at their origin).
    3. Number of depart actions needed (one for each unserved passenger).

    # Assumptions
    - Floors are ordered linearly. The 'above' predicates define this order.
    - The lift can carry multiple passengers.
    - The cost of each action (move, board, depart) is 1.

    # Heuristic Initialization
    - Parse 'above' facts from static information to determine the floor order and assign numerical levels to floors. The floor that is 'above' the most other floors is considered the lowest (level 1).
    - Parse 'destin' facts from static information to map each passenger to their destination floor.
    - Identify all passengers defined in the problem instance (those with a destination).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the lift's current floor from the state fact '(lift-at <f>)'.
    2. Identify all unserved passengers. A passenger is unserved if they are known to the problem (have a destination) but the fact '(served <passenger>)' is not in the state.
    3. Determine the set of 'event floors' that the lift must visit:
       - Add the origin floor for every passenger currently waiting at their origin ('(origin <p> <f>)') as a pickup event floor.
       - Add the destination floor for every unserved passenger (retrieved from the static 'destin' facts) as a dropoff event floor.
    4. Calculate the minimum number of moves required for the lift to visit all 'event floors' starting from its current floor.
       - If there are no event floors, 0 moves are needed.
       - Otherwise, find the minimum and maximum floor levels among the event floors.
       - The minimum moves is calculated as the distance between the minimum and maximum event floor levels plus the minimum distance from the current floor level to either the minimum or maximum event floor level. This represents the travel to reach the required range and sweep through it.
    5. Count the number of 'board' actions needed. This is equal to the number of passengers currently waiting at their origin ('(origin <p> <f>)'). Each such passenger requires one 'board' action.
    6. Count the number of 'depart' actions needed. This is equal to the total number of unserved passengers. Each unserved passenger requires one 'depart' action at their destination.
    7. The total heuristic value is the sum of the minimum moves, the needed board actions, and the needed depart actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        # self.goals = task.goals # Not directly used in this heuristic
        static_facts = task.static

        # 1. Map floors to numerical levels based on 'above' facts.
        floor_above_counts = {}
        all_floors = set()
        # First pass to collect all floors and initialize counts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                f_lower, f_higher = parts[1], parts[2]
                all_floors.add(f_lower)
                all_floors.add(f_higher)
                # Initialize counts for all floors found
                floor_above_counts[f_lower] = floor_above_counts.get(f_lower, 0)
                floor_above_counts[f_higher] = floor_above_counts.get(f_higher, 0)

        # Second pass to populate counts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                f_lower, f_higher = parts[1], parts[2]
                floor_above_counts[f_lower] += 1 # f_lower is above f_higher

        # Sort floors by the number of floors they are above (descending)
        # The floor above the most others is the lowest.
        sorted_floors = sorted(list(all_floors), key=lambda f: floor_above_counts.get(f, 0), reverse=True)

        self.floor_levels = {floor: i + 1 for i, floor in enumerate(sorted_floors)}

        # Handle case with 0 or 1 floor (no 'above' facts)
        if not self.floor_levels and all_floors:
             self.floor_levels = {list(all_floors)[0]: 1}


        # 2. Map passengers to their destination floors and identify all passengers.
        self.destin_map = {}
        self.all_passengers = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "destin":
                p, d = parts[1], parts[2]
                self.destin_map[p] = d
                self.all_passengers.add(p)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Find lift's current floor.
        lift_at_fact = next((fact for fact in state if get_parts(fact) and get_parts(fact)[0] == "lift-at"), None)
        if lift_at_fact is None:
             # Should not happen in a valid miconic state
             return float('inf')

        f_current = get_parts(lift_at_fact)[1]
        current_level = self.floor_levels.get(f_current)
        if current_level is None:
             # Should not happen if floor_levels is built correctly
             return float('inf')


        # 2. Identify unserved passengers.
        served_passengers = {get_parts(fact)[1] for fact in state if get_parts(fact) and get_parts(fact)[0] == "served"}
        unserved_passengers = {p for p in self.all_passengers if p not in served_passengers}

        # If all passengers are served, the goal is reached.
        if not unserved_passengers:
            return 0

        # 3. Determine 'event floors'.
        pickup_events = set()
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == "origin":
                  p, o = parts[1], parts[2]
                  pickup_events.add(o)

        dropoff_events = {self.destin_map[p] for p in unserved_passengers if p in self.destin_map}

        all_event_floors = pickup_events.union(dropoff_events)

        # If there are no event floors but unserved passengers,
        # it implies all unserved passengers are currently boarded
        # and the lift is already at their destination floors.
        # The only remaining actions are 'depart'.
        if not all_event_floors:
             # Moves = 0. Boards = 0. Departs = number of unserved passengers.
             return len(unserved_passengers)


        # 4. Calculate minimum moves.
        event_levels = [self.floor_levels[f] for f in all_event_floors if f in self.floor_levels]
        if not event_levels:
             # Should not happen if all_event_floors is not empty and floor_levels is correct
             return float('inf')

        min_level = min(event_levels)
        max_level = max(event_levels)

        # Minimum moves to visit all floors in [min_level, max_level] starting from current_level
        # is distance between ends + distance from current to closer end.
        moves = (max_level - min_level) + min(abs(current_level - min_level), abs(current_level - max_level))


        # 5. Count board actions needed.
        # This is the number of passengers currently waiting at their origin.
        board_actions_needed = len(pickup_events)


        # 6. Count depart actions needed.
        # This is the number of unserved passengers.
        depart_actions_needed = len(unserved_passengers)


        # 7. Total heuristic value.
        total_cost = moves + board_actions_needed + depart_actions_needed

        return total_cost
