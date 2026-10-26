from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Helper function to split a PDDL fact string into its components."""
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Estimates the remaining cost to serve all passengers by summing
    the required passenger actions (board/depart) and an estimate
    of the necessary lift movement.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by pre-processing static information
        about floor levels and passenger destinations.

        Args:
            task: The planning task object containing initial state, goals,
                  operators, and static facts.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Also need initial state to find all passengers

        # --- Heuristic Initialization ---
        # 1. Map floors to numerical levels based on 'above' predicates.
        #    (above f_lower f_higher) means f_higher is directly above f_lower.
        above_map = {}
        all_floors = set()
        floors_with_lower = set() # Floors that appear as f_higher

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'above':
                f_lower, f_higher = parts[1], parts[2]
                above_map[f_lower] = f_higher
                all_floors.add(f_lower)
                all_floors.add(f_higher)
                floors_with_lower.add(f_higher)

        # Find the lowest floor (a floor that is not f_higher for any above fact)
        lowest_floor = None
        for floor in all_floors:
            if floor not in floors_with_lower:
                lowest_floor = floor
                break

        # Build floor_to_level map using BFS starting from the lowest floor
        self.floor_to_level = {}
        if lowest_floor:
            queue = deque([(lowest_floor, 0)])
            self.floor_to_level[lowest_floor] = 0

            while queue:
                current_floor, current_level = queue.popleft()
                next_floor = above_map.get(current_floor)
                if next_floor and next_floor not in self.floor_to_level:
                    self.floor_to_level[next_floor] = current_level + 1
                    queue.append((next_floor, current_level + 1))

        # 2. Map passengers to their destination floors.
        self.passenger_to_dest_floor = {}
        all_passengers = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'destin':
                passenger, floor = parts[1], parts[2]
                self.passenger_to_dest_floor[passenger] = floor
                all_passengers.add(passenger)
            # Also collect passengers from origin facts in static (if any)
            if parts[0] == 'origin':
                 all_passengers.add(parts[1])

        # Collect passengers from initial state facts as well
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] in ['origin', 'boarded', 'served']:
                 all_passengers.add(parts[1])

        self.all_passengers = frozenset(all_passengers)


    def __call__(self, node):
        """
        Computes the heuristic value for a given state node.

        Args:
            node: The state node for which to compute the heuristic.

        Returns:
            An integer estimate of the remaining actions to reach the goal.
        """
        state = node.state

        # Check if goal is reached (all passengers served)
        # The base Heuristic class might have a goal_reached method,
        # but we calculate unserved passengers anyway.
        served_passengers = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'served'}
        unserved_passengers = self.all_passengers - served_passengers

        # --- Step-By-Step Thinking for Computing Heuristic ---

        # If all passengers are served, the heuristic is 0.
        if not unserved_passengers:
            return 0

        # 1. Identify the current lift floor.
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at':
                current_lift_floor = parts[1]
                break

        # Should always find the lift location in a valid state
        if current_lift_floor is None:
             # This indicates an invalid state representation or task definition
             # Return infinity or a very large number to prune this path
             return float('inf') # Or sys.maxsize if sys is imported

        current_lift_level = self.floor_to_level.get(current_lift_floor)
        # Should always find the floor level
        if current_lift_level is None:
             return float('inf')


        # 2. Count required passenger actions (board and depart).
        #    Each waiting passenger needs a 'board' action.
        #    Each unserved passenger needs a 'depart' action.
        num_waiting = 0
        required_floors = set() # Floors the lift must visit

        for passenger in unserved_passengers:
            is_waiting = False
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == 'origin' and parts[1] == passenger:
                    num_waiting += 1
                    is_waiting = True
                    required_floors.add(parts[2]) # Add origin floor to required stops
                    break # Found origin, move to next passenger check

            # Add destination floor to required stops for all unserved passengers
            dest_floor = self.passenger_to_dest_floor.get(passenger)
            if dest_floor: # Should always have a destination
                 required_floors.add(dest_floor)
            else:
                 # Invalid task definition? Passenger without destination.
                 return float('inf')


        # h_passengers = (actions to board waiting) + (actions to depart unserved)
        h_passengers = num_waiting + len(unserved_passengers)

        # 3. Estimate movement cost.
        #    The lift must visit all floors in required_floors.
        #    A lower bound on movement is the vertical span covered,
        #    including the current lift floor and all required floors.

        h_movement = 0
        if required_floors:
            required_levels = [self.floor_to_level[f] for f in required_floors if f in self.floor_to_level]

            # If any required floor level is unknown, something is wrong.
            if len(required_levels) != len(required_floors):
                 return float('inf')

            min_req_level = min(required_levels)
            max_req_level = max(required_levels)

            # The lift must travel from its current level to cover the range
            # from the minimum required level to the maximum required level.
            # The total vertical distance covered is at least the span
            # from the lowest level visited (min(current, min_req))
            # to the highest level visited (max(current, max_req)).
            h_movement = max(current_lift_level, max_req_level) - min(current_lift_level, min_req_level)

        # Total heuristic is the sum of passenger actions and estimated movement.
        total_heuristic = h_passengers + h_movement

        return total_heuristic

