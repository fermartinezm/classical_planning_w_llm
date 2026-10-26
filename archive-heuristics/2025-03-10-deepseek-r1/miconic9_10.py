from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

class miconic9Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers by summing the individual costs for each passenger. The cost for each passenger is computed based on whether they are boarded or not, and the distance between the elevator's current position and their origin/destination floors.

    # Assumptions
    - Each passenger has a fixed origin and destination floor, known from the initial state.
    - The 'above' relations form a total order, allowing depth-based distance calculation.
    - The elevator can move directly to any lower floor in one step (up action) and requires one step per higher floor (down actions).

    # Heuristic Initialization
    - Extract 'origin' and 'destin' for each passenger from the initial state.
    - Build a depth map for each floor based on the 'above' static facts.
    - Store the 'above' relations to determine direct movement.

    # Step-By-Step Thinking for Computing Heuristic
    1. Determine the elevator's current floor from the state.
    2. For each passenger:
        a. If served, skip.
        b. If boarded, calculate steps from current floor to destination and add depart action.
        c. If not boarded, calculate steps from current floor to origin, board action, steps to destination, and depart action.
    3. Sum all individual costs.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static
        self.origin = {}
        self.destin = {}
        self.depth_map = defaultdict(int)
        self.above_set = set()

        # Extract origin and destin from initial state
        for fact in task.initial_state:
            parts = self._get_parts(fact)
            if parts[0] == 'origin':
                self.origin[parts[1]] = parts[2]
            elif parts[0] == 'destin':
                self.destin[parts[1]] = parts[2]

        # Process static 'above' facts
        for fact in static_facts:
            parts = self._get_parts(fact)
            if parts[0] == 'above':
                a, b = parts[1], parts[2]
                self.above_set.add((a, b))
                self.depth_map[b] += 1

    def _get_parts(self, fact):
        """Split a PDDL fact into its components."""
        return fact[1:-1].split()

    def _get_distance(self, from_floor, to_floor):
        """Calculate steps between two floors based on depth."""
        depth_from = self.depth_map[from_floor]
        depth_to = self.depth_map[to_floor]
        if depth_to > depth_from:
            return 1  # Down via up action
        else:
            return depth_from - depth_to  # Up via down actions

    def __call__(self, node):
        state = node.state
        current_elevator_floor = None
        for fact in state:
            parts = self._get_parts(fact)
            if parts[0] == 'lift-at':
                current_elevator_floor = parts[1]
                break
        if not current_elevator_floor:
            return 0  # Should not happen

        total_cost = 0
        for passenger in self.origin:
            # Check if served
            served = any(self._get_parts(fact) == ['served', passenger] for fact in state)
            if served:
                continue

            # Check if boarded
            boarded = any(self._get_parts(fact) == ['boarded', passenger] for fact in state)

            if boarded:
                destin_floor = self.destin[passenger]
                steps = self._get_distance(current_elevator_floor, destin_floor)
                total_cost += steps + 1  # depart action
            else:
                origin_floor = self.origin[passenger]
                destin_floor = self.destin[passenger]
                steps_to_origin = self._get_distance(current_elevator_floor, origin_floor)
                steps_to_destin = self._get_distance(origin_floor, destin_floor)
                total_cost += steps_to_origin + 1 + steps_to_destin + 1

        return total_cost
