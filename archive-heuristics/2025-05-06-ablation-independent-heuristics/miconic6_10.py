import heapq
import logging

from heuristics.heuristic_base import Heuristic
from task import Operator, Task


class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing
    the number of necessary board and depart actions for unserved passengers
    and the estimated minimum number of lift movement actions required to
    visit all floors where actions (pickup or dropoff) are needed.

    Assumptions:
    1. Floor names are strings starting with 'f' followed by a number (e.g., 'f1', 'f10').
       The numerical part is used to determine the floor order and calculate distances.
    2. The PDDL state representation is complete for dynamic facts (`origin`, `boarded`, `lift-at`, `served`).
       Any unserved passenger not listed with an `origin` or `boarded` fact in the state
       is considered unreachable from this state, leading to an infinite heuristic value.
    3. The goal is to serve all passengers listed in the task's goals.

    Heuristic Initialization:
    The constructor processes the static facts from the task:
    - It builds a dictionary mapping each passenger to their destination floor using `(destin ?p ?f)` facts.
    - It identifies all floor names mentioned in `(above ?f1 ?f2)` facts.
    - It sorts the identified floor names numerically based on the number following 'f'.
    - It creates a mapping from sorted floor names to their numerical index (0-based).

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state. If yes, the heuristic is 0.
    2. Identify the current floor of the lift from the state fact `(lift-at ?f)`. If not found, the state is invalid or a dead end, return infinity.
    3. Identify all passengers who are not yet served, based on the task's goals.
    4. Build maps of current passenger locations: `origin_map` (passenger -> floor) and `boarded_passengers` (set of boarded passengers) by iterating through the state facts.
    5. Initialize `required_floors` set, `n_waiting` count, and `n_boarded` count.
    6. Iterate through all passengers whose destinations are known (from static facts).
    7. For each passenger, check if they are served in the current state.
    8. If a passenger is not served:
        - Check if they are waiting at an origin floor according to `origin_map`. If yes, add their origin floor to `required_floors` and increment `n_waiting`.
        - Else, check if they are boarded according to `boarded_passengers`. If yes, add their destination floor (from `self.destinations`) to `required_floors` and increment `n_boarded`.
        - If a passenger is unserved but neither waiting nor boarded according to the state, this heuristic assumes they are unreachable, but we don't explicitly handle this here; the check for `required_floors` being empty later covers cases where no progress can be estimated.
    9. Calculate `total_actions = n_waiting + n_boarded`. This is a lower bound on the number of board and depart actions needed.
    10. If `required_floors` is empty:
        - This implies there are no waiting passengers, and any boarded passengers are already at their destination (which must be the current floor, otherwise the destination would be in `required_floors`).
        - If `n_boarded > 0`, the remaining actions are just the `n_boarded` depart actions. Return `n_boarded`.
        - If `n_boarded == 0`, it means no waiting and no boarded unserved passengers were found in the state. Since the goal is not met (checked in step 1), some goal passengers are not accounted for in the state, implying an unsolvable state for this heuristic. Return infinity.
    11. If `required_floors` is not empty:
        - Convert the current lift floor and all required floors to their numerical indices using `self.floor_map`.
        - Find the minimum (`min_req_idx`) and maximum (`max_req_idx`) indices among the required floors.
        - Calculate the estimated move actions: `min(abs(current_idx - min_req_idx), abs(current_idx - max_req_idx)) + (max_req_idx - min_req_idx)`. This estimates the cost to reach the nearest extreme required floor and then traverse the range of required floors.
        - The total heuristic value is `total_actions + move_estimate`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goals = task.goals
        self.destinations = {}
        all_floor_names = set()

        # Parse static facts to get destinations and floor names
        for fact_str in task.static:
            predicate, args = self._parse_fact(fact_str)
            if predicate == 'destin':
                p, d = args
                self.destinations[p] = d
            elif predicate == 'above':
                f1, f2 = args
                all_floor_names.add(f1)
                all_floor_names.add(f2)

        # Sort floors numerically and create floor map
        # Assumes floor names are like 'f1', 'f10', 'f20'
        try:
            self.sorted_floors = sorted(list(all_floor_names), key=lambda f: int(f[1:]))
            self.floor_map = {f: i for i, f in enumerate(self.sorted_floors)}
        except ValueError:
            # Handle cases where floor names might not follow the fX pattern
            # Fallback to alphabetical sort or raise error
            logging.error("Miconic heuristic assumes floor names are fX, failed to parse.")
            # Fallback: simple alphabetical sort (might not match 'above' facts)
            self.sorted_floors = sorted(list(all_floor_names))
            self.floor_map = {f: i for i, f in enumerate(self.sorted_floors)}
            # Or raise: raise ValueError("Floor names must be in 'fX' format")


    def _parse_fact(self, fact_str):
        """Parses a PDDL fact string into predicate and arguments."""
        # Remove surrounding brackets
        content = fact_str[1:-1]
        # Split by space
        parts = content.split()
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def __call__(self, node):
        state = node.state

        # 1. Check if goal is reached
        if self.goals <= state:
            return 0

        # 2. Find current lift floor
        current_floor = None
        for fact_str in state:
            predicate, args = self._parse_fact(fact_str)
            if predicate == 'lift-at':
                current_floor = args[0]
                break

        if current_floor is None:
            # Lift location not found in state - invalid state?
            return float('inf')

        # 3. Identify unserved passengers and required floors/counts
        required_floors = set()
        n_waiting = 0
        n_boarded = 0

        # Build maps of current passenger status from state
        origin_map = {}
        boarded_passengers = set()
        served_passengers = set()

        for fact_str in state:
            predicate, args = self._parse_fact(fact_str)
            if predicate == 'origin':
                p, o = args
                origin_map[p] = o
            elif predicate == 'boarded':
                p = args[0]
                boarded_passengers.add(p)
            elif predicate == 'served':
                p = args[0]
                served_passengers.add(p)

        # Iterate through all passengers whose destinations are known
        for p in self.destinations.keys():
            if p not in served_passengers:
                # Passenger is unserved
                if p in origin_map:
                    # Passenger is waiting at origin
                    required_floors.add(origin_map[p])
                    n_waiting += 1
                elif p in boarded_passengers:
                    # Passenger is boarded, needs dropoff at destination
                    if p in self.destinations:
                        required_floors.add(self.destinations[p])
                        n_boarded += 1
                    else:
                        # Should not happen in a valid problem definition
                        logging.error(f"Passenger {p} is boarded but destination is unknown.")
                        return float('inf')
                # else: Passenger is unserved but not waiting and not boarded
                # (according to the state). This heuristic cannot estimate cost
                # for them. They are implicitly handled by the goal check;
                # if goal is not met and no relevant passengers are found, return inf.


        # 9. Calculate total board/depart actions
        total_actions = n_waiting + n_boarded

        # 10. Handle case where no floors need explicit visiting for pickup/dropoff
        if not required_floors:
            if total_actions > 0:
                 # This happens if all unserved passengers found in the state
                 # are boarded and their destination is the current floor.
                 # The only remaining actions are departs.
                 return total_actions # Which is n_boarded in this case
            else:
                 # No waiting, no boarded unserved passengers in state.
                 # Since goal is not met (checked in step 1), some goal passengers
                 # are not accounted for in the state. Assume unsolvable.
                 return float('inf')

        # 11. Calculate move estimate if required_floors is not empty
        try:
            required_indices = {self.floor_map[f] for f in required_floors}
            min_req_idx = min(required_indices)
            max_req_idx = max(required_indices)
            current_idx = self.floor_map[current_floor]

            # Move estimate: distance to nearest required floor + span of required floors
            move_estimate = min(abs(current_idx - min_req_idx), abs(current_idx - max_req_idx)) + (max_req_idx - min_req_idx)

            # Total heuristic value
            heuristic_value = total_actions + move_estimate
            return heuristic_value

        except KeyError as e:
            # This might happen if a floor mentioned in state/static is not in floor_map
            # (e.g., floor not mentioned in any 'above' fact).
            logging.error(f"Floor mapping error: Could not find index for floor {e}. State might be invalid.")
            return float('inf')

