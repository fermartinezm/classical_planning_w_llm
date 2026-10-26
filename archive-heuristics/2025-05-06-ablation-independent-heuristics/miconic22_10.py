from heuristics.heuristic_base import Heuristic
from task import Operator, Task
import re

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the miconic domain.

    Summary:
        Estimates the cost to reach the goal by summing the minimum actions
        required for boarding/departing unserved passengers and the minimum
        lift travel required to visit the necessary floors.

    Assumptions:
        - The 'above' predicates define a total order on floors.
        - Floors are named consistently (e.g., f1, f2, ...), allowing sorting
          based on the number suffix to determine the floor order. This heuristic
          assumes '(above f_i f_j)' for i < j implies f_i is higher than f_j,
          leading to f1 being the highest floor and fN the lowest.
        - The goal is always to serve all passengers.

    Heuristic Initialization:
        - Parses the static 'above' facts and object definitions to identify all
          floor names.
        - Creates a mapping from floor names to integer floor numbers. It assumes
          floor names are in the format 'f<number>' and sorts them numerically.
          Based on the example static facts `(above f_i f_j)` for `i < j`, which
          means `f_i` is higher than `f_j`, the floor 'f1' is mapped to the highest
          floor number and 'fN' to the lowest, where N is the total number of floors.
        - Stores the goal facts (which passengers need to be served).
        - Stores the static passenger destination facts.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is a goal state by verifying if all goal facts
           (served passengers) are present in the state. If yes, the heuristic is 0.
        2. Identify the current floor of the lift by finding the fact '(lift-at ?f)'
           in the current state. Convert the floor name '?f' to its integer number
           using the precomputed mapping. If the lift location is not found or the
           floor is unknown, return infinity.
        3. Initialize counts for required board actions (`n_board`), required depart
           actions (`n_depart`), and a set of required floor numbers (`required_floor_nums`)
           that the lift must visit.
        4. Identify all unserved passengers by checking which goal facts '(served p)'
           are not present in the current state.
        5. Iterate through the current state facts to determine the status of each
           unserved passenger (waiting at origin or boarded).
        6. For each unserved passenger 'p':
           - Increment `n_depart` (each unserved passenger eventually needs a depart action).
           - If '(origin p ?origin_f)' is true in the state:
             - Increment `n_board` (needs a board action).
             - Add the integer floor number of '?origin_f' to `required_floor_nums`.
           - If '(boarded p)' is true in the state:
             - Find the destination floor '?destin_f' for passenger 'p' from the
               precomputed static passenger destination mapping.
             - Add the integer floor number of '?destin_f' to `required_floor_nums`.
           - If an unserved passenger has neither '(origin ...)' nor '(boarded ...)'
             true, the state is inconsistent or unreachable; return infinity.
        7. Calculate the minimum lift travel cost (`min_travel`):
           - If `required_floor_nums` is empty, `min_travel` is 0.
           - If not empty, find the minimum (`min_req_floor`) and maximum (`max_req_floor`)
             floor numbers in the set.
           - The minimum travel required to visit all floors in the range
             [min_req_floor, max_req_floor] starting from `current_f_num` is the
             distance to reach one end of the range plus the distance to traverse
             the range. This is calculated as
             `(max_req_floor - min_req_floor) + min(abs(current_f_num - min_req_floor), abs(current_f_num - max_req_floor))`.
        8. The total heuristic value is the sum of `n_board`, `n_depart`, and `min_travel`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goals = task.goals
        self.static_facts = task.static

        # 1. Parse floor names and map them to integers
        self.floor_to_num = self._parse_floors(task.facts, self.static_facts)
        self.num_to_floor = {v: k for k, v in self.floor_to_num.items()}
        self.max_floor_num = max(self.floor_to_num.values()) if self.floor_to_num else 0

        # Store passenger destinations (static)
        self.passenger_destin = self._parse_destinations(self.static_facts)


    def _parse_floors(self, all_facts, static_facts):
        """Parses floor names and assigns integer numbers based on 'above' facts."""
        floor_names = set()

        # Extract all floor names from initial state, goals, and static facts
        # Facts are strings like '(predicate arg1 arg2 ...)'
        def extract_floor_from_fact(fact_str):
            parts = fact_str.strip("()").split()
            if not parts:
                return []
            predicate = parts[0]
            if predicate == 'lift-at' and len(parts) == 2:
                return [parts[1]]
            elif predicate in ['origin', 'destin', 'above'] and len(parts) == 3:
                 # Floors are typically the last argument in origin/destin, and both in above
                 floors = [parts[2]]
                 if predicate == 'above':
                     floors.append(parts[1])
                 return floors
            return []

        for fact in all_facts:
             floor_names.update(extract_floor_from_fact(fact))

        for goal in self.goals:
             floor_names.update(extract_floor_from_fact(goal))

        for fact in static_facts:
             floor_names.update(extract_floor_from_fact(fact))


        if not floor_names:
            return {} # No floors found

        # Assume floor names are f<number> and sort numerically to determine order.
        # Based on example (above fi fj) for i<j meaning fi is higher,
        # f1 is the highest, fN is the lowest.
        try:
            # Sort floors based on the integer part of their name
            sorted_floors = sorted(list(floor_names), key=lambda x: int(x[1:]))
            num_floors = len(sorted_floors)
            # Map f1 to num_floors, f2 to num_floors-1, ..., fN to 1
            floor_to_num = {f: num_floors - i for i, f in enumerate(sorted_floors)}
            return floor_to_num
        except (ValueError, IndexError):
            # Handle cases where floor names are not in f<number> format or empty
            # For a domain-dependent heuristic, assuming f<number> is reasonable.
            # If it fails, the heuristic cannot be computed meaningfully.
            # Returning empty means all floor lookups will fail, leading to infinity heuristic.
            print("Warning: Floor names not in expected 'f<number>' format. Cannot determine floor order reliably.")
            return {}


    def _parse_destinations(self, static_facts):
        """Parses static 'destin' facts."""
        passenger_destin = {}
        for fact in static_facts:
            if fact.startswith('(destin '):
                parts = fact.strip("()").split()
                if len(parts) == 3:
                    passenger = parts[1]
                    destin_floor = parts[2]
                    passenger_destin[passenger] = destin_floor
        return passenger_destin


    def __call__(self, node):
        """Computes the miconic heuristic for a given state."""
        state = node.state

        # 1. Check for goal state
        if self.goals <= state:
            return 0

        # 2. Identify current lift floor
        current_f = None
        for fact in state:
            if fact.startswith('(lift-at '):
                parts = fact.strip("()").split()
                if len(parts) == 2:
                    current_f = parts[1]
                    break

        if current_f is None:
             # Lift location unknown, state is likely invalid or unreachable
             return float('inf')

        current_f_num = self.floor_to_num.get(current_f)
        if current_f_num is None:
             # Unknown floor name for lift location (not in our mapping)
             return float('inf')


        # 3. Initialize counts and required floors
        n_board = 0
        n_depart = 0
        required_floor_nums = set()

        # 4. Identify unserved passengers
        unserved_passengers = set()
        for goal in self.goals:
             if goal.startswith('(served '):
                 parts = goal.strip("()").split()
                 if len(parts) == 2:
                     passenger = parts[1]
                     if goal not in state:
                         unserved_passengers.add(passenger)

        # 5. & 6. Iterate through state facts to find status of unserved passengers
        passenger_status = {} # {p: 'origin_f' or 'boarded'}

        for fact in state:
             if fact.startswith('(origin '):
                 parts = fact.strip("()").split()
                 if len(parts) == 3:
                     p, f = parts[1], parts[2]
                     if p in unserved_passengers:
                         passenger_status[p] = f # Store origin floor
             elif fact.startswith('(boarded '):
                 parts = fact.strip("()").split()
                 if len(parts) == 2:
                     p = parts[1]
                     if p in unserved_passengers:
                         passenger_status[p] = 'boarded' # Store status

        # Calculate counts and required floors based on unserved passengers
        for p in unserved_passengers:
             n_depart += 1 # Each unserved passenger needs a depart action

             status = passenger_status.get(p)

             if status and status != 'boarded': # Status is an origin floor name
                 origin_f = status
                 n_board += 1 # Needs a board action
                 origin_f_num = self.floor_to_num.get(origin_f)
                 if origin_f_num is not None:
                     required_floor_nums.add(origin_f_num) # Lift must visit origin floor
                 else:
                     # Unknown origin floor name
                     return float('inf')

             elif status == 'boarded':
                 # Passenger is boarded
                 destin_f = self.passenger_destin.get(p)
                 if destin_f:
                     destin_f_num = self.floor_to_num.get(destin_f)
                     if destin_f_num is not None:
                         required_floor_nums.add(destin_f_num) # Lift must visit destination floor
                     else:
                         # Unknown destination floor name
                         return float('inf')
                 else:
                     # Destination unknown for boarded passenger (should not happen in valid PDDL)
                     return float('inf')
             else:
                 # Unserved passenger with unknown status (not origin, not boarded)?
                 # This indicates an inconsistent state based on domain rules.
                 # print(f"Warning: Unserved passenger {p} with inconsistent status.")
                 return float('inf')


        # 7. Calculate minimum lift travel cost
        min_travel = 0
        if required_floor_nums:
            min_req_floor = min(required_floor_nums)
            max_req_floor = max(required_floor_nums)

            # Minimum travel is the distance to the closest end of the required range
            # plus the distance to traverse the range.
            dist_to_min_end = abs(current_f_num - min_req_floor)
            dist_to_max_end = abs(current_f_num - max_req_floor)
            dist_traverse_range = max_req_floor - min_req_floor

            min_travel = min(dist_to_min_end, dist_to_max_end) + dist_traverse_range


        # 8. Total heuristic
        h_value = n_board + n_depart + min_travel

        return h_value
