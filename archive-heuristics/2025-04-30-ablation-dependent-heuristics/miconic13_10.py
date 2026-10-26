import math
from collections import deque

class miconicHeuristic:
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    This heuristic estimates the remaining cost to reach the goal state
    (all passengers served) by summing the estimated costs for boarding
    passengers, departing passengers, and the minimum required lift travel.
    It is designed for greedy best-first search and is not admissible,
    but aims to provide a good estimate to minimize node expansions.

    Assumptions:
    - The PDDL domain follows the structure of the miconic domain provided.
    - Floor ordering is defined by transitive 'above' predicates in static facts.
    - Passenger destinations are defined by 'destin' predicates in static facts.
    - Passenger initial origins are defined by 'origin' predicates in the initial state.
    - A state is represented as a frozenset of fact strings.
    - The Task object provides static facts, initial state, and goals.

    Heuristic Initialization:
    The heuristic is initialized with the planning Task object. It precomputes
    and stores the following information from the static facts and initial state:
    1. A mapping from floor names (e.g., 'f1') to numerical indices based on
       the 'above' relations, representing their order from lowest (index 0)
       to highest. This is done by building a graph of 'above' relations and
       counting how many floors are transitively below each floor.
    2. A dictionary mapping passenger names to their destination floor names.
    3. A set of all passenger names in the problem instance.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Identify the current floor of the lift by finding the fact '(lift-at ?f)'
       in the state. Convert the floor name to its numerical index using the
       precomputed floor map.
    2. Identify all passengers who have not yet been served by checking for
       '(served ?p)' facts in the state.
    3. If there are no unserved passengers, the state is a goal state, and the
       heuristic value is 0.
    4. Identify unserved passengers who are currently boarded and unserved passengers
       who are currently at their origin floors.
    5. Initialize the heuristic value `h`. Add 1 for each unserved boarded passenger
       (estimating the 'depart' action). Add 2 for each unserved passenger at their
       origin (estimating 'board' and 'depart' actions).
    6. Determine the set of 'required' floors the lift must visit: This set
       includes the origin floor `o` for every unserved passenger at their origin,
       and the destination floor `d` for every unserved boarded passenger.
    7. If the set of required floors is empty (should only happen if all passengers
       are served), the travel cost is 0.
    8. If the set of required floors is not empty, find the minimum and maximum
       floor indices among the required floors using the precomputed floor map.
    9. Estimate the minimum travel actions needed: This is the number of floor
       movements required to go from the current lift floor to cover the range
       of required floors. Calculate this as
       `(max_required_idx - min_required_idx) + min(abs(current_floor_idx - min_required_idx), abs(current_floor_idx - max_required_idx))`.
       Add this travel cost to `h`.
    10. Return the final value of `h`.
    """
    def __init__(self, task):
        self.task = task
        self.floor_map = {}
        self.destinations = {}
        self.all_passengers = set()

        floors = set()
        above_relations = [] # Store (f1, f2) where f1 is above f2

        # Collect floors and above relations from static facts
        for fact_str in task.static:
            pred, args = self._parse_fact(fact_str)
            if pred == 'above':
                f1, f2 = args
                floors.add(f1)
                floors.add(f2)
                above_relations.append((f1, f2))
            elif pred == 'destin':
                p, d = args
                self.destinations[p] = d
                self.all_passengers.add(p)

        # Collect floors and passengers from initial state (may include floors/passengers not in static)
        for fact_str in task.initial_state:
             pred, args = self._parse_fact(fact_str)
             if pred == 'lift-at':
                 floors.add(args[0])
             elif pred == 'origin':
                 p, o = args
                 floors.add(o)
                 self.all_passengers.add(p)
             elif pred == 'boarded':
                 self.all_passengers.add(args[0])
             elif pred == 'served':
                 self.all_passengers.add(args[0])

        # Ensure all floors mentioned in destinations are included
        for d in self.destinations.values():
             floors.add(d)

        # Determine floor order and create floor_map
        # Build graph where edge f_i -> f_j means f_i is above f_j
        above_graph = {f: set() for f in floors}
        for f1, f2 in above_relations:
            above_graph[f1].add(f2)

        # Count how many floors are transitively below each floor
        num_below = {f: 0 for f in floors}
        # Use BFS/DFS from each floor to find reachable nodes (floors below)
        for start_f in floors:
            count = 0
            visited = {start_f}
            q = deque([start_f])
            while q:
                curr_f = q.popleft()
                for next_f in above_graph.get(curr_f, set()):
                    if next_f not in visited:
                        visited.add(next_f)
                        q.append(next_f)
                        count += 1
            num_below[start_f] = count

        # Sort floors by the number of floors below them (ascending)
        sorted_floors = sorted(floors, key=lambda f: num_below[f])

        # Create floor_map
        self.floor_map = {f: i for i, f in enumerate(sorted_floors)}

    def _parse_fact(self, fact_str):
        """Helper to parse PDDL fact string."""
        # Removes outer parentheses and splits by space
        parts = fact_str.strip('()').split()
        return parts[0], parts[1:] # predicate, arguments


    def __call__(self, state):
        """
        Computes the heuristic value for a given state.

        @param state: The current state (frozenset of fact strings).
        @return: The estimated number of actions to reach the goal.
        """
        # 1. Find current lift floor
        current_floor = None
        # Identify unserved passengers, boarded passengers, and origin passengers in one pass
        served_passengers = set()
        boarded_unserved = set()
        origin_unserved = {} # {passenger: origin_floor}
        current_floor = None # Find lift location

        for fact_str in state:
            pred, args = self._parse_fact(fact_str)
            if pred == 'served':
                served_passengers.add(args[0])
            elif pred == 'boarded':
                # Tentatively add to boarded_unserved, will filter later
                boarded_unserved.add(args[0])
            elif pred == 'origin':
                 # Tentatively add to origin_unserved, will filter later
                 origin_unserved[args[0]] = args[1]
            elif pred == 'lift-at':
                current_floor = args[0]


        # Should always find lift-at in a valid state
        if current_floor is None:
             # If goal is reached, h=0. Otherwise, state is likely invalid.
             if self.task.goal_reached(state):
                 return 0
             else:
                 # Invalid state, return high cost
                 return math.inf

        current_floor_idx = self.floor_map[current_floor]

        # Filter boarded_unserved and origin_unserved based on served_passengers
        boarded_unserved = {p for p in boarded_unserved if p not in served_passengers}
        origin_unserved = {p: o for p, o in origin_unserved.items() if p not in served_passengers}

        # 2. Identify unserved passengers (union of those boarded and those at origin)
        # Note: This heuristic focuses only on unserved passengers who are
        # currently boarded or at their origin. Passengers who might be unserved
        # but not in these states (e.g., due to an invalid state representation)
        # are not explicitly counted in the board/depart actions, but their
        # destinations/origins might still be included in required_floors if
        # they were previously at origin or boarded.
        # The check for total unserved count ensures h=0 only at the goal.
        total_unserved_count = len(self.all_passengers) - len(served_passengers)


        # 3. If no unserved passengers, goal reached
        if total_unserved_count == 0:
            return 0

        # 5. Initialize heuristic with board/depart actions
        # Count only unserved passengers who are actually in the state as boarded or at origin
        h = len(boarded_unserved) # Depart actions needed for boarded passengers
        h += len(origin_unserved) * 2 # Board + Depart actions needed for passengers at origin

        # 6. Determine required floors
        required_floors = set()
        for p in boarded_unserved:
            # Ensure destination exists (should be in static)
            if p in self.destinations:
                required_floors.add(self.destinations[p])

        for p, o in origin_unserved.items():
            required_floors.add(o)
            # Ensure destination exists
            if p in self.destinations:
                 required_floors.add(self.destinations[p])

        # 7. Calculate travel cost
        if not required_floors:
             # This case implies there are unserved passengers (total_unserved_count > 0)
             # but none are currently at an origin or boarded. This should not happen
             # in valid states according to the domain definition.
             # If it happens, the board/depart count `h` is 0.
             # We still need to serve these passengers, but their location is unknown
             # or invalid for the heuristic calculation.
             # Returning a value > 0 is necessary if total_unserved_count > 0.
             # The simplest is to return the board/depart count (which is 0 here)
             # plus the number of unserved passengers whose location is unknown.
             # However, sticking to the defined components: board, depart, travel.
             # If required_floors is empty, travel is 0.
             # The heuristic becomes just the board/depart count (which is 0).
             # This is incorrect if total_unserved_count > 0.
             # Let's assume valid states where unserved passengers are always at origin or boarded.
             # In that case, if total_unserved_count > 0, required_floors must be non-empty.
             # So, if we reach here, required_floors must be non-empty.
             pass # h is already calculated, travel_cost will be added next

        else: # required_floors is not empty
            required_indices = {self.floor_map[f] for f in required_floors}
            min_req_idx = min(required_indices)
            max_req_idx = max(required_indices)

            # 9. Estimate minimum travel actions
            # Travel to the closer end of the required range, then sweep the range
            travel_cost = (max_req_idx - min_req_idx) + min(abs(current_floor_idx - min_req_idx), abs(current_floor_idx - max_req_idx))

            h += travel_cost

        return h
