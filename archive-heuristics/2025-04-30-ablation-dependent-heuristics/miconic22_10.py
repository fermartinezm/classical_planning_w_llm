# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Helper function to parse a PDDL fact string into a tuple."""
    # Remove leading/trailing brackets and split by space
    # Example: '(predicate arg1 arg2)' -> ('predicate', 'arg1', 'arg2')
    parts = fact_string[1:-1].split()
    return tuple(parts)

class miconicHeuristic:
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    This heuristic estimates the cost to reach the goal state (all passengers served)
    by summing up the estimated minimum actions required for each unserved passenger
    independently. It does not account for synergies like picking up or dropping off
    multiple passengers at the same floor or transporting multiple passengers
    simultaneously. This makes it non-admissible but potentially effective for
    greedy best-first search by prioritizing states where passengers are closer
    to being served.

    Assumptions:
    - The floor structure is linear and ordered according to the 'above' predicates
      provided in the static facts. The heuristic relies on mapping floor names
      to numerical indices based on this ordering.
    - PDDL facts are represented as strings in the format '(predicate arg1 arg2 ...)'.
    - Object names within facts are simple strings (e.g., 'p1', 'f2').
    - The initial state and static facts contain enough information to identify all
      relevant floors and passenger destinations.
    - In any reachable state, an unserved passenger is either waiting at their origin
      or is boarded in the lift.

    Heuristic Initialization:
    Upon initialization, the heuristic processes the static facts and initial state
    from the task definition.
    1. It identifies all floor objects present in the problem instance by examining
       the arguments of relevant predicates ('above', 'destin', 'lift-at', 'origin')
       in the static and initial state facts.
    2. It builds a mapping from floor names (strings) to numerical indices (integers).
       This mapping is derived from the 'above' predicates. A directed graph is
       constructed where an edge exists from floor f_j to f_i if '(above f_i f_j)'
       is a static fact (meaning f_i is above f_j, so f_j is below f_i). The in-degree
       of a floor f_i in this graph represents the number of floors directly below f_i.
       Floors with an in-degree of 0 are the lowest floors. Kahn's algorithm
       (topological sort) is used to process floors from lowest to highest, assigning
       increasing indices starting from 0.
    3. It stores the destination floor for each passenger by parsing the '(destin ?p ?f)'
       static facts into a dictionary mapping passenger names to their destination
       floor names.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state, the heuristic value is computed as follows:
    1. Check if the state is a goal state (all passengers served). This is done by
       calling `task.goal_reached(state)`. If true, the heuristic value is 0.
    2. Identify the current floor of the lift by finding the fact '(lift-at ?f)' in the state.
       Convert this floor name to its numerical index using the pre-computed map `self.floor_to_index`.
    3. Identify the status of all passengers: waiting at origin, boarded, or served.
       This is done by iterating through the facts in the current state.
    4. Initialize the total heuristic value to 0.
    5. Iterate through all passengers whose destinations are known (from `self.destin_map`).
       For each passenger `p`:
       a. If the passenger `p` is already served (fact '(served p)' is in the state),
          they contribute 0 to the heuristic. Continue to the next passenger.
       b. If the passenger `p` is waiting at origin floor `origin_f` (fact '(origin p origin_f)'
          is in the state):
          i. Get the numerical index of the origin floor (`origin_idx`) and the destination
             floor (`destin_idx`) using `self.floor_to_index`.
          ii. Calculate the estimated minimum actions for this passenger:
              - Travel from the current lift floor to the origin floor: `abs(current_lift_idx - origin_idx)` moves.
              - Board the passenger: 1 action.
              - Travel from the origin floor to the destination floor: `abs(origin_idx - destin_idx)` moves.
              - Depart the passenger: 1 action.
              Individual cost = `abs(current_lift_idx - origin_idx) + 1 + abs(origin_idx - destin_idx) + 1`.
          iii. Add this individual cost to the `total_heuristic`.
       c. If the passenger `p` is boarded (fact '(boarded p)' is in the state):
          i. Get the numerical index of the destination floor (`destin_idx`) using
             `self.floor_to_index`.
          ii. Calculate the estimated minimum actions for this passenger:
              - Travel from the current lift floor to the destination floor: `abs(current_lift_idx - destin_idx)` moves.
              - Depart the passenger: 1 action.
              Individual cost = `abs(current_lift_idx - destin_idx) + 1`.
          iii. Add this individual cost to the `total_heuristic`.
       d. (Implicit) If a passenger is not served, not waiting, and not boarded, they
          do not contribute to the heuristic. This case should not occur in valid states
          given the domain's state transitions.
    6. Return the final `total_heuristic` value.
    """
    def __init__(self, task):
        self.floor_to_index = {}
        self.destin_map = {}
        self._process_static_and_initial(task.static, task.initial_state)

    def _process_static_and_initial(self, static_facts, initial_state_facts):
        """Processes static and initial state facts to build floor map and destin map."""
        floor_names = set()
        # above_graph: f_j -> list of f_i where (above f_i f_j) (f_j is below f_i)
        above_graph = {}
        # in_degree: f -> count of floors directly below it
        in_degree = {}

        # Extract floor names and destin map from static facts
        for fact_str in static_facts:
            parsed = parse_fact(fact_str)
            pred = parsed[0]
            if pred == 'above':
                f_i, f_j = parsed[1], parsed[2]
                floor_names.add(f_i)
                floor_names.add(f_j)
            elif pred == 'destin':
                 # parsed[1] is passenger, parsed[2] is floor
                 floor_names.add(parsed[2])
                 self.destin_map[parsed[1]] = parsed[2] # Store destin map here

        # Extract floor names from initial state facts
        for fact_str in initial_state_facts:
             parsed = parse_fact(fact_str)
             pred = parsed[0]
             if pred == 'lift-at':
                 floor_names.add(parsed[1])
             elif pred == 'origin':
                 # parsed[1] is passenger, parsed[2] is floor
                 floor_names.add(parsed[2])
             # boarded, served don't have floor args

        # Initialize graph and in-degrees for all found floors
        for f in floor_names:
            above_graph[f] = []
            in_degree[f] = 0

        # Build graph (f_j -> f_i if f_i is above f_j) and calculate in-degrees
        # In-degree of f_i counts how many floors are directly below it.
        for fact_str in static_facts:
            parsed = parse_fact(fact_str)
            if parsed[0] == 'above':
                f_i, f_j = parsed[1], parsed[2]
                # Edge f_j -> f_i (f_j is below f_i)
                above_graph[f_j].append(f_i)
                in_degree[f_i] += 1 # f_i has one more floor directly below it (f_j)

        # Kahn's algorithm for topological sort (assigning indices 0, 1, 2... from lowest floor up)
        # Floors with in-degree 0 in this graph are the lowest floors.
        queue = [f for f in floor_names if in_degree[f] == 0]
        index = 0
        assigned_count = 0

        while queue:
            # Dequeue a floor with in-degree 0 (a lowest remaining floor)
            u = queue.pop(0)
            self.floor_to_index[u] = index
            index += 1
            assigned_count += 1

            # For each floor 'v_above_u' that is directly above 'u'
            # (i.e., there is an edge u -> v_above_u in the above_graph)
            for v_above_u in above_graph.get(u, []):
                 in_degree[v_above_u] -= 1 # v_above_u has one less floor below it that hasn't been assigned an index
                 if in_degree[v_above_u] == 0:
                     queue.append(v_above_u)

        # Basic consistency check
        if assigned_count != len(floor_names):
             # This indicates the 'above' predicates do not form a single connected component
             # covering all identified floors, or contain cycles (invalid PDDL).
             # For standard miconic benchmarks, this should not happen.
             # In a real planner, robust error handling or fallback might be needed.
             pass # print(f"Warning: Could not assign index to all floors. Assigned {assigned_count}/{len(floor_names)}")


    def __call__(self, state, task):
        """Computes the heuristic value for the given state."""
        # Check if goal is reached
        if task.goal_reached(state):
            return 0

        current_lift_floor = None
        waiting_passengers = {} # p -> origin_floor
        boarded_passengers = set()
        served_passengers = set()

        # Parse state to find lift position and passenger status
        for fact_str in state:
            parsed = parse_fact(fact_str)
            pred = parsed[0]
            if pred == 'lift-at':
                current_lift_floor = parsed[1]
            elif pred == 'origin':
                p, f = parsed[1], parsed[2]
                waiting_passengers[p] = f
            elif pred == 'boarded':
                p = parsed[1]
                boarded_passengers.add(p)
            elif pred == 'served':
                p = parsed[1]
                served_passengers.add(p)

        # current_lift_floor must be found if the state is valid and not goal
        # (unless goal is reached from a state without lift-at, which is impossible in miconic)
        # Add a safeguard just in case, though it indicates an issue with state representation or domain.
        if current_lift_floor is None:
             # This state is likely invalid or represents an unresolvable situation
             # Return infinity or a very large number.
             # In Python, float('inf') represents infinity.
             return float('inf')

        # Get index of current lift floor
        current_idx = self.floor_to_index.get(current_lift_floor)
        # Safeguard if lift floor is not in our map (shouldn't happen if floor extraction is correct)
        if current_idx is None:
             return float('inf')


        total_heuristic = 0

        # Iterate over all passengers whose destinations are known
        for p in self.destin_map.keys():
            # Check if passenger is already served
            if p in served_passengers:
                continue

            # Get destination floor index
            destin_floor = self.destin_map[p]
            destin_idx = self.floor_to_index.get(destin_floor)
            # Safeguard if destination floor is not in our map
            if destin_idx is None:
                 return float('inf')


            # Check if passenger is waiting at origin
            if p in waiting_passengers:
                origin_floor = waiting_passengers[p]
                origin_idx = self.floor_to_index.get(origin_floor)
                 # Safeguard if origin floor is not in our map
                if origin_idx is None:
                     return float('inf')

                # Cost: travel to origin + board + travel to destin + depart
                individual_cost = abs(current_idx - origin_idx) + 1 + abs(origin_idx - destin_idx) + 1
                total_heuristic += individual_cost

            # Check if passenger is boarded
            elif p in boarded_passengers:
                # Cost: travel to destin + depart
                individual_cost = abs(current_idx - destin_idx) + 1
                total_heuristic += individual_cost

            # If passenger is not served, not waiting, and not boarded, they are ignored.
            # This implies they don't need actions from this state to be served.
            # This aligns with the goal definition (served passengers).

        return total_heuristic
