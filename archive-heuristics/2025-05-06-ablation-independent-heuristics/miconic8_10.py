from collections import defaultdict

from heuristics.heuristic_base import Heuristic
from task import Operator, Task


class miconicHeuristic(Heuristic):
    """
    Summary:
        Domain-dependent heuristic for the Miconic domain.
        Estimates the total number of actions required to reach the goal state
        by summing the estimated costs for each unserved passenger independently.
        The cost for a passenger waiting at their origin includes travel to the origin floor,
        the board action, travel to the destination floor, and the depart action.
        The cost for a passenger already boarded includes travel to the destination floor
        and the depart action. Lift travel cost between floors is estimated by the
        absolute difference in their floor indices.

    Assumptions:
        - The PDDL domain follows the standard Miconic structure.
        - The `(above f1 f2)` predicate implies that `f2` is immediately above `f1`,
          defining a linear order of floors.
        - Floors form a single linear sequence.
        - All passengers have a defined initial origin (in the initial state)
          and a defined destination (in the static facts).
        - The state representation is consistent, meaning any unserved passenger
          is either waiting at an origin floor (`(origin p f)`) or is boarded (`(boarded p)`).
        - The state always contains exactly one `(lift-at ?f)` fact.

    Heuristic Initialization:
        - Parses static facts (`task.static`) and initial state (`task.initial_state`)
          to identify all floor names and passenger names present in the problem.
        - Builds a map `floor_to_index` by first constructing an adjacency list
          from `(above f1 f2)` facts (f1 -> f2), finding the lowest floor (in-degree 0),
          and then traversing the `above` chain to assign increasing integer indices
          starting from 0.
        - Stores passenger destinations (`passenger_destin`) parsed from static facts.
        - Stores the set of all passenger names (`all_passengers`).

    Step-By-Step Thinking for Computing Heuristic:
        1. Identify the current floor of the lift from the input state (`node.state`)
           by finding the fact `(lift-at ?f)`. Get its corresponding integer index
           using the precomputed `floor_to_index` map.
        2. Identify the set of all passengers who are not yet served (`P_unserved`)
           by checking which passengers do not have the fact `(served p)` in the state.
           If `P_unserved` is empty, the current state is a goal state, and the
           heuristic value is 0.
        3. Identify the set of passengers currently waiting at their origin (`P_waiting`)
           by finding all facts `(origin p f)` in the state.
        4. Identify the set of passengers currently boarded (`P_boarded`) by finding
           all facts `(boarded p)` in the state.
        5. Initialize the total heuristic value `h` to 0.
        6. Iterate through each passenger `p` in the set `P_unserved`:
           a. Retrieve the destination floor `f_d` for passenger `p` from the
              precomputed `passenger_destin` map and get its index `f_d_index`.
              Handle cases where destination is not found (should not happen in valid problems) by returning infinity.
           b. Check if passenger `p` is in the `P_waiting` set:
              i. If yes, find their current origin floor `f_o` from the state's
                 `(origin p f)` facts and get its index `f_o_index`.
                 Handle cases where origin is not found (should not happen if in P_waiting) by returning infinity.
              ii. Add the estimated cost for this waiting passenger to `h`. This cost
                  is calculated as: `abs(current_lift_index - f_o_index)` (travel to origin)
                  `+ 1` (board action)
                  `+ abs(f_o_index - f_d_index)` (travel from origin to destination)
                  `+ 1` (depart action).
           c. Check if passenger `p` is in the `P_boarded` set (if not waiting, they must be boarded based on assumptions):
              i. If yes, add the estimated cost for this boarded passenger to `h`.
                 This cost is calculated as: `abs(current_lift_index - f_d_index)` (travel to destination)
                 `+ 1` (depart action).
        7. Return the final calculated heuristic value `h`.
    """
    def __init__(self, task: Task):
        super().__init__()
        self.task = task

        self.floor_to_index = {}
        self.passenger_destin = {}
        self.all_passengers = set()

        floor_names = set()
        above_map = {} # f_lower -> f_higher
        in_degree = defaultdict(int) # f -> count

        # Parse initial state and static facts to find all floors and passengers
        # Note: initial_state contains dynamic facts at the start, static contains static facts
        # We need initial_state to find initial origins, static for destinations and above relations
        all_relevant_facts = set(task.initial_state) | set(task.static)

        for fact_str in all_relevant_facts:
            parts = fact_str.strip('()').split()
            if not parts: continue # Skip empty strings or malformed facts

            predicate = parts[0]
            if predicate == 'above' and len(parts) == 3:
                f1, f2 = parts[1], parts[2]
                floor_names.add(f1)
                floor_names.add(f2)
                # (above f1 f2) means f2 is immediately above f1
                above_map[f1] = f2
                in_degree[f2] += 1
                in_degree[f1] += 0 # Ensure f1 is in map even if in_degree is 0
            elif predicate == 'destin' and len(parts) == 3:
                p, f = parts[1], parts[2]
                self.passenger_destin[p] = f
                self.all_passengers.add(p)
                floor_names.add(f)
            elif predicate == 'origin' and len(parts) == 3:
                 # Origin facts are dynamic, but we collect floor names from initial ones
                 p, f = parts[1], parts[2]
                 self.all_passengers.add(p)
                 floor_names.add(f)
            elif predicate == 'lift-at' and len(parts) == 2:
                 f = parts[1]
                 floor_names.add(f)
            # We don't need to parse 'boarded' or 'served' in __init__

        # Ensure all found floors are in in_degree map
        for f in floor_names:
             in_degree[f] += 0 # Add floor with 0 if not present

        # Find the lowest floor (in-degree 0)
        lowest_floor = None
        for f, degree in in_degree.items():
            if degree == 0:
                lowest_floor = f
                break

        # Build floor_to_index map by traversing upwards
        if lowest_floor is not None:
            current_floor = lowest_floor
            index = 0
            while current_floor is not None:
                self.floor_to_index[current_floor] = index
                current_floor = above_map.get(current_floor)
                index += 1
        # If lowest_floor is None or traversal doesn't cover all floor_names,
        # it suggests an issue with the problem definition (e.g., disconnected floors, cycles).
        # In a valid miconic problem, this traversal should build the full map.
        # If floor_to_index is empty or incomplete, subsequent lookups will return None,
        # which is handled in __call__ by returning infinity.


    def __call__(self, node):
        state = node.state

        # 1. Find current lift floor and index
        current_lift_floor = None
        for fact_str in state:
            if fact_str.startswith('(lift-at '):
                parts = fact_str.strip('()').split()
                if len(parts) == 2:
                    current_lift_floor = parts[1]
                    break

        # Should always find lift-at in a valid state
        if current_lift_floor is None:
             # This indicates an invalid state representation or domain issue
             # Returning infinity makes this state undesirable for the search
             return float('inf')

        current_lift_index = self.floor_to_index.get(current_lift_floor)
        # Should always find the lift floor in the precomputed map
        if current_lift_index is None:
             return float('inf')


        # 2. Identify P_unserved, P_waiting, P_boarded
        served_passengers = set()
        origin_facts = {} # p -> f
        boarded_facts = set()

        for fact_str in state:
            parts = fact_str.strip('()').split()
            if not parts: continue # Skip empty strings or malformed facts

            predicate = parts[0]
            if predicate == 'served' and len(parts) == 2:
                served_passengers.add(parts[1])
            elif predicate == 'origin' and len(parts) == 3:
                p, f = parts[1], parts[2]
                origin_facts[p] = f
            elif predicate == 'boarded' and len(parts) == 2:
                boarded_facts.add(parts[1])

        P_unserved = self.all_passengers - served_passengers
        P_waiting = set(origin_facts.keys())
        P_boarded = boarded_facts

        # 3. Calculate the sum of costs
        h = 0

        if not P_unserved:
            # Goal state: all passengers are served
            return 0

        for p in P_unserved:
            f_d = self.passenger_destin.get(p)
            # Assuming all passengers have a destination defined in static facts
            if f_d is None:
                 # Invalid problem definition? Passenger exists but no destination.
                 return float('inf')

            f_d_index = self.floor_to_index.get(f_d)
            # Assuming all destination floors are in the precomputed map
            if f_d_index is None:
                 return float('inf')


            if p in P_waiting:
                # Passenger is waiting at origin
                f_o = origin_facts.get(p)
                # Assuming if a passenger is in P_waiting, their origin fact is in the state
                if f_o is None:
                     # Inconsistent state? Passenger in P_waiting set but no origin fact found.
                     # This shouldn't happen in a valid state reachable from initial state.
                     return float('inf')

                f_o_index = self.floor_to_index.get(f_o)
                # Assuming all origin floors are in the precomputed map
                if f_o_index is None:
                     return float('inf')

                # Cost: travel to origin + board + travel to destin + depart
                h += abs(current_lift_index - f_o_index) + 1 + abs(f_o_index - f_d_index) + 1

            elif p in P_boarded:
                # Passenger is boarded
                # Cost: travel to destin + depart
                h += abs(current_lift_index - f_d_index) + 1
            # Else: passenger is unserved but neither waiting nor boarded? (inconsistent state?)
            # Based on domain, unserved passengers are either waiting or boarded.
            # If state is inconsistent, this heuristic might be inaccurate, but we assume consistency.

        return h
