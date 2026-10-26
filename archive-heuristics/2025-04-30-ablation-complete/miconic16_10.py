# Add the necessary import for the base class if it's not implicitly available
# from heuristics.heuristic_base import Heuristic
# Assuming Heuristic is available in the environment where this code runs.

from fnmatch import fnmatch
import math # Import math for infinity

# Helper functions from examples
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential errors or unexpected fact formats
        # print(f"Warning: get_parts received unexpected fact format: {fact}")
        return [] # Or raise an error
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    passengers. It considers the movement cost for the lift to visit all
    necessary floors (origins for waiting passengers, destinations for
    boarded passengers) and adds an estimate of the pending board/depart
    actions required for each unserved passenger.

    # Assumptions
    - Floors are linearly ordered, defined by `(above f_higher f_lower)` facts.
      Direct adjacency `(f_higher, f_lower)` is defined by `(above f_higher f_lower)`
      being true and there being no intermediate floor `f_intermediate` such that
      `(above f_higher f_intermediate)` and `(above f_intermediate f_lower)` are true.
    - Passenger destinations `(destin p f)` are static facts provided in `task.static`.
    - All relevant passengers have their destination defined in `task.static`.
    - The cost of move, board, and depart actions is 1.

    # Heuristic Initialization
    - Parses `(above f_higher f_lower)` facts to determine direct floor adjacencies
      and build a mapping from floor names to their index in the sorted floor
      order (lowest floor is index 0).
    - Parses `(destin p f)` facts from `task.static` to map each passenger
      to their destination floor.
    - Identifies all relevant passengers from the destinations found.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  Get the current floor of the lift by finding the fact `(lift-at ?f)` in the state. Find its corresponding index using the pre-calculated floor-to-index map.
    2.  Identify all passengers who are not yet served. A passenger `p` is unserved if `(served p)` is not present in the state. We consider only passengers whose destination is known from the static facts.
    3.  If there are no unserved passengers, the state is a goal state, and the heuristic value is 0.
    4.  Identify the set of "required" floors that the lift must visit:
        -   For each unserved passenger `p` who is currently waiting at an origin floor `f` (i.e., `(origin p f)` is true in the state), `f` is a required floor for pickup.
        -   For each unserved passenger `p` who is currently boarded (i.e., `(boarded p)` is true in the state), their destination floor `destin(p)` is a required floor for dropoff.
    5.  Map the required floors identified in step 4 to their corresponding indices using the floor-to-index map. Collect these into a set `required_indices`.
    6.  Calculate the estimated action cost: For each unserved passenger, estimate the number of board/depart actions needed. A passenger waiting at their origin needs 1 board and 1 depart (total 2). A passenger already boarded needs 1 depart (total 1). Sum these up for all unserved passengers.
    7.  If the set of `required_indices` is empty:
        -   This indicates that all unserved passengers are either waiting at the current lift floor or are boarded and need to depart at the current lift floor. No movement to a different floor is immediately required to reach a pickup/dropoff location.
        -   The heuristic value is the estimated action cost calculated in step 6.
    8.  If the set of `required_indices` is not empty:
        -   Find the minimum (`min_req_index`) and maximum (`max_req_index`) indices among the `required_indices`.
        -   Estimate the movement cost for the lift to visit all required floors. A reasonable non-admissible estimate is the distance needed to traverse the entire range of required floors (`max_req_index - min_req_index`) plus the minimum distance from the current lift floor (`current_index`) to either end of this range (`min(abs(current_index - min_req_index), abs(current_index - max_req_index))`).
        -   The total heuristic value is the sum of the estimated movement cost and the estimated action cost (from step 6).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        super().__init__(task)

        # 1. Parse (above f_higher f_lower) facts to build floor order
        above_pairs = set()
        all_floors = set()
        for fact in self.static:
            if match(fact, "above", "*", "*"):
                f_higher, f_lower = get_parts(fact)[1:]
                above_pairs.add((f_higher, f_lower))
                all_floors.add(f_higher)
                all_floors.add(f_lower)

        # Determine direct adjacencies
        directly_below_map = {}
        # Iterate through all pairs (f_higher, f_lower) where f_higher is above f_lower
        for f_higher, f_lower in above_pairs:
            is_direct = True
            # Check if there is any intermediate floor f_intermediate
            for f_intermediate in all_floors:
                if f_intermediate != f_lower and f_intermediate != f_higher:
                    # f_intermediate is between f_higher and f_lower if (above f_higher f_intermediate)
                    # and (above f_intermediate f_lower) are both true.
                    if (f_higher, f_intermediate) in above_pairs and (f_intermediate, f_lower) in above_pairs:
                        is_direct = False
                        break # Found an intermediate, so this is not a direct adjacency
            if is_direct:
                directly_below_map[f_higher] = f_lower

        # Find the highest floor (a floor that is not a value in directly_below_map)
        floors_below = set(directly_below_map.values())
        highest_floor = None
        for f in all_floors:
            if f not in floors_below:
                highest_floor = f
                break

        # Build ordered list from highest to lowest using direct adjacencies
        ordered_floors_htol = []
        current = highest_floor
        while current is not None:
            ordered_floors_htol.append(current)
            current = directly_below_map.get(current)

        # Reverse for lowest to highest order
        ordered_floors_ltoh = ordered_floors_htol[::-1]

        # Create floor_to_index map
        self.floor_to_index = {f: i for i, f in enumerate(ordered_floors_ltoh)}

        # 2. Parse (destin p f) facts from static to map passengers to destinations
        self.destin_map = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                p, f = get_parts(fact)[1:]
                self.destin_map[p] = f

        # Relevant passengers are those with a defined destination
        self.relevant_passengers = set(self.destin_map.keys())


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # 1. Get current lift floor and index
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break

        # Should always find lift location in a valid state
        if current_floor is None or current_floor not in self.floor_to_index:
             # This indicates an invalid state representation or floor parsing error
             return math.inf # Return infinity for states that seem unreachable or invalid

        current_index = self.floor_to_index[current_floor]

        # 2. Identify unserved passengers
        unserved_passengers = {
            p for p in self.relevant_passengers
            if f"(served {p})" not in state
        }

        # 3. If no unserved passengers, it's a goal state
        if not unserved_passengers:
            return 0

        # 4. Identify required floors (origins for waiting, destinations for boarded)
        required_indices = set()
        num_waiting_unserved = 0
        num_boarded_unserved = 0

        # Find origins for waiting passengers and destinations for boarded passengers
        origin_facts = {fact for fact in state if match(fact, "origin", "*", "*")}
        boarded_facts = {fact for fact in state if match(fact, "boarded", "*")}

        for p in unserved_passengers:
            is_waiting = any(match(fact, "origin", p, "*") for fact in origin_facts)
            is_boarded = any(match(fact, "boarded", p) for fact in boarded_facts)

            if is_waiting:
                # Find the origin floor for this waiting passenger
                origin_floor = None
                for fact in origin_facts:
                    if match(fact, "origin", p, "*"):
                        origin_floor = get_parts(fact)[2]
                        break
                if origin_floor and origin_floor in self.floor_to_index:
                    required_indices.add(self.floor_to_index[origin_floor])
                    num_waiting_unserved += 1
                # else: origin floor not found or not in floor map - problem with state/static?
            elif is_boarded:
                # Passenger is boarded, needs to go to destination
                destin_floor = self.destin_map.get(p)
                if destin_floor and destin_floor in self.floor_to_index:
                    required_indices.add(self.floor_to_index[destin_floor])
                    num_boarded_unserved += 1
                # else: destination not known or not in floor map - problem with static?

        # Remove None from required_indices in case of unknown floors (shouldn't happen in valid problems)
        required_indices.discard(None)

        # 6. Calculate the estimated action cost
        # Each waiting passenger needs board (1) + depart (1) = 2 actions
        # Each boarded passenger needs depart (1) = 1 action
        action_cost = (num_waiting_unserved * 2) + num_boarded_unserved

        # 7. Handle case where required_indices is empty
        if not required_indices:
            # All unserved passengers must be waiting/boarded at the current floor
            # Heuristic is just the action cost needed at the current floor
            return action_cost

        # 8. Handle case where required_indices is not empty
        min_req_index = min(required_indices)
        max_req_index = max(required_indices)

        # Movement cost estimate: distance to cover the range + distance from current to closer end
        movement_cost = (max_req_index - min_req_index) + min(abs(current_index - min_req_index), abs(current_index - max_req_index))

        # Total heuristic
        total_cost = movement_cost + action_cost

        return total_cost
