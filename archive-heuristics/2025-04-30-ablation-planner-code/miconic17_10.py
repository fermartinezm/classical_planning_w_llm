from fnmatch import fnmatch
# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total cost by summing the minimum number of
    board/depart actions required for unserved passengers and the estimated
    minimum lift movement cost to visit all necessary floors (origins for
    waiting passengers, destinations for boarded passengers).

    # Assumptions
    - The lift has a capacity of 1 passenger (implicit from action definitions).
    - Floors are totally ordered, and this order is defined by the `above` facts.
      `(above f_i f_j)` means floor `f_i` is immediately below floor `f_j`.
    - The cost of move, board, and depart actions is 1.

    # Heuristic Initialization
    - Parses the `above` facts from the static information to build a sorted
      list of floors and a mapping from floor name to its numerical index.
    - Parses the `destin` facts from the static information to store the
      destination floor for each passenger.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Current Lift Location:** Find the floor where the lift is currently located using the `(lift-at ?f)` fact in the state. Get its index.

    2.  **Identify Unserved Passengers:** Iterate through all passengers (identified from `destin` facts in static). For each passenger, check if the `(served ?p)` fact is present in the current state. If not, the passenger is unserved.

    3.  **Categorize Unserved Passengers:** For each unserved passenger `p`:
        -   If `(origin ?p ?f_origin)` is in the state, the passenger is waiting at their origin floor `f_origin`. Add `p` to the `Waiting` set and `f_origin` to the `PickupFloors` set.
        -   If `(boarded ?p)` is in the state, the passenger is currently inside the lift. Add `p` to the `Boarded` set. Find their destination floor `f_destin` using the pre-parsed `passenger_destin` map and add `f_destin` to the `DropoffFloors` set.

    4.  **Calculate Action Cost:**
        -   Each passenger currently waiting needs a `board` action (cost 1) and a `depart` action (cost 1). Total = 2 actions.
        -   Each passenger currently boarded needs a `depart` action (cost 1). Total = 1 action.
        -   The total action cost is `2 * |Waiting| + 1 * |Boarded|`.

    5.  **Identify Required Floors to Visit:** The lift must visit all floors in `PickupFloors` (to pick up waiting passengers) and all floors in `DropoffFloors` (to drop off boarded passengers). Create a combined set `FloorsToVisit = PickupFloors union DropoffFloors`.

    6.  **Calculate Movement Cost:**
        -   If `FloorsToVisit` is empty (meaning all relevant passengers are served or there are no unserved passengers requiring stops), the movement cost is 0.
        -   If `FloorsToVisit` is not empty:
            -   Sort the floors in `FloorsToVisit` based on their numerical index.
            -   The minimum movement required to visit all these floors, starting from the current lift floor, is estimated as the distance from the current lift floor to the first floor in the sorted list, plus the sum of distances between consecutive floors in the sorted list. This represents traversing the required floors in a monotonic (up or down) path.
            -   `move_cost = abs(index(current_lift_floor) - index(first_required_floor)) + sum(abs(index(f_i) - index(f_{i+1})))` for consecutive floors `f_i, f_{i+1}` in the sorted list.

    7.  **Total Heuristic Value:** The heuristic value is the sum of the calculated `action_cost` and `move_cost`.

    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        # The set of facts that must hold in goal states. We assume all passengers must be served.
        self.goals = task.goals
        # Static facts are used to determine floor order and passenger destinations.
        static_facts = task.static

        # 1. Parse floor order from `above` facts
        # Build a map from a floor to the floor immediately above it.
        below_to_above = {}
        all_floors = set()
        floors_with_something_below = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above" and len(parts) == 3:
                f_below, f_above = parts[1], parts[2]
                below_to_above[f_below] = f_above
                all_floors.add(f_below)
                all_floors.add(f_above)
                floors_with_something_below.add(f_above)

        # Find the lowest floor (the one not above any other floor, i.e., not in floors_with_something_below)
        lowest_floor = None
        for floor in all_floors:
            if floor not in floors_with_something_below:
                lowest_floor = floor
                break

        # Build the ordered list of floors and the index map
        self.floor_list = []
        self.floor_indices = {}
        current_floor = lowest_floor
        index = 0
        while current_floor is not None:
            self.floor_list.append(current_floor)
            self.floor_indices[current_floor] = index
            current_floor = below_to_above.get(current_floor)
            index += 1

        # Handle case with only one floor or no above facts
        if not self.floor_list and all_floors:
             # If there are floors but no above facts, assume they are all on the same level or order doesn't matter
             # For simplicity, just add them in arbitrary order, index 0
             for floor in all_floors:
                 if floor not in self.floor_indices:
                     self.floor_list.append(floor)
                     self.floor_indices[floor] = 0


        # 2. Store passenger destinations
        self.passenger_destin = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "destin" and len(parts) == 3:
                passenger, destination_floor = parts[1], parts[2]
                self.passenger_destin[passenger] = destination_floor

        # Identify all passengers from static facts (origin or destin)
        self.all_passengers = set(self.passenger_destin.keys())
        for fact in static_facts:
             parts = get_parts(fact)
             if parts and parts[0] == "origin" and len(parts) == 3:
                 self.all_passengers.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify Current Lift Location
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at" and len(parts) == 2:
                current_lift_floor = parts[1]
                break

        if current_lift_floor is None:
             # Should not happen in a valid miconic state, but handle defensively
             return float('inf') # Cannot proceed without lift location

        current_floor_idx = self.floor_indices.get(current_lift_floor, 0) # Default to 0 if floor not found (shouldn't happen)


        # 2. & 3. Identify Unserved Passengers and Required Floors
        waiting_passengers = set()
        boarded_passengers = set()
        pickup_floors = set()
        dropoff_floors = set()
        served_passengers = set()

        # First, find all served passengers
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "served" and len(parts) == 2:
                served_passengers.add(parts[1])

        # Then, identify waiting and boarded among the unserved
        for passenger in self.all_passengers:
            if passenger not in served_passengers:
                is_waiting = False
                is_boarded = False
                passenger_origin = None

                for fact in state:
                    parts = get_parts(fact)
                    if parts and parts[0] == "origin" and len(parts) == 3 and parts[1] == passenger:
                        is_waiting = True
                        passenger_origin = parts[2]
                        break # Found origin, no need to check further facts for this passenger's origin

                for fact in state:
                    parts = get_parts(fact)
                    if parts and parts[0] == "boarded" and len(parts) == 2 and parts[1] == passenger:
                        is_boarded = True
                        break # Found boarded, no need to check further facts for this passenger's boarded status

                if is_waiting:
                    waiting_passengers.add(passenger)
                    if passenger_origin: # Should always be true if is_waiting is true
                        pickup_floors.add(passenger_origin)
                elif is_boarded:
                    boarded_passengers.add(passenger)
                    # Add destination floor to dropoff floors
                    dest_floor = self.passenger_destin.get(passenger)
                    if dest_floor: # Should always exist for a valid problem
                        dropoff_floors.add(dest_floor)
                # Note: A passenger should be either waiting, boarded, or served.
                # If not served and not waiting/boarded, something is wrong or they are not relevant to the goal.
                # Assuming valid states, they must be waiting or boarded if not served.


        # Check if goal is reached (all passengers served)
        if len(served_passengers) == len(self.all_passengers):
             return 0

        # 4. Calculate Action Cost
        # Each waiting passenger needs 1 board + 1 depart = 2 actions
        # Each boarded passenger needs 1 depart = 1 action
        action_cost = 2 * len(waiting_passengers) + 1 * len(boarded_passengers)

        # 5. Identify Required Floors to Visit
        required_stops = pickup_floors.union(dropoff_floors)

        # 6. Calculate Movement Cost
        move_cost = 0
        if required_stops:
            # Sort required floors by index
            sorted_required_floors = sorted(list(required_stops), key=lambda f: self.floor_indices.get(f, 0))

            # Estimate movement cost: distance from current floor to the first required stop,
            # plus distance traversing between consecutive required stops.
            first_stop_floor = sorted_required_floors[0]
            first_stop_idx = self.floor_indices.get(first_stop_floor, 0)
            move_cost += abs(current_floor_idx - first_stop_idx)

            for i in range(len(sorted_required_floors) - 1):
                f1 = sorted_required_floors[i]
                f2 = sorted_required_floors[i+1]
                idx1 = self.floor_indices.get(f1, 0)
                idx2 = self.floor_indices.get(f2, 0)
                move_cost += abs(idx1 - idx2)

        # 7. Total Heuristic Value
        total_cost = action_cost + move_cost

        return total_cost

# Example Usage (requires a dummy Task and Node class)
# class DummyTask:
#     def __init__(self, goals, static):
#         self.goals = goals
#         self.static = static

# class DummyNode:
#     def __init__(self, state):
#         self.state = state

# # Example 1 Test
# task1_static = frozenset({'(destin p1 f2)', '(origin p1 f1)', '(above f1 f2)'})
# task1_goals = frozenset({'(served p1)'})
# task1 = DummyTask(task1_goals, task1_static)

# state1 = frozenset({'(lift-at f2)', '(origin p1 f1)', '(destin p1 f2)', '(above f1 f2)'})
# node1 = DummyNode(state1)

# heuristic1 = miconicHeuristic(task1)
# h_value1 = heuristic1(node1)
# print(f"Heuristic for Example 1: {h_value1}") # Expected: 4

# # Example 2 Test (Partial - just setup)
# task2_static_str = """
#     (destin p1 f20) (destin p2 f15) (destin p3 f6) (destin p4 f17) (destin p5 f6)
#     (destin p6 f17) (destin p7 f11) (destin p8 f5) (destin p9 f7) (destin p10 f13)
#     (above f1 f2) (above f2 f3) (above f3 f4) (above f4 f5) (above f5 f6)
#     (above f6 f7) (above f7 f8) (above f8 f9) (above f9 f10) (above f10 f11)
#     (above f11 f12) (above f12 f13) (above f13 f14) (above f14 f15) (above f15 f16)
#     (above f16 f17) (above f17 f18) (above f18 f19) (above f19 f20)
#     (origin p1 f2) (origin p2 f20) (origin p3 f8) (origin p4 f16) (origin p5 f8)
#     (origin p6 f5) (origin p7 f3) (origin p8 f18) (origin p9 f12) (origin p10 f16)
# """
# task2_static = frozenset(f.strip() for f in task2_static_str.split('\n') if f.strip())
# task2_goals_str = """
#     (served p1) (served p2) (served p3) (served p4) (served p5)
#     (served p6) (served p7) (served p8) (served p9) (served p10)
# """
# task2_goals = frozenset(f.strip() for f in task2_goals_str.split('\n') if f.strip())
# task2 = DummyTask(task2_goals, task2_static)

# state2_str = """
#     (lift-at f19)
#     (origin p1 f2) (origin p2 f20) (origin p3 f8) (origin p4 f16) (origin p5 f8)
#     (origin p6 f5) (origin p7 f3) (origin p8 f18) (origin p9 f12) (origin p10 f16)
# """ # Initial state from example 2
# state2 = frozenset(f.strip() for f in state2_str.split('\n') if f.strip())
# node2 = DummyNode(state2)

# heuristic2 = miconicHeuristic(task2)
# h_value2 = heuristic2(node2)
# print(f"Heuristic for Example 2 (Initial State): {h_value2}") # Expected: 56

# # Example 2 Test (Mid State - p4, p5 boarded, p8, p6 served, lift at f2)
# state2_mid_str = """
#     (lift-at f2)
#     (origin p1 f2) (origin p2 f20) (origin p3 f8) (origin p7 f3) (origin p9 f12) (origin p10 f16)
#     (boarded p4) (boarded p5)
#     (served p8) (served p6)
# """
# state2_mid = frozenset(f.strip() for f in state2_mid_str.split('\n') if f.strip())
# node2_mid = DummyNode(state2_mid)

# heuristic2_mid = miconicHeuristic(task2)
# h_value2_mid = heuristic2_mid(node2_mid)
# print(f"Heuristic for Example 2 (Mid State): {h_value2_mid}")
# # Expected calculation for mid state:
# # Current lift: f2 (idx 1)
# # Unserved: p1, p2, p3, p4, p5, p7, p9, p10
# # Waiting: {p1(f2), p2(f20), p3(f8), p7(f3), p9(f12), p10(f16)} -> 6 passengers
# # Boarded: {p4(f17), p5(f6)} -> 2 passengers
# # Action Cost = 2 * 6 + 1 * 2 = 12 + 2 = 14
# # Pickup Floors: {f2, f20, f8, f3, f12, f16}
# # Dropoff Floors: {f17, f6}
# # Required Stops: {f2, f3, f6, f8, f12, f16, f17, f20}
# # Indices: {1, 2, 5, 7, 11, 15, 16, 19}
# # Sorted Required Floors: [f2, f3, f6, f8, f12, f16, f17, f20]
# # Indices: [1, 2, 5, 7, 11, 15, 16, 19]
# # Current idx: 1 (f2)
# # Move Cost = abs(1 - 1) + abs(1 - 2) + abs(2 - 5) + abs(5 - 7) + abs(7 - 11) + abs(11 - 15) + abs(15 - 16) + abs(16 - 19)
# # Move Cost = 0 + 1 + 3 + 2 + 4 + 4 + 1 + 3 = 18
# # Total h = 14 + 18 = 32
# # print(f"Expected Heuristic for Example 2 (Mid State): 32")

