from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    E.g. "(origin p1 f1)" -> ["origin", "p1", "f1"]
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - fact: The fact string, e.g. "(origin p1 f1)"
    - args: pattern parts, e.g. "origin", "*", "f1"
    Returns True if all parts match, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic function for the Miconic domain.

    Summary:
    This heuristic estimates the minimum number of required actions to serve all
    passengers (i.e., board and deliver them to their destination floors) from
    the current state until all passengers are served. The heuristic guides
    greedy best-first search though it is not guaranteed to be admissible.

    Assumptions:
    - Serving a passenger requires boarding them at their origin floor and departing
      them at their destination floor.
    - The lift can move only one floor up or down per action.
    - The heuristic does not model intermediate constraints such as the order of boarding,
      but approximates effort based on minimum lift movements and necessary board/depart actions.
    - The 'above' facts form a transitive order and floors can be ranked accordingly.

    Heuristic Initialization:
    - Extract the set of floors and build a floor-index map to allow quick distance calculations.
    - Store static facts:
      * The destination floor of each passenger.
      * The origin floor of each passenger.
    - Build a fast lookup of floors' vertical ordering from 'above' static facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the lift's current floor from the state.
    2. Extract the sets of passengers who are:
       - waiting to be boarded (have origin floor but not boarded).
       - boarded but not served.
    3. For each waiting passenger:
       - Determine the lift movement cost to reach their origin floor (from lift current or previous target floor),
         plus one board action.
    4. For each boarded passenger:
       - Determine the lift movement cost to their destination floor plus one depart action.
    5. Since the lift can only move between discrete floors and must visit all relevant floors
       (boarding and destination floors), compute a lower bound on the total lift movement as the minimal
       path covering all relevant floors (approximated by sorting floors and summing distances).
    6. Combine the number of passengers boarding and departing actions and the needed lift moves for an estimate.
    7. Return 0 if all passengers are served (goal state), else the computed estimate.


    This heuristic balances between complexity and informativeness,
    is efficient to compute, and helps minimize expanded nodes during search.
    """

    def __init__(self, task):
        # Extract and store static info for fast lookup during heuristic calls.
        self.goals = task.goals
        static_facts = task.static

        # Extract all floors from static facts (e.g., 'above' facts mention floors)
        floors = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "above":
                floors.add(parts[1])
                floors.add(parts[2])
            elif parts[0] == "destin":
                floors.add(parts[2])
            elif parts[0] == "origin":
                floors.add(parts[2])
        self.floors = sorted(floors)

        # Build floor index map for distance calculation
        self.floor_idx = {floor: i for i, floor in enumerate(self.floors)}

        # Extract passenger -> destination floor mapping
        self.passenger_dest = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "destin":
                passenger, floor = parts[1], parts[2]
                self.passenger_dest[passenger] = floor

        # Extract passenger -> origin floor mapping
        self.passenger_origin = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "origin":
                passenger, floor = parts[1], parts[2]
                self.passenger_origin[passenger] = floor

    def __call__(self, node):
        state = node.state

        # Quick goal check: if all passengers served, heuristic is 0
        if self.goals <= state:
            return 0

        # Find lift current floor
        lift_floor = None
        for fact in state:
            if fact.startswith("(lift-at "):
                parts = get_parts(fact)
                lift_floor = parts[1]
                break
        if lift_floor is None:
            # No lift position found - should not happen, assign heuristic infinity or large value
            return float('inf')

        # Identify passengers' boarded and served status
        boarded_passengers = set()
        served_passengers = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "boarded":
                boarded_passengers.add(parts[1])
            elif parts[0] == "served":
                served_passengers.add(parts[1])

        # Passengers who still need boarding (have origin, not boarded or served)
        waiting_passengers = [
            p for p in self.passenger_origin
            if (f"(origin {p} {self.passenger_origin[p]})" in state) and p not in boarded_passengers and p not in served_passengers
        ]
        # Boarded but not served passengers
        boarded_not_served = [p for p in boarded_passengers if p not in served_passengers]

        # Floors where passengers need to board
        boarding_floors = set(self.passenger_origin[p] for p in waiting_passengers)
        # Floors where boarded passengers must depart
        departing_floors = set(self.passenger_dest[p] for p in boarded_not_served)

        # All relevant floors that lift must visit (excluding current lift floor)
        target_floors = sorted(set(boarding_floors | departing_floors))

        # Calculate lift movement cost heuristic:
        # Approximate minimal travel distance by visiting floors in sorted order,
        # starting from lift current floor and covering all required floors.

        # If no floors to visit, no movement needed
        if not target_floors:
            move_cost = 0
        else:
            def floor_distance(f1, f2):
                # Absolute difference in floor indices
                return abs(self.floor_idx[f1] - self.floor_idx[f2])

            # Compute minimal distance to cover all target floors:
            # We approximate this as the minimal total travel of lift starting
            # from current floor covering all target floors in sorted order,
            # choosing the minimal traversal direction (ascending or descending).

            # Two candidate routes:
            # 1. Move from lift_floor to min target floor, then to max target floor
            dist_asc = floor_distance(lift_floor, target_floors[0]) + \
                       floor_distance(target_floors[0], target_floors[-1])
            # 2. Move from lift_floor to max target floor, then to min target floor
            dist_desc = floor_distance(lift_floor, target_floors[-1]) + \
                        floor_distance(target_floors[-1], target_floors[0])

            move_cost = min(dist_asc, dist_desc)

        # Number of board actions needed (1 per waiting passenger)
        board_actions = len(waiting_passengers)
        # Number of depart actions needed (1 per boarded but not served passenger)
        depart_actions = len(boarded_not_served)

        # Total heuristic estimate: lift moves + boarding + departing
        # Each action counts as one step.
        heuristic_value = move_cost + board_actions + depart_actions

        return heuristic_value
