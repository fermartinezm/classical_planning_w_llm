from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(boarded p1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Summary:
    --------
    This heuristic estimates the minimal number of actions required to serve
    all remaining passengers, i.e. to board passengers at their origin floors,
    transport them to their destination floors, and depart them. The lift can move
    only up or down one floor at a time, board passengers only if it is at the origin floor,
    and depart passengers only if it is at their destination floor.

    Assumptions:
    ------------
    - The lift carries any number of passengers simultaneously once boarded.
    - Passengers not yet boarded must be boarded at their origin floors.
    - Passengers already boarded must be departed at their destination floors.
    - Served passengers require no further actions.
    - Lift movement cost counts the minimal vertical moves to serve all remaining passengers.
    - The heuristic does not have to be admissible, but aims to accurately estimate
      the cost to goal to minimize node expansions while being efficiently computable.

    Heuristic Initialization:
    -------------------------
    - Extract the set of floors and passengers.
    - Build a map of floors with their numeric height to allow distance calculations.
      The "above" predicate describes ordering: if (above f1 f2), then f1 is below f2.
      We deduce the numeric floor order from the transitive closure of "above".
    - Extract for each passenger their origin and destination floors from static facts.
    - The lift’s current floor is extracted from the state.
    - Identify passengers still waiting to be boarded, passengers already boarded (but not served),
      and passengers already served.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. Identify the current floor of the lift.
    2. Extract passengers in three categories:
       - Not boarded: passengers still at origins.
       - Boarded: passengers currently on lift, need to be departed at destinations.
       - Served: passengers already served, no cost needed.
    3. Determine the floors that need to be visited:
       - Origins of not boarded passengers (for boarding).
       - Destinations of boarded passengers (for departing).
    4. Compute the minimal lift travel cost to visit all these floors starting at current floor,
       assuming the lift can travel only up or down one floor step at a time.
       The heuristic estimates this by:
       - Taking the minimal and maximal floor height among all relevant floors plus current floor,
         and computing the vertical distance required to span these floors:
         distance = (max_floor - min_floor)
       - The rationale is that the lift must at least travel to cover all floors with boarding or departing actions.
    5. Add the cost for each boarding and departing action:
       - Each passenger must be boarded once and departed once, i.e. 2 actions per passenger.
       - Served passengers add no cost.
    6. Return the sum of vertical movement cost and passenger action costs as heuristic.
    7. The heuristic is 0 exactly if all passengers are served (goal state).

    This approach is efficient as it uses precomputed floor distances and counts of passengers,
    avoiding expensive pathfinding. It also provides a reasonable approximation of minimal
    action steps required in this elevator domain.
    """

    def __init__(self, task):
        """Initialize heuristic by extracting static facts and preparing floor order."""
        self.goals = task.goals  # All (served p) facts required.
        self.static = task.static

        # Extract all floors appearing in static facts to build floor ordering.
        floors = set()
        above_pairs = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "above":  # (above ?floor1 ?floor2)
                f1, f2 = parts[1], parts[2]
                floors.add(f1)
                floors.add(f2)
                above_pairs.add((f1, f2))
            elif parts[0] == "destin":
                floors.add(parts[2])
            elif parts[0] == "origin":
                floors.add(parts[2])

        # Compute numeric floor levels by topological sorting over floors based on the "above" relation.
        # "above f1 f2" means f1 is below f2, i.e., f2 is at one floor level above f1.
        # We'll assign integer levels starting at the lowest floor as 1, increasing upwards.

        # Build graph to find floor ranks.
        # Create adjacency: edges f1->f2 means f2 is above f1.
        adj = {floor: set() for floor in floors}
        incoming = {floor: 0 for floor in floors}
        for lower, higher in above_pairs:
            adj[lower].add(higher)
            incoming[higher] += 1

        # Kahn's algorithm to topologically order floors from bottom (lowest floor) upwards.
        from collections import deque

        queue = deque([f for f in floors if incoming[f] == 0])
        floor_levels = dict()
        level = 1
        # Because "above" represents strict ordering, start with floors without any incoming edges as lowest floors
        # Assign levels during BFS +1 for each higher floor
        while queue:
            floor = queue.popleft()
            floor_levels[floor] = level
            for neigh in adj[floor]:
                incoming[neigh] -= 1
                if incoming[neigh] == 0:
                    queue.append(neigh)
            level += 1

        # Handling potential disconnected components or cycles:
        # Floors not assigned yet get unique levels after assigning known ones.
        # This covers floors without above relations.
        unassigned = floors - floor_levels.keys()
        for floor in unassigned:
            floor_levels[floor] = level
            level += 1

        # Extract passenger origin and destination mappings from static facts.
        self.passenger_origins = {}
        self.passenger_destinations = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "origin":
                passenger, floor = parts[1], parts[2]
                self.passenger_origins[passenger] = floor
            elif parts[0] == "destin":
                passenger, floor = parts[1], parts[2]
                self.passenger_destinations[passenger] = floor

        # Save floor levels to instance variable for heuristic use.
        self.floor_levels = floor_levels

        # Extract all passengers (those with origin defined).
        self.passengers = set(self.passenger_origins.keys())

    def __call__(self, node):
        """Estimate the minimum number of actions required to serve all passengers from current state."""
        state = node.state

        # Quick goal test: heuristic is zero only if all passengers are served.
        # The goal is that all (served p) hold for all passengers.
        if self.goals <= state:
            return 0

        # Extract current lift floor: (lift-at ?floor)
        lift_floor = None
        for fact in state:
            if fact.startswith("(lift-at "):
                lift_floor = get_parts(fact)[1]
                break
        if lift_floor is None:
            # Defensive: if no lift-at fact, assume arbitrary floor (lowest)
            lift_floor = min(self.floor_levels, key=lambda f: self.floor_levels[f])

        lift_level = self.floor_levels[lift_floor]

        # Identify passenger status categories from state:
        # boarded: (boarded p) in state
        # served:  (served p) in state
        # not boarded: passengers not boarded and not served
        boarded_passengers = set()
        served_passengers = set()
        for p in self.passengers:
            if f"(served {p})" in state:
                served_passengers.add(p)
            elif f"(boarded {p})" in state:
                boarded_passengers.add(p)
        not_boarded_passengers = self.passengers - boarded_passengers - served_passengers

        # Floors that the lift must visit:
        # - origins of not boarded passengers (to board them)
        # - destinations of boarded passengers (to depart them)
        floors_to_visit = set()

        for p in not_boarded_passengers:
            floors_to_visit.add(self.passenger_origins[p])
        for p in boarded_passengers:
            floors_to_visit.add(self.passenger_destinations[p])

        if not floors_to_visit:
            # No passengers waiting to board or depart, so cost is zero
            return 0

        floor_levels_to_visit = [self.floor_levels[floor] for floor in floors_to_visit]
        # Include current lift floor level to reflect starting position
        floor_levels_to_visit.append(lift_level)

        min_floor = min(floor_levels_to_visit)
        max_floor = max(floor_levels_to_visit)

        # Estimated minimum movement actions:
        # The lift must at least travel across the vertical span of all floors involving passenger activity
        movement_cost = max_floor - min_floor

        # Boarding and departing costs:
        # Each passenger not boarded must be boarded once
        # Each boarded passenger must be departed once
        boarding_departing_cost = len(not_boarded_passengers) + len(boarded_passengers)

        # Total heuristic cost: movement + passenger actions
        return movement_cost + boarding_departing_cost
