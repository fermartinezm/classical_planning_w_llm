import re
from heuristics.heuristic_base import Heuristic
from task import Task


class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    Estimates the cost to reach the goal by summing the estimated lift movement
    cost and the number of required board and depart actions.
    The movement cost is estimated as the minimum distance the lift must travel
    to visit all floors where passengers need to be picked up or dropped off.
    This is calculated as the distance from the current lift floor to the
    nearest required stop floor (either the lowest or highest required floor),
    plus the distance spanning the range of all required stop floors.
    The action cost is estimated as the number of passengers waiting at their
    origin (requiring a board action) plus the number of passengers boarded
    (requiring a depart action).

    Assumptions:
    - The 'above' predicates define a total order on floors.
    - Unserved passengers are either at their origin or boarded.
    - The lift can move between adjacent floors (defined by 'above').
    - Action costs are 1.
    - Floor names can be reliably extracted from fact strings (e.g., '(lift-at f2)' -> 'f2').
    - All floors relevant to the problem (mentioned in init, goal, static) are part of the 'above' hierarchy.
    - Problem instances are well-formed according to the domain rules (e.g., lift is always at exactly one floor, passengers have valid origins/destinations).

    Heuristic Initialization:
    - Parses 'above' predicates from static facts to create a mapping from
      floor names to integer levels. The lowest floor is level 1.
    - Parses 'destin' predicates from static facts to map passengers to their
      destination floors.
    - Stores the set of goal passengers (those who need to be 'served').

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is a goal state (all goal passengers served). If yes, return 0.
    2. Iterate through the state facts to identify:
       - The current floor of the lift.
       - Unserved passengers who are at their origin and their origin floor.
       - Unserved passengers who are boarded.
       - Passengers who are served (to exclude them from unserved counts).
    3. Based on the findings from step 2, determine the set of floors where passengers need to be picked up (origin floors of passengers at origin) and the set of floors where passengers need to be dropped off (destination floors of boarded passengers, looked up from pre-parsed static destin facts).
    4. Combine these into a set of required stop floors.
    5. Validate state consistency: Check if the lift location was found and if all unserved goal passengers are accounted for (either at origin or boarded). If not, return infinity (indicating an invalid or unsolvable state from this point).
    6. If the set of required stop floors is empty, the movement cost is 0.
    7. If the set of required stop floors is not empty:
       - Map the required stop floors to their integer levels using the pre-calculated floor-to-level map.
       - Find the minimum and maximum levels among the required stop floors.
       - Get the level of the current lift floor using the floor-to-level map.
       - Calculate the estimated movement cost:
         - Distance from current level to the minimum required level: `abs(current_level - min_required_level)`
         - Distance from current level to the maximum required level: `abs(current_level - max_required_level)`
         - Distance spanning the required range: `max_required_level - min_required_level`
         - Movement cost = `min(distance_to_min, distance_to_max) + distance_spanning_range`
    8. Calculate the estimated action cost:
       - Number of passengers at origin (requiring board): `|P_at_origin|`
       - Number of boarded passengers (requiring depart): `|P_boarded|`
       - Action cost = `|P_at_origin| + |P_boarded|`
    9. The heuristic value is the sum of the estimated movement cost and the estimated action cost.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task

        # --- Heuristic Initialization ---

        # 1. Map floors to levels based on 'above' predicates
        self.floor_to_level = self._build_floor_level_map(task.static)

        # 2. Map passengers to destinations based on 'destin' predicates
        self.passenger_to_destin = {}
        for fact in task.static:
            if fact.startswith('(destin '):
                # Example: '(destin p1 f2)'
                parts = fact.split(' ')
                if len(parts) == 3:
                    p = parts[1]
                    f_destin = parts[2][:-1] # Remove ')'
                    self.passenger_to_destin[p] = f_destin

        # 3. Store goal passengers
        self.goal_passengers = set()
        for goal in task.goals:
            if goal.startswith('(served '):
                # Example: '(served p1)'
                p = goal.split(' ')[1][:-1] # Remove ')'
                self.goal_passengers.add(p)

    def _build_floor_level_map(self, static_facts):
        """Builds a map from floor name to integer level based on 'above' facts."""
        floor_below_counts = {}
        all_floors_in_above = set()

        # Collect all floors mentioned in 'above' facts
        for fact in static_facts:
            if fact.startswith('(above '):
                parts = fact.split(' ')
                if len(parts) == 3:
                    f1 = parts[1] # Floor above
                    f2 = parts[2][:-1] # Floor below
                    all_floors_in_above.add(f1)
                    all_floors_in_above.add(f2)

        # Initialize counts for all found floors
        for floor in all_floors_in_above:
            floor_below_counts[floor] = 0

        # Count floors below each floor based on 'above'
        for fact in static_facts:
            if fact.startswith('(above '):
                parts = fact.split(' ')
                if len(parts) == 3:
                    f1 = parts[1] # Floor above
                    # f2 = parts[2][:-1] # Floor below
                    # f1 is above f2, so f2 is below f1. Increment count for f1.
                    if f1 in floor_below_counts: # Should always be true if collected correctly
                         floor_below_counts[f1] += 1

        # Assign levels: floor with 0 below is level 1, etc.
        # Sort floors by their below count to get the order from lowest to highest
        sorted_floors = sorted(floor_below_counts.items(), key=lambda item: item[1])

        floor_to_level = {}
        # Assign levels based on the sorted order (rank)
        for level, (floor, count) in enumerate(sorted_floors):
             floor_to_level[floor] = level + 1

        return floor_to_level


    def __call__(self, node):
        """
        Computes the miconic heuristic for the given state.
        """
        state = node.state

        # 1. Check if goal is reached
        if self.task.goal_reached(state):
             return 0

        # 2. Find current lift floor and identify unserved passengers' status and required stops in one pass
        current_floor = None
        P_at_origin = set()
        P_boarded = set()
        S_pickup = set()
        S_dropoff = set()

        # Pre-calculate served passengers for quick lookup
        served_passengers_in_state = {fact.split(' ')[1][:-1] for fact in state if fact.startswith('(served ')}

        for fact in state:
            if fact.startswith('(lift-at '):
                parts = fact.split(' ')
                if len(parts) == 2:
                    current_floor = parts[1][:-1] # Extract floor name
            elif fact.startswith('(origin '):
                parts = fact.split(' ')
                if len(parts) == 3:
                    p = parts[1]
                    f = parts[2][:-1]
                    # Check if this passenger is a goal passenger and not yet served
                    if p in self.goal_passengers and p not in served_passengers_in_state:
                        P_at_origin.add(p)
                        S_pickup.add(f)
            elif fact.startswith('(boarded '):
                parts = fact.split(' ')
                if len(parts) == 2:
                    p = parts[1][:-1]
                     # Check if this passenger is a goal passenger and not yet served
                    if p in self.goal_passengers and p not in served_passengers_in_state:
                        P_boarded.add(p)
                        # Add destination floor to dropoff stops
                        if p in self.passenger_to_destin:
                            S_dropoff.add(self.passenger_to_destin[p])
                        else:
                            # Boarded passenger without destination? Invalid problem.
                            return float('inf')

        # Validate state consistency:
        # - Lift must be at exactly one floor.
        # - All goal passengers must be either served, at origin, or boarded.
        # We already checked served via task.goal_reached.
        # Check lift location
        if current_floor is None or current_floor not in self.floor_to_level:
             return float('inf') # Cannot compute heuristic without lift location/level

        current_level = self.floor_to_level[current_floor]

        # Check if all unserved goal passengers are accounted for (either at origin or boarded)
        # Get the set of all unserved goal passengers
        P_unserved_goal = self.goal_passengers - served_passengers_in_state
        P_unserved_accounted = P_at_origin.union(P_boarded)

        if len(P_unserved_accounted) != len(P_unserved_goal):
             # This means some unserved goal passenger is neither at origin nor boarded. Invalid state.
             return float('inf')

        # 6. Combine required stops
        S_required = S_pickup.union(S_dropoff)

        # 7. Calculate movement cost
        movement_cost = 0
        if S_required: # Only calculate movement if there are stops needed
             required_levels = {self.floor_to_level[f] for f in S_required if f in self.floor_to_level}

             if not required_levels:
                 # Required floors are not in the floor_to_level map? Invalid state/setup.
                 return float('inf')

             min_level_R = min(required_levels)
             max_level_R = max(required_levels)

             # Movement cost calculation
             dist_to_min = abs(current_level - min_level_R)
             dist_to_max = abs(current_level - max_level_R)
             range_dist = max_level_R - min_level_R

             movement_cost = min(dist_to_min, dist_to_max) + range_dist
        # If S_required is empty, movement_cost remains 0, which is correct.

        # 8. Calculate action cost
        action_cost = len(P_at_origin) + len(P_boarded)

        # 9. Total heuristic value
        h_value = movement_cost + action_cost

        return h_value
