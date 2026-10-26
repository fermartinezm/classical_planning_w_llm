from heuristics.heuristic_base import Heuristic
from task import Operator, Task


class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        This heuristic estimates the remaining effort to serve all goal passengers.
        It sums the number of board/depart actions needed for unserved passengers
        currently waiting or boarded, and adds an estimate of the lift movement
        required to visit all floors where pickups or dropoffs are needed.

    Assumptions:
        - The 'above' predicates in the static information define a total linear
          order of floors.
        - 'destin' facts for all relevant passengers are available in the static
          information.
        - In any state reachable in a solvable problem instance, any unserved
          goal passenger is either waiting at their origin (represented by an
          '(origin ?p ?f)' fact) or is boarded (represented by a '(boarded ?p)'
          fact). Passengers who are unserved and neither waiting nor boarded
          are considered unreachable from this state (and the heuristic returns
          infinity).
        - Valid states always contain exactly one '(lift-at ?f)' fact.

    Heuristic Initialization:
        The constructor processes the static information from the task.
        It builds a mapping from floor names to numerical indices based on the
        'above' predicates to quickly calculate floor distances.
        It stores the destination floor for each passenger from the 'destin' facts.
        It identifies the set of all goal passengers.

    Step-By-Step Thinking for Computing Heuristic:
        1. Identify the current floor of the lift from the state facts. If not found,
           and there are unserved goal passengers, return infinity.
        2. Identify which goal passengers are currently served.
        3. Determine the set of unserved goal passengers. If this set is empty,
           the goal is reached, and the heuristic value is 0.
        4. For each unserved goal passenger, check if they are waiting at their
           origin (by looking for an '(origin ?p ?f)' fact in the state) or
           boarded (by looking for a '(boarded ?p)' fact in the state). If an
           unserved goal passenger is neither waiting nor boarded, return infinity.
        5. Count the number of unserved passengers who are waiting. This contributes
           to the heuristic as they each require a 'board' action.
        6. Count the number of unserved passengers who are boarded. This contributes
           to the heuristic as they each require a 'depart' action.
        7. Identify the set of 'required floors'. These are the origin floors
           for all unserved waiting passengers and the destination floors for
           all unserved boarded passengers.
        8. If there are no required floors, the estimated moves cost is 0.
        9. If there are required floors, calculate the estimated moves cost:
           Find the minimum and maximum floor indices among the required floors.
           The estimated moves is the vertical distance from the current lift
           floor index to the closest of the min/max required floor indices,
           plus the total vertical span between the min and max required floor indices.
           Estimated moves = min(|current_idx - min_req_idx|, |current_idx - max_req_idx|)
                             + (max_req_idx - min_req_idx).
           Handle cases where a required floor is not in the floor index map by
           returning infinity (indicates inconsistent problem definition).
        10. The total heuristic value is the sum of the counts from step 5 and 6
            (board/depart actions) and the estimated moves cost from step 9.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task # Store task for access to goals, static, etc.

        # --- Heuristic Initialization ---
        self.floor_to_index = {}
        self.index_to_floor = {}
        self.passenger_destin = {}

        # Parse destin facts from static info
        destin_facts = [fact for fact in task.static if fact.startswith('(destin ')]
        for fact in destin_facts:
            # fact is like '(destin p1 f2)'
            parts = fact.strip('()').split()
            p = parts[1]
            f = parts[2]
            self.passenger_destin[p] = f

        # Parse above facts and build floor order
        above_facts = [fact for fact in task.static if fact.startswith('(above ')]
        all_floors = set()
        floors_with_something_below = set() # Floors f such that (above ?x f) exists
        for fact in above_facts:
            # fact is like '(above f1 f2)'
            parts = fact.strip('()').split()
            f_below = parts[1]
            f_above = parts[2]
            all_floors.add(f_below)
            all_floors.add(f_above)
            floors_with_something_below.add(f_above)

        # Find the lowest floor (a floor not in floors_with_something_below)
        # Assuming there is exactly one lowest floor in a connected floor structure
        lowest_floor = None
        potential_lowest = all_floors - floors_with_something_below
        if potential_lowest:
             lowest_floor = potential_lowest.pop()
        # If no lowest floor found but there are floors, it indicates a cycle or disconnected graph
        # We proceed assuming a linear structure is intended.

        ordered_floors = []
        if lowest_floor:
            # Build the ordered list of floors starting from the lowest
            ordered_floors = [lowest_floor]
            current_floor = lowest_floor
            while len(ordered_floors) < len(all_floors):
                next_floor = None
                # Find the floor f_next such that (above current_floor f_next) is true
                for fact in above_facts:
                    parts = fact.strip('()').split()
                    f_below = parts[1]
                    f_above = parts[2]
                    if f_below == current_floor and f_above not in ordered_floors:
                         next_floor = f_above
                         break # Assuming only one floor is directly above
                if next_floor:
                    ordered_floors.append(next_floor)
                    current_floor = next_floor
                else:
                    # Should only happen when we reached the highest floor or if floors are disconnected
                    break

        # Map floor names to indices
        for i, floor in enumerate(ordered_floors):
            self.floor_to_index[floor] = i
            self.index_to_floor[i] = floor

        # Store goal passengers for quick lookup
        self.goal_passengers = set()
        for goal_fact in task.goals:
            # Goal facts are like '(served p1)'
            parts = goal_fact.strip('()').split()
            if parts[0] == 'served':
                self.goal_passengers.add(parts[1])

    def __call__(self, node):
        """
        Computes the miconic heuristic for the given state.

        Keyword arguments:
        node -- the current state node
        """
        state = node.state

        # --- Step-By-Step Thinking for Computing Heuristic ---

        # 1. Identify current lift floor
        current_floor = None
        for fact in state:
            if fact.startswith('(lift-at '):
                current_floor = fact.strip('()').split()[1]
                break

        # If lift-at is missing, the state is likely invalid or a dead end.
        # If there are goals but no lift, it's probably unreachable.
        if current_floor is None:
             if self.goal_passengers:
                 return float('inf')
             else:
                 # No goals, no lift? Goal is empty, already reached.
                 return 0

        # 2. Identify which goal passengers are currently served
        served_passengers_in_state = {fact.strip('()').split()[1] for fact in state if fact.startswith('(served ')}

        # 3. Determine the set of unserved goal passengers
        unserved_goal_passengers = {p for p in self.goal_passengers if p not in served_passengers_in_state}

        # If this set is empty, the goal is reached, and the heuristic value is 0.
        if not unserved_goal_passengers:
            return 0

        # 4. For each unserved goal passenger, find their state (waiting or boarded)
        waiting_passengers_unserved = [] # List of (passenger, origin_floor)
        boarded_passengers_unserved = [] # List of passenger

        # Build maps for quick lookup of origin and boarded facts in the current state
        origin_facts_map = {} # {passenger: floor}
        boarded_passengers_set = set() # {passenger}

        for fact in state:
            parts = fact.strip('()').split()
            predicate = parts[0]
            if predicate == 'origin':
                origin_facts_map[parts[1]] = parts[2]
            elif predicate == 'boarded':
                boarded_passengers_set.add(parts[1])

        # Categorize unserved goal passengers
        for p in unserved_goal_passengers:
            if p in origin_facts_map:
                waiting_passengers_unserved.append((p, origin_facts_map[p]))
            elif p in boarded_passengers_set:
                boarded_passengers_unserved.append(p)
            else:
                # This passenger is unserved, not waiting, and not boarded.
                # Based on assumptions, this state might be a dead end for this passenger.
                # Return infinity.
                # print(f"Warning: Unserved goal passenger {p} is neither waiting nor boarded.")
                return float('inf') # Indicate unreachable goal

        # 5. Count waiting unserved passengers (board actions needed)
        # 6. Count boarded unserved passengers (depart actions needed)
        h_base = len(waiting_passengers_unserved) + len(boarded_passengers_unserved)

        # 7. Identify required floors
        pickup_floors = {f for (p, f) in waiting_passengers_unserved}
        dropoff_floors = {self.passenger_destin[p] for p in boarded_passengers_unserved}
        target_floors = pickup_floors | dropoff_floors

        # 8. Calculate estimated moves
        h_moves = 0
        if target_floors:
            # Ensure all target floors are in our floor index map (should be if parsing is correct)
            try:
                target_indices = {self.floor_to_index[f] for f in target_floors}
                min_target_idx = min(target_indices)
                max_target_idx = max(target_indices)
                current_idx = self.floor_to_index[current_floor]

                # 9. Estimated moves calculation
                # Distance from current floor to the closest end of the required range
                dist_to_min = abs(current_idx - min_target_idx)
                dist_to_max = abs(current_idx - max_target_idx)
                dist_to_closest_end = min(dist_to_min, dist_to_max)

                # Total vertical span of the required floors
                span = max_target_idx - min_target_idx

                h_moves = dist_to_closest_end + span

            except KeyError as e:
                 # This might happen if a floor from state/destin was not in above facts
                 # Indicates inconsistent problem definition or state. Treat as unreachable.
                 # print(f"Error: Floor {e} not found in floor index map.")
                 return float('inf')


        # 10. Total heuristic value
        h_value = h_base + h_moves

        return h_value
