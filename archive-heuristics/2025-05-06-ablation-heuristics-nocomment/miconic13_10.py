from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper function to parse a PDDL fact string into a list of parts."""
    # Removes surrounding parentheses and splits by space
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        Estimates the cost to reach the goal by summing the estimated lift
        movement cost and the number of pending board/depart actions.
        The move cost is estimated as the distance required to travel from the
        current lift floor to the range of floors that need to be visited
        (origin floors for waiting passengers, destination floors for boarded
        passengers), plus the distance to traverse that range. The action
        cost is simply the count of passengers waiting to board or currently
        boarded (each requiring at least one more action).

    Assumptions:
        - The 'above' predicates define a linear ordering of floors, and floor
          names like 'f1', 'f2', etc., correspond to this order such that
          lexicographical sorting matches the floor levels. 'f1' is the lowest,
          'f2' is the next, and so on. This assumption simplifies floor leveling.
        - Action costs are uniform (implicitly 1).

    Heuristic Initialization:
        - Parses static facts to build a mapping from floor names to integer
          levels based on lexicographical sorting of all floor objects found
          in static facts (identified by names starting with 'f' followed by digits).
        - Stores passenger destination floors from 'destin' facts found in
          static facts.
        - Stores the set of passengers that need to be served, identified
          from the 'served' predicates in the task's goal state.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is the goal state (all goal passengers
           are served). If yes, the heuristic value is 0.
        2. Identify the current floor of the lift by finding the fact
           '(lift-at ?f)' in the current state.
        3. Identify passengers who are currently waiting at their origin floor.
           These are passengers 'p' for whom '(origin p ?f)' is in the state
           and '(served p)' is not in the state. Store their origin floors.
        4. Identify passengers who are currently boarded in the lift. These are
           passengers 'p' for whom '(boarded p)' is in the state and
           '(served p)' is not in the state. Use the pre-calculated
           passenger destinations to find their destination floors.
        5. Determine the set of floors the lift *must* visit to make progress
           towards serving the unserved passengers. This set includes:
           - The origin floor for each waiting unserved passenger.
           - The destination floor for each boarded unserved passenger.
        6. If there are no required stops (meaning no unserved passengers are
           waiting or boarded), the heuristic is 0 (this case is covered by
           step 1, but this check acts as a safeguard).
        7. If there are required stops:
           - Get the floor levels for the current lift floor and all required
             stop floors using the map built during initialization.
           - Find the minimum and maximum floor levels among the required stops.
           - Calculate the estimated move cost: This is the sum of the vertical
             distance from the current lift level to the nearest required level
             (either the minimum or maximum required level) and the total
             vertical range covered by the required levels (maximum required
             level - minimum required level).
             `move_cost = (max_required_level - min_required_level) + min(abs(current_lift_level - min_required_level), abs(current_lift_level - max_required_level))`
        8. Calculate the estimated action cost: This is the sum of the number
           of waiting unserved passengers (each needs a 'board' action) and
           the number of boarded unserved passengers (each needs a 'depart'
           action).
           `action_cost = num_waiting_unserved_passengers + num_boarded_unserved_passengers`
        9. The total heuristic value for the state is the sum of the estimated
           move cost and the estimated action cost.
           `h = move_cost + action_cost`
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # 1. Build floor levels map based on lexicographical sort
        all_floors = set()
        # Collect all floor objects mentioned in static facts
        for fact in static_facts:
             parts = get_parts(fact)
             if len(parts) > 1: # Predicate has arguments
                 for arg in parts[1:]:
                     # Assume floors start with 'f' followed by digits
                     if isinstance(arg, str) and arg.startswith('f') and arg[1:].isdigit():
                          all_floors.add(arg)

        sorted_floors = sorted(list(all_floors))
        self.floor_levels = {floor: i + 1 for i, floor in enumerate(sorted_floors)}

        # 2. Store passenger destinations
        self.passenger_destinations = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'destin':
                person, floor = parts[1], parts[2]
                self.passenger_destinations[person] = floor

        # 3. Store goal passengers (all passengers who need to be served)
        # These are the passengers mentioned in the goal state (served ?p)
        self.goal_passengers = set()
        for goal_fact in self.goals:
             parts = get_parts(goal_fact)
             if parts[0] == 'served':
                 self.goal_passengers.add(parts[1])


    def __call__(self, node):
        state = node.state

        # 1. Check if goal state
        # The goal is reached if all goal passengers are served.
        all_served = True
        for passenger in self.goal_passengers:
            if f'(served {passenger})' not in state:
                all_served = False
                break
        if all_served:
            return 0

        # 2. Identify current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at':
                current_lift_floor = parts[1]
                break
        # current_lift_floor should always be found in a valid state

        # 3. Identify waiting passengers
        waiting_passengers = {} # {passenger: origin_floor}
        # 4. Identify boarded passengers
        boarded_passengers = set()

        # Iterate through state facts to find waiting and boarded unserved passengers
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'origin':
                person, floor = parts[1], parts[2]
                # Check if this passenger is a goal passenger and not yet served
                if person in self.goal_passengers and f'(served {person})' not in state:
                     waiting_passengers[person] = floor
            elif parts[0] == 'boarded':
                person = parts[1]
                 # Check if this passenger is a goal passenger and not yet served
                if person in self.goal_passengers and f'(served {person})' not in state:
                    boarded_passengers.add(person)

        # 5. Determine required stops
        pickup_floors = set(waiting_passengers.values())
        dropoff_floors = {self.passenger_destinations[p] for p in boarded_passengers}
        required_stops = pickup_floors | dropoff_floors

        # 6. If no required stops, return 0 (should be covered by goal check)
        if not required_stops:
             return 0 # Should only happen if all goal passengers are served

        # 7. Calculate move cost
        current_lift_level = self.floor_levels[current_lift_floor]
        required_levels = {self.floor_levels[f] for f in required_stops}
        min_required_level = min(required_levels)
        max_required_level = max(required_levels)

        # Cost to reach the range + cost to traverse the range
        move_cost = (max_required_level - min_required_level) + min(abs(current_lift_level - min_required_level), abs(current_lift_level - max_required_level))

        # 8. Calculate action cost
        boarding_cost = len(waiting_passengers)
        departing_cost = len(boarded_passengers)
        action_cost = boarding_cost + departing_cost

        # 9. Total heuristic
        total_heuristic = move_cost + action_cost

        return total_heuristic
