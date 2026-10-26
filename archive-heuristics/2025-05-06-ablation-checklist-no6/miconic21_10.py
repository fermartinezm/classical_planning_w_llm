from heuristics.heuristic_base import Heuristic
# Assuming Heuristic base class is available in heuristics.heuristic_base

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the number of actions required to reach the goal state
    (all passengers served).

    Heuristic = (Number of waiting passengers) +
                (Number of boarded passengers) +
                (Estimated movement cost)

    Estimated movement cost:
    Calculated as the minimum distance from the current elevator floor
    to any required floor, plus the total vertical span of all required floors.
    Required floors are the origin floors of waiting passengers and the
    destination floors of boarded passengers.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Floor ordering to map floor names to levels.
        - Passenger origins and destinations.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state
        self.all_facts = task.facts # Contains all possible ground facts

        # 1. Collect all floor objects and passenger objects
        self.all_floors = set()
        self.all_passengers = set()
        for fact_str in self.all_facts:
            parts = get_parts(fact_str)
            predicate = parts[0]
            if predicate == 'origin':
                if len(parts) > 2: # Ensure fact has expected structure
                    self.all_passengers.add(parts[1])
                    self.all_floors.add(parts[2])
            elif predicate == 'destin':
                 if len(parts) > 2: # Ensure fact has expected structure
                    self.all_passengers.add(parts[1])
                    self.all_floors.add(parts[2])
            elif predicate == 'above':
                 if len(parts) > 2: # Ensure fact has expected structure
                    self.all_floors.add(parts[1])
                    self.all_floors.add(parts[2])
            elif predicate in ['boarded', 'served']:
                 if len(parts) > 1: # Ensure fact has expected structure
                    self.all_passengers.add(parts[1])
            elif predicate == 'lift-at':
                 if len(parts) > 1: # Ensure fact has expected structure
                    self.all_floors.add(parts[1])

        # 2. Build floor_to_level map based on 'above' predicates
        # Level of a floor f is the number of floors f' such that (above f' f) is true.
        self.floor_to_level = {}
        for f_i in self.all_floors:
            count = 0
            for f_j in self.all_floors:
                if f_i != f_j and '(above {} {})'.format(f_j, f_i) in self.static_facts:
                    count += 1
            self.floor_to_level[f_i] = count

        # 3. Build origin_map and destin_map
        self.origin_map = {}
        self.destin_map = {}
        # Origins are typically in the initial state
        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'origin' and len(parts) > 2:
                self.origin_map[parts[1]] = parts[2]
        # Destinations are typically static
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'destin' and len(parts) > 2:
                self.destin_map[parts[1]] = parts[2]
        # Note: Destinations might also be implicitly defined by goal facts like (served p)
        # and corresponding (destin p d) facts in static/initial state.
        # We rely on destin facts being present in static or initial state.


    def __call__(self, node):
        """Compute the domain-dependent heuristic value for the given state."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # 1. Find current elevator floor
        current_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at' and len(parts) > 1:
                current_floor = parts[1]
                break
        # In a valid miconic state, the lift must be at some floor.
        # If not found, it indicates an invalid state representation.
        # We assume valid states are provided.

        current_floor_level = self.floor_to_level.get(current_floor, -1) # Use .get for safety

        # 2. Identify unserved, waiting, and boarded passengers
        unserved_passengers = set()
        waiting_passengers_count = 0
        boarded_passengers_count = 0
        required_floors = set()

        # Collect all 'served' facts from the goal and current state for efficient lookup
        goal_served_facts = {str(g) for g in self.goals if get_parts(str(g))[0] == 'served'}
        state_served_facts = {str(s) for s in state if get_parts(str(s))[0] == 'served'}

        # Identify unserved passengers based on the goal
        for p in self.all_passengers:
            goal_served_p = '(served {})'.format(p)
            if goal_served_p in goal_served_facts and goal_served_p not in state_served_facts:
                 unserved_passengers.add(p)

        # Categorize unserved passengers and identify required floors
        for p in unserved_passengers:
            origin_p = self.origin_map.get(p)
            destin_p = self.destin_map.get(p)

            # Check if passenger p is waiting at origin
            if origin_p and '(origin {} {})'.format(p, origin_p) in state:
                waiting_passengers_count += 1
                if origin_p in self.floor_to_level: # Ensure floor is valid
                    required_floors.add(origin_p)

            # Check if passenger p is boarded
            elif '(boarded {})'.format(p) in state:
                boarded_passengers_count += 1
                if destin_p and destin_p in self.floor_to_level: # Ensure destination floor is valid
                    required_floors.add(destin_p)
                # else: boarded passenger with no known destination? Invalid problem?

            # Note: Passengers not waiting or boarded but unserved must be in an invalid state
            # (e.g., dropped off at wrong floor, or origin/destin facts missing).
            # We only count waiting and boarded passengers towards the base cost.

        # 3. Calculate base cost (board/depart actions needed for waiting/boarded)
        base_cost = waiting_passengers_count + boarded_passengers_count

        # 4. Calculate movement cost
        movement_cost = 0
        if required_floors:
            required_levels = {self.floor_to_level[f] for f in required_floors if f in self.floor_to_level}
            if required_levels: # Ensure there are valid required levels
                min_level = min(required_levels)
                max_level = max(required_levels)

                # Distance from current floor to the closest required floor
                # Handle case where current_floor_level is -1 (invalid state)
                if current_floor_level != -1:
                    min_dist_to_req = min(abs(current_floor_level - level) for level in required_levels)
                else:
                    # If current floor is unknown, assume maximum possible distance to reach the zone
                    min_dist_to_req = max_level - min_level # Or a large constant

                # Vertical span of required floors
                span_req = max_level - min_level

                # Movement cost estimate: distance to reach the zone + distance to traverse the zone
                movement_cost = min_dist_to_req + span_req
            # else: required_floors had names not in floor_to_level? Invalid problem?

        # Total heuristic value
        total_heuristic = base_cost + movement_cost

        return total_heuristic
