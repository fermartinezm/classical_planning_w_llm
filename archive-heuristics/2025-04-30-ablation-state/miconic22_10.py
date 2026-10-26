from heuristics.heuristic_base import Heuristic
# No need for fnmatch as we parse facts manually

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential whitespace issues
    return fact.strip()[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers.
    It counts the required board and depart actions for unserved passengers
    and adds an estimate of the lift movement cost to visit the necessary floors.

    # Assumptions
    - Floors are linearly ordered as defined by the 'above' predicates.
    - The lift can carry multiple passengers.
    - The cost of board, depart, up, and down actions is 1.
    - The 'above' predicate (above f_higher f_lower) means f_higher is physically above f_lower.
      The 'up' action moves from f_higher to f_lower, and 'down' from f_lower to f_higher,
      which is counter-intuitive but handled by using physical floor levels.

    # Heuristic Initialization
    - Parses 'above' facts from static information to build a mapping from floor names to numerical levels.
    - Parses 'destin' facts from static information to know the destination floor for each passenger.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift from the state.
    2. Identify which passengers are already served from the state.
    3. For each passenger not yet served:
       - If the passenger is waiting at their origin floor (indicated by an 'origin' fact in the state):
         - Increment the count of required 'board' actions (cost 1).
         - Add the passenger's origin floor to the set of floors the lift must visit.
       - If the passenger is boarded in the lift (indicated by a 'boarded' fact in the state):
         - Increment the count of required 'depart' actions (cost 1).
         - Find the passenger's destination floor using the pre-parsed destination information.
         - Add the passenger's destination floor to the set of floors the lift must visit.
    4. Calculate the total number of required 'board' and 'depart' actions.
    5. Calculate the estimated movement cost:
       - Determine the numerical levels for all floors the lift must visit using the pre-built floor level map.
       - If there are no floors to visit, the movement cost is 0.
       - Otherwise, find the minimum and maximum levels among the required floors.
       - Find the numerical level of the lift's current floor.
       - The estimated movement cost is the distance from the current floor level to the closest required floor level (either the min or max level of the required range), plus the total span of the required floor levels (max level - min level). This estimates the minimum number of move actions required to reach the closest necessary floor and then traverse the entire range of necessary floors.
    6. The total heuristic value is the sum of the required 'board' actions, required 'depart' actions, and the estimated movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations from static facts.
        """
        self.goals = task.goals # Store goals, although not directly used in this heuristic calculation.
        static_facts = task.static

        # Build floor level mapping from 'above' predicates
        # (above f_higher f_lower) means f_higher is directly above f_lower
        above_map = {} # f_lower -> f_higher
        all_floors = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'above':
                f_higher, f_lower = parts[1], parts[2]
                above_map[f_lower] = f_higher
                all_floors.add(f_higher)
                all_floors.add(f_lower)

        self.floor_level = {}
        if all_floors:
            floors_below_others = set(above_map.keys())
            floors_above_others = set(above_map.values())

            # The lowest floor is one that is below another floor, but no floor is below it.
            # It's a key in above_map, but not a value in above_map.
            potential_lowest = floors_below_others - floors_above_others

            lowest_floor = None
            if len(potential_lowest) == 1:
                 lowest_floor = potential_lowest.pop()
            elif len(all_floors) == 1: # Case with only one floor
                 lowest_floor = list(all_floors)[0]
            # Note: Assumes a single linear chain of floors.

            if lowest_floor:
                current_floor = lowest_floor
                level = 1
                # Traverse upwards from the lowest floor, assigning levels
                while current_floor in above_map:
                     self.floor_level[current_floor] = level
                     current_floor = above_map[current_floor]
                     level += 1
                # Add the highest floor which is not a key in above_map
                self.floor_level[current_floor] = level
            # Note: If lowest_floor is None, floor_level remains empty.
            # The heuristic calculation handles cases where floor levels might be missing.


        # Store passenger destinations from 'destin' facts
        self.passenger_destin = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'destin':
                passenger, floor = parts[1], parts[2]
                self.passenger_destin[passenger] = floor

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal state.
        """
        state = node.state

        served_passengers = set()
        current_floor = None
        waiting_passengers_info = {} # {passenger: origin_floor}
        boarded_passengers = set()

        # First pass to identify served passengers and lift location
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'served':
                served_passengers.add(parts[1])
            elif parts[0] == 'lift-at':
                current_floor = parts[1]

        # Second pass to identify waiting and boarded passengers (excluding served)
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'origin' and parts[1] not in served_passengers:
                passenger, floor = parts[1], parts[2]
                waiting_passengers_info[passenger] = floor
            elif parts[0] == 'boarded' and parts[1] not in served_passengers:
                boarded_passengers.add(parts[1])

        # Count required board and depart actions
        num_waiting = len(waiting_passengers_info)
        num_boarded = len(boarded_passengers)

        # Collect floors that need servicing (origin of waiting, destin of boarded)
        required_floors = set()
        for floor in waiting_passengers_info.values():
            required_floors.add(floor)
        for passenger in boarded_passengers:
            # Add destination floor for boarded passengers
            if passenger in self.passenger_destin:
                 required_floors.add(self.passenger_destin[passenger])
            # else: Passenger destination not found? Should not happen in valid problems.

        # Calculate movement cost
        movement_cost = 0
        # Only calculate movement if there are floors to visit and floor levels are known
        if required_floors and self.floor_level:
            # Get levels for required floors, only including floors for which we have levels
            required_levels = {self.floor_level[f] for f in required_floors if f in self.floor_level}

            if required_levels: # Ensure there are valid levels among required floors
                min_r_level = min(required_levels)
                max_r_level = max(required_levels)

                # Get the level of the current lift floor
                current_level = self.floor_level.get(current_floor)

                if current_level is not None: # Ensure current_floor was found in floor_level map
                    # Estimate movement cost using TSP on a line distance
                    # Distance from current level to closest required level + span of required levels
                    dist_to_min = abs(current_level - min_r_level)
                    dist_to_max = abs(current_level - max_r_level)
                    range_span = max_r_level - min_r_level
                    movement_cost = min(dist_to_min, dist_to_max) + range_span
                # else: current_floor not in floor_level map? Should not happen if parsing is correct.
            # else: required_floors had names not in floor_level map? Should not happen in valid problems.


        # Total heuristic is the sum of required actions at floors and estimated movement cost
        total_heuristic = num_waiting + num_boarded + movement_cost

        # The heuristic is 0 if and only if all passengers are served.
        # If all passengers are served, num_waiting=0, num_boarded=0, required_floors={}, movement_cost=0.
        # The heuristic is non-negative and finite for solvable states.

        return total_heuristic
