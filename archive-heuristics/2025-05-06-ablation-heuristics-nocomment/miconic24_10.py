from heuristics.heuristic_base import Heuristic
import math # For abs and min

def get_parts(fact):
    """Helper function to split a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Estimates the remaining cost to serve all passengers by summing the number
    of required board and depart actions and an estimate of the lift movement
    cost.

    Summary:
    The heuristic estimates the remaining plan length by summing three components:
    1. The number of 'board' actions needed for passengers waiting at their origin.
    2. The number of 'depart' actions needed for all unserved passengers.
    3. An estimate of the lift movement cost to visit all floors where pickups
       or dropoffs are required.

    Assumptions:
    - All actions have a unit cost of 1.
    - The lift has unlimited capacity for passengers.
    - Floor levels are linearly ordered, defined by the 'above' predicate.
    - The heuristic is non-admissible; it aims to guide a greedy best-first search
      efficiently rather than guarantee optimal solutions.

    Heuristic Initialization:
    In the constructor, the heuristic pre-processes static information:
    - It identifies all floors and determines their numerical levels based on
      the 'above' predicate facts. A mapping from floor name to level is created.
      The floor that is immediately above no other floor is considered the highest
      and assigned the maximum level. The floor immediately below no other floor
      is considered the lowest and assigned level 1.
    - It stores the destination floor for each passenger from the 'destin' facts.
    - It identifies all passengers involved in the problem by looking at 'origin'
      and 'destin' facts in the initial state and static facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the current floor of the lift by finding the '(lift-at ?f)' fact
       in the current state. Get its corresponding numerical level using the
       pre-calculated level map.
    2. Determine the set of passengers who have not yet been served by checking
       for the absence of '(served ?p)' facts for each known passenger. If this
       set is empty, the goal is reached, and the heuristic value is 0.
    3. Count the number of unserved passengers who are currently waiting at their
       origin floor (i.e., not boarded). Each of these requires a 'board' action
       at their origin floor. This count contributes directly to the heuristic value.
    4. Count the total number of unserved passengers. Each of these requires a
       'depart' action eventually at their destination floor. This count contributes
       directly to the heuristic value.
    5. Identify the set of floors where pickups are currently required: these are
       the origin floors of unserved passengers who are not yet boarded. Get their
       corresponding numerical levels using the level map.
    6. Identify the set of floors where dropoffs are currently required: these are
       the destination floors of unserved passengers who are currently boarded. Get
       their corresponding numerical levels using the level map.
    7. Combine the levels from steps 5 and 6 into a single sorted list of unique
       required visit levels.
    8. Estimate the lift movement cost:
       - If there are no required visit levels (the list from step 7 is empty),
         the movement cost is 0. This happens when all unserved passengers are
         already at their destination floor inside the lift, waiting only for
         the 'depart' action.
       - Otherwise, let L_min be the minimum level and L_max be the maximum level
         among the required visit levels. The estimated movement cost is calculated
         as the minimum travel distance from the current lift level to either L_min
         or L_max, plus the total range of required levels (L_max - L_min). This
         estimates the minimum vertical travel needed to reach the range of relevant
         floors and traverse it to visit all required stops within that range.
    9. The final heuristic value is the sum of the counts from steps 3 and 4,
       and the movement cost from step 8.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to find all floors and passengers

        # 1. Collect all floors and passengers
        all_floors = set()
        all_passengers = set()
        above_map = {} # fX -> fY if fX is immediately above fY

        # Collect floors and passengers from static facts and initial state
        for fact in static_facts | initial_state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'above':
                if len(parts) == 3: # Ensure correct number of arguments
                    fX, fY = parts[1], parts[2]
                    all_floors.add(fX)
                    all_floors.add(fY)
                    above_map[fX] = fY
            elif predicate == 'origin' or predicate == 'destin':
                 if len(parts) == 3: # Ensure correct number of arguments
                    p, f = parts[1], parts[2]
                    all_passengers.add(p)
                    all_floors.add(f)
            elif predicate == 'lift-at':
                 if len(parts) == 2: # Ensure correct number of arguments
                    f = parts[1]
                    all_floors.add(f)
            # Add 'boarded' and 'served' passengers to the set of all passengers
            elif predicate == 'boarded' or predicate == 'served':
                 if len(parts) == 2: # Ensure correct number of arguments
                    p = parts[1]
                    all_passengers.add(p)


        self.all_passengers = frozenset(all_passengers)

        # 2. Build level_map
        self.level_map = {}
        if all_floors:
            # Find the highest floor: a floor that is not a VALUE in the above_map.
            floors_below = set(above_map.values())
            highest_floor = next((f for f in all_floors if f not in floors_below), None)

            if highest_floor:
                current_floor = highest_floor
                current_level = len(all_floors) # Assign highest level to highest floor

                # Traverse downwards to build the level map
                while current_floor in self.level_map or current_level == len(all_floors): # Start loop correctly
                    self.level_map[current_floor] = current_level

                    # Find the floor immediately below current_floor
                    below_floor = above_map.get(current_floor)

                    if below_floor and below_floor in all_floors:
                        current_floor = below_floor
                        current_level -= 1
                    else:
                        # Reached the lowest floor or a floor not part of the main sequence
                        break
            # Handle cases with 0 or 1 floor if necessary, though miconic usually has >1 floor
            elif len(all_floors) == 1:
                 self.level_map[list(all_floors)[0]] = 1


        # 3. Store destination map
        self.destin_map = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'destin' and len(parts) == 3:
                p, f = parts[1], parts[2]
                self.destin_map[p] = f

    def __call__(self, node):
        state = node.state

        # 1. Find current lift floor and level
        current_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at' and len(parts) == 2:
                current_floor = parts[1]
                break
        # If lift location is unknown (should not happen in valid states), return infinity
        if current_floor is None:
             return float('inf')

        current_level = self.level_map.get(current_floor, 0) # Default to 0 if floor not found (shouldn't happen)


        # 2. Find unserved passengers
        served_passengers = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'served' and len(get_parts(fact)) == 2}
        unserved_passengers = self.all_passengers - served_passengers

        if not unserved_passengers:
            return 0 # Goal state

        # 3. Count board actions needed and identify pickup levels
        n_board_needed = 0
        pickup_levels = set()

        # Find passengers waiting at origin
        passengers_at_origin = {}
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'origin' and len(parts) == 3:
                 p, f = parts[1], parts[2]
                 passengers_at_origin[p] = f

        for p in unserved_passengers:
             # Check if this unserved passenger is not boarded
             is_boarded = any(get_parts(fact_b)[0] == 'boarded' and get_parts(fact_b)[1] == p for fact_b in state if len(get_parts(fact_b)) == 2)
             if not is_boarded:
                 n_board_needed += 1
                 # Get origin floor from state
                 origin_floor = passengers_at_origin.get(p)
                 if origin_floor and origin_floor in self.level_map:
                     pickup_levels.add(self.level_map[origin_floor])


        # 4. Count depart actions needed and identify dropoff levels
        n_depart_needed = len(unserved_passengers)
        dropoff_levels = set()

        for p in unserved_passengers:
             # Check if this unserved passenger is boarded
             is_boarded = any(get_parts(fact_b)[0] == 'boarded' and get_parts(fact_b)[1] == p for fact_b in state if len(get_parts(fact_b)) == 2)
             if is_boarded:
                 # Get destination floor from pre-calculated map
                 destin_floor = self.destin_map.get(p)
                 if destin_floor and destin_floor in self.level_map:
                     dropoff_levels.add(self.level_map[destin_floor])

        # 5. Combine required visit levels
        all_required_levels = sorted(list(pickup_levels | dropoff_levels))

        # 6. Estimate movement cost
        movement_cost = 0
        if all_required_levels:
            L_min = all_required_levels[0]
            L_max = all_required_levels[-1]
            # Cost to reach the range + cost to traverse the range
            movement_cost = min(abs(current_level - L_min), abs(current_level - L_max)) + (L_max - L_min)

        # 7. Calculate total heuristic value
        heuristic_value = n_board_needed + n_depart_needed + movement_cost

        return heuristic_value
