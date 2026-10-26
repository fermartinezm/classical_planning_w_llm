from heuristics.heuristic_base import Heuristic
from task import Task
import re

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
        Estimates the cost to reach the goal (all passengers served) by summing
        the estimated movement cost for the lift and the estimated number of
        board/depart actions needed. The movement cost is estimated as the
        distance from the current lift floor to the closer of the lowest/highest
        required floors, plus the distance between the lowest and highest
        required floors. Required floors are the origins of waiting passengers
        and destinations of boarded passengers. The action cost is the number
        of waiting passengers plus the number of boarded passengers.

    Assumptions:
        - The 'above' predicates define a linear ordering of floors.
        - Passenger destinations ('destin') are static.
        - The 'origin' predicate tracks the current waiting floor of a passenger.
        - Unserved passengers are either waiting at some floor (indicated by
          '(origin p f)' in the state) or are boarded ('(boarded p)' in the state).
          Any other state for an unserved passenger (e.g., unserved, not boarded,
          and no '(origin p f)' fact) is considered unreachable/invalid
          and results in an infinite heuristic value.
        - The goal is always to serve all passengers.

    Heuristic Initialization:
        - Parses the 'above' facts from task.facts to determine the linear order
          of floors and assign a level (integer) to each floor. Assumes a single
          linear tower structure. Includes a fallback to sorting floor names
          if the 'above' relations are ambiguous or incomplete.
        - Parses the static 'destin' facts from task.static to store the
          destination floor for each passenger.
        - Collects all passenger names mentioned in initial state, static facts,
          or goals to ensure all relevant passengers are considered.
        - Stores the set of goal facts.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is a goal state. If yes, return 0.
        2. Identify the current floor of the lift from the state. If not found
           or its level is unknown, return infinity (invalid state).
        3. Initialize an empty set `floors_to_visit` and an action counter `actions_needed`.
        4. Pre-parse relevant passenger status facts ('served', 'boarded', 'origin')
           from the current state for efficient lookup.
        5. Iterate through all known passengers:
            a. If the passenger is marked as served in the state, skip.
            b. If the passenger is unserved:
                i. Increment `actions_needed` by 1 (representing the need for either a board or a depart action).
                ii. Check if the passenger is marked as boarded in the state.
                    - If boarded: Look up their destination floor from the pre-calculated map. If destination is known and its level is known, add it to `floors_to_visit`. Otherwise, return infinity (problem setup error).
                    - If not boarded: Look up their current origin floor from the pre-parsed state facts. If found and its level is known, add this floor to `floors_to_visit`. If not found (unserved, not boarded, and not waiting at any listed origin), return infinity (invalid/unreachable state based on domain actions).
        6. Calculate the movement cost:
            a. If `floors_to_visit` is empty, the movement cost is 0.
            b. If `floors_to_visit` is not empty:
                i. Get the floor levels for all floors in `floors_to_visit`. These are guaranteed to be known due to checks in step 5.
                ii. Find the minimum (`min_level_visit`) and maximum (`max_level_visit`) levels among the floors to visit.
                iii. Get the level of the current lift floor (`current_level`).
                iv. The movement cost is calculated as:
                    `(max_level_visit - min_level_visit) + min(abs(current_level - min_level_visit), abs(current_level - max_level_visit))`.
                    This estimates the travel from the current floor to the closer
                    extreme (min or max required floor), plus the travel covering
                    the full span between the min and max required floors.
        7. The total heuristic value is the sum of the movement cost and `actions_needed`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goals = task.goals
        self.static_facts = task.static

        self.floor_levels = {}
        self.passenger_destination = {}
        self.all_passengers = set()

        above_relations = {}
        all_floors = set()

        # Collect all floor names and above relations from all facts
        # task.facts contains all possible ground atoms in the domain instance
        for fact_str in task.facts:
             pred, args = self._parse_fact(fact_str)
             if pred == 'above' and len(args) == 2:
                 f1, f2 = args
                 above_relations[f1] = f2 # f2 is directly above f1
                 all_floors.add(f1)
                 all_floors.add(f2)
             # Collect floor names from other predicates just in case 'above' doesn't list all
             elif pred in ['lift-at', 'origin', 'destin'] and len(args) > 0:
                 for arg in args:
                     # Simple check if it looks like a floor (starts with 'f')
                     if isinstance(arg, str) and arg.startswith('f'):
                         all_floors.add(arg)
             elif pred in ['boarded', 'served'] and len(args) == 1:
                 # Collect passenger names
                 self.all_passengers.add(args[0])
             elif pred == 'origin' and len(args) == 2:
                 # Collect passenger names from initial origins
                 self.all_passengers.add(args[0])
             elif pred == 'destin' and len(args) == 2:
                 # Collect passenger names from destinations
                 self.all_passengers.add(args[0])


        if not all_floors:
             # Should not happen in a valid miconic problem
             # Heuristic will return inf later if floor levels are empty
             pass

        # Find the lowest floor based on 'above' relations
        # A floor is lowest if it is a key in above_relations but not a value
        floors_that_are_above_others = set(above_relations.values())
        potential_lowest_floors = [f for f in above_relations.keys() if f not in floors_that_are_above_others]

        lowest_floor = None
        if len(potential_lowest_floors) == 1:
             lowest_floor = potential_lowest_floors[0]
        elif len(potential_lowest_floors) > 1:
             # Multiple potential lowest floors - implies disconnected towers or error
             # Or, the single lowest floor is not a key in above_relations if it's the only floor.
             # Check floors that are not values at all.
             floors_not_above_anything = all_floors - floors_that_are_above_others
             if len(floors_not_above_anything) == 1:
                  lowest_floor = list(floors_not_above_anything)[0]
             else:
                 # Still ambiguous or multiple towers. Fallback to sorting names.
                 pass # lowest_floor remains None

        if lowest_floor is None and all_floors:
             # Fallback: Assign levels based on sorting floor names (f1, f2, ...)
             try:
                 # Filter out any non-floor objects that might have been added to all_floors by mistake
                 floor_names_only = [f for f in all_floors if isinstance(f, str) and f.startswith('f')]
                 sorted_floors = sorted(floor_names_only, key=lambda f: int(f[1:]))
                 for i, floor in enumerate(sorted_floors):
                     self.floor_levels[floor] = i + 1
             except (ValueError, IndexError):
                 # Floor names are not in f1, f2, ... format or list is empty. Cannot sort reliably.
                 # Cannot determine levels. Heuristic will return inf.
                 self.floor_levels = {} # Clear any partial levels

        elif lowest_floor is not None:
            # Build levels from the lowest floor using the 'above' chain
            current_floor = lowest_floor
            level = 1
            visited_floors = set()
            while current_floor is not None and current_floor not in visited_floors:
                visited_floors.add(current_floor)
                self.floor_levels[current_floor] = level
                current_floor = above_relations.get(current_floor)
                level += 1

            # Check if all floors were assigned a level
            if len(self.floor_levels) != len(all_floors):
                 # Some floors are disconnected or not part of the main tower chain
                 # Clear levels and rely on fallback or return inf
                 self.floor_levels = {} # Cannot determine levels for all floors reliably


        # 2. Extract passenger destination from static facts
        for fact_str in self.static_facts:
            pred, args = self._parse_fact(fact_str)
            if pred == 'destin' and len(args) == 2:
                p, f = args
                self.passenger_destination[p] = f
                # self.all_passengers.add(p) # Already added during fact collection


    def _parse_fact(self, fact_str):
         # Remove leading/trailing brackets and split by space
        parts = fact_str.strip("()").split()
        return parts[0] if parts else None, parts[1:] if len(parts) > 1 else []


    def __call__(self, node):
        state = node.state

        # 1. Check if goal state
        if self.goals <= state:
            return 0

        # Cannot compute meaningful heuristic without floor levels
        if not self.floor_levels:
             return float('inf')

        # 2. Find current lift floor
        current_lift_floor = None
        for fact_str in state:
            if fact_str.startswith('(lift-at '):
                # Extract floor name from '(lift-at floor_name)'
                parts = fact_str.strip("()").split()
                if len(parts) == 2:
                    current_lift_floor = parts[1]
                break

        if current_lift_floor is None:
             # Lift location unknown - invalid state?
             return float('inf')

        current_level = self.floor_levels.get(current_lift_floor)
        if current_level is None:
             # Current lift floor not found in floor levels - problem setup error?
             return float('inf')

        # 3. Initialize
        floors_to_visit = set()
        actions_needed = 0

        # Pre-parse relevant passenger status facts ('served', 'boarded', 'origin')
        state_facts_dict = {}
        for fact_str in state:
             pred, args = self._parse_fact(fact_str)
             if pred in ['origin', 'destin'] and len(args) == 2:
                  state_facts_dict[(pred, args[0])] = args[1]
             elif pred in ['boarded', 'served'] and len(args) == 1:
                  state_facts_dict[(pred, args[0])] = True
             # lift-at is handled separately

        # 5. Iterate through unserved passengers
        for passenger in self.all_passengers:
            if state_facts_dict.get(('served', passenger)):
                continue # Passenger is served, no cost

            # Passenger is unserved
            actions_needed += 1 # Needs either board or depart action

            if state_facts_dict.get(('boarded', passenger)):
                # Passenger is boarded, needs to go to destination
                dest_floor = self.passenger_destination.get(passenger)
                if dest_floor and dest_floor in self.floor_levels: # Check if destination floor is valid/known
                    floors_to_visit.add(dest_floor)
                else:
                    # Destination unknown or destination floor level unknown - problem setup error?
                    return float('inf')
            else:
                # Passenger is not served and not boarded, must be waiting at origin
                # Find their current origin floor in the state
                current_origin_floor = state_facts_dict.get(('origin', passenger))

                if current_origin_floor and current_origin_floor in self.floor_levels: # Check if origin floor is valid/known
                    floors_to_visit.add(current_origin_floor)
                else:
                    # Unserved, not boarded, and current origin unknown or origin floor level unknown
                    # This implies an invalid/unreachable state based on domain actions.
                    return float('inf')


        # 6. Calculate movement cost
        movement_cost = 0
        if floors_to_visit:
            # All floors in floors_to_visit are guaranteed to have levels due to checks above
            levels_to_visit = [self.floor_levels[f] for f in floors_to_visit]

            min_level_visit = min(levels_to_visit)
            max_level_visit = max(levels_to_visit)

            # Movement cost estimate
            # Travel from current level to the span [min_level_visit, max_level_visit]
            # plus the span itself.
            # Distance to closer end of span + span length
            movement_cost = (max_level_visit - min_level_visit) + \
                            min(abs(current_level - min_level_visit), abs(current_level - max_level_visit))

        # 7. Total heuristic
        return movement_cost + actions_needed
