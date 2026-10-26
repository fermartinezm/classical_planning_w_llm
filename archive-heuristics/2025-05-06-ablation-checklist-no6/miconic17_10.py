from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if pattern is longer than fact parts
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    This heuristic estimates the cost to reach the goal by summing the
    estimated costs for each unserved passenger independently. It calculates
    the cost for a passenger as the sum of:
    1. Movement cost for the lift to go from its current location to the
       passenger's origin floor (if not boarded).
    2. Cost of the 'board' action (if not boarded).
    3. Movement cost for the lift to go from the origin floor (or current
       lift floor if already boarded) to the passenger's destination floor.
    4. Cost of the 'depart' action.

    This heuristic is non-admissible as it ignores the fact that the lift
    can transport multiple passengers and visit multiple floors in a single trip,
    effectively double-counting movement costs.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations
        from the static facts and goals.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Parse floor order from static facts using the 'above' predicate
        # (above f_higher f_lower) means f_higher is directly above f_lower
        above_map = {} # floor_lower -> floor_higher
        all_floors = set()

        for fact in self.static_facts:
            if match(fact, "above", "*", "*"):
                f_higher, f_lower = get_parts(fact)[1:]
                above_map[f_lower] = f_higher
                all_floors.add(f_lower)
                all_floors.add(f_higher)

        # Find the bottom floor: a floor that is not the 'higher' part of any 'above' predicate
        # Or, a floor that is a 'lower' part but never a 'higher' part.
        higher_floors_in_above = set(above_map.values())
        lower_floors_in_above = set(above_map.keys())

        bottom_floor = None
        # A floor is the bottom if it's mentioned but no floor is directly below it (i.e., it's not a value in above_map)
        # This is incorrect. A floor is the bottom if no floor is directly *above* it (i.e., it's not a key in above_map).
        # Let's find a floor that is in all_floors but is not a key in above_map.
        bottom_candidates = all_floors - lower_floors_in_above

        if len(bottom_candidates) == 1:
             bottom_floor = list(bottom_candidates)[0]
        elif len(all_floors) == 1: # Case with only one floor
             bottom_floor = list(all_floors)[0]
        else:
             # Handle ambiguous or missing bottom floor (e.g., cycle, disconnected floors, or empty)
             # This shouldn't happen in valid miconic problems, but for robustness:
             # If there are floors, try finding the top floor instead (not a value in above_map)
             top_candidates = all_floors - higher_floors_in_above
             if len(top_candidates) == 1:
                 top_floor = list(top_candidates)[0]
                 # Build map downwards from top
                 self.floor_to_index = {}
                 self.index_to_floor = {}
                 current_floor = top_floor
                 index = len(all_floors) - 1 # Assign highest index to top floor
                 # Need below_map to traverse downwards
                 below_map = {v: k for k, v in above_map.items()} # floor_higher -> floor_lower
                 while current_floor is not None and index >= 0:
                     self.floor_to_index[current_floor] = index
                     self.index_to_floor[index] = current_floor
                     index -= 1
                     current_floor = below_map.get(current_floor) # Get the floor directly below
             else:
                 # Still ambiguous or empty. Fallback: sort alphabetically.
                 print("Warning: Could not determine floor order. Assuming alphabetical sort.")
                 if all_floors:
                     sorted_floors = sorted(list(all_floors))
                     self.floor_to_index = {f: i for i, f in enumerate(sorted_floors)}
                     self.index_to_floor = {i: f for i, f in enumerate(sorted_floors)}
                 else:
                     self.floor_to_index = {}
                     self.index_to_floor = {}


        # If bottom_floor was uniquely identified, build map upwards
        if bottom_floor is not None:
            self.floor_to_index = {}
            self.index_to_floor = {}
            current_floor = bottom_floor
            index = 0
            while current_floor is not None:
                self.floor_to_index[current_floor] = index
                self.index_to_floor[index] = current_floor
                index += 1
                current_floor = above_map.get(current_floor) # Get the floor directly above

        # Check if all floors were included (optional but good practice)
        if len(self.floor_to_index) != len(all_floors):
             print("Warning: Not all floors were included in the ordered list. Check 'above' predicates.")
             # Missing floors will cause KeyError later if encountered.


        # Parse passenger destinations from static facts
        self.destin_map = {} # passenger -> floor
        self.all_passengers = set()

        # Collect all passengers mentioned in goals (served) and static (destin)
        for goal in self.goals:
             if match(goal, "served", "*"):
                 p = get_parts(goal)[1]
                 self.all_passengers.add(p)

        for fact in self.static_facts:
            if match(fact, "destin", "*", "*"):
                p, f_dest = get_parts(fact)[1:]
                self.destin_map[p] = f_dest
                self.all_passengers.add(p) # Ensure all passengers are collected


    def __call__(self, node):
        """
        Compute a non-admissible estimate of the minimal number of required actions
        by summing independent costs for each unserved passenger.
        """
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        h = 0 # Initialize heuristic value

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        if current_lift_floor is None:
             # This shouldn't happen in a valid state, but handle defensively
             print("Error: Lift location not found in state.")
             return float('inf') # Cannot compute heuristic without lift location

        current_idx = self.floor_to_index.get(current_lift_floor)
        if current_idx is None:
             print(f"Error: Lift is at unknown floor {current_lift_floor}.")
             return float('inf') # Cannot compute heuristic if lift floor is not in our map


        unserved_passengers_info = [] # List of (p, state_type, floor)
        served_passengers = set()

        # Collect served passengers
        for fact in state:
             if match(fact, "served", "*"):
                 served_passengers.add(get_parts(fact)[1])

        # Collect unserved passengers and their state/location
        for p in self.all_passengers:
            if p not in served_passengers:
                is_origin = False
                is_boarded = False
                origin_floor = None
                # Check state facts for origin or boarded
                for fact in state:
                    if match(fact, "origin", p, "*"):
                        is_origin = True
                        origin_floor = get_parts(fact)[2]
                        break # Found origin, passenger cannot be boarded too
                    if match(fact, "boarded", p):
                        is_boarded = True
                        break # Found boarded

                if is_origin:
                    unserved_passengers_info.append((p, 'origin', origin_floor))
                elif is_boarded:
                    unserved_passengers_info.append((p, 'boarded', None)) # Floor is lift's current floor, but we need destin
                # Note: Passengers not in origin/boarded/served states are not explicitly handled.
                # Assuming valid miconic states always have a passenger in one of these states
                # if they are not served.

        # Sum costs for each unserved passenger independently
        for p, state_type, floor_info in unserved_passengers_info:
            # Ensure passenger has a destination
            if p not in self.destin_map:
                 print(f"Error: Unserved passenger {p} has no destination defined.")
                 return float('inf') # Cannot solve if destination is unknown

            f_dest = self.destin_map[p]
            dest_idx = self.floor_to_index.get(f_dest)
            if dest_idx is None:
                 print(f"Error: Destination floor {f_dest} for passenger {p} is unknown.")
                 return float('inf') # Cannot compute heuristic if destination floor is not in our map

            if state_type == 'origin':
                f_orig = floor_info
                orig_idx = self.floor_to_index.get(f_orig)
                if orig_idx is None:
                     print(f"Error: Origin floor {f_orig} for passenger {p} is unknown.")
                     return float('inf') # Cannot compute heuristic if origin floor is not in our map

                # Cost for this passenger: move to origin, board, move to destin, depart
                h += abs(current_idx - orig_idx) # Movement cost to origin
                h += 1 # board action cost
                h += abs(orig_idx - dest_idx) # Movement cost from origin to destin
                h += 1 # depart action cost
            elif state_type == 'boarded':
                # Cost for this passenger: move to destin, depart
                h += abs(current_idx - dest_idx) # Movement cost to destin
                h += 1 # depart action cost

        return h
