from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(origin p1 f1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers
    in the elevator system. It considers:
    - Passengers that still need to be picked up
    - Passengers that are boarded but not yet served
    - The elevator's current position and required movements

    # Assumptions:
    - The elevator can only move between floors connected by 'above' relations
    - Each passenger has exactly one origin and one destination floor
    - The elevator can carry multiple passengers simultaneously
    - Boarding and departing each take one action
    - Moving between adjacent floors takes one action

    # Heuristic Initialization
    - Extract passenger origins and destinations from static facts
    - Build a graph of floor connections from 'above' relations
    - Store goal conditions (all passengers must be served)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each unserved passenger:
       a) If not boarded:
          - Add cost to move elevator to passenger's origin floor
          - Add boarding action
       b) If boarded:
          - Add cost to move elevator to passenger's destination floor
          - Add departing action
    2. Optimize movement by:
       - Considering passengers that can be picked up along the way
       - Grouping passengers with same destination
    3. The total heuristic is the sum of:
       - All boarding and departing actions
       - Minimal elevator movements to serve all passengers
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract passenger origins and destinations
        self.origins = {}
        self.destinations = {}
        
        # Build floor connectivity graph
        self.above_relations = set()
        
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "origin":
                self.origins[parts[1]] = parts[2]
            elif parts[0] == "destin":
                self.destinations[parts[1]] = parts[2]
            elif parts[0] == "above":
                self.above_relations.add((parts[1], parts[2]))

    def __call__(self, node):
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state
        
        # Check if goal is already reached
        if self.goals <= state:
            return 0
            
        # Track current elevator position
        current_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_floor = get_parts(fact)[1]
                break
                
        if current_floor is None:
            return float('inf')  # Invalid state
            
        # Count unserved passengers
        unserved_passengers = []
        boarded_passengers = set()
        
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "origin":
                passenger = parts[1]
                if f"(served {passenger})" not in state:
                    unserved_passengers.append((passenger, "origin"))
            elif parts[0] == "boarded":
                passenger = parts[1]
                if f"(served {passenger})" not in state:
                    boarded_passengers.add(passenger)
                    
        # Add boarded but unserved passengers
        for passenger in boarded_passengers:
            unserved_passengers.append((passenger, "destin"))
            
        # Calculate minimal movement cost
        total_cost = 0
        current_pos = current_floor
        
        # We'll process passengers in batches to optimize movement
        while unserved_passengers:
            # Find closest passenger to current position
            min_dist = float('inf')
            next_passenger = None
            next_action = None
            next_floor = None
            
            for passenger, action_type in unserved_passengers:
                if action_type == "origin":
                    floor = self.origins[passenger]
                else:  # "destin"
                    floor = self.destinations[passenger]
                    
                # Calculate distance from current_pos to floor
                dist = self._floor_distance(current_pos, floor)
                if dist < min_dist:
                    min_dist = dist
                    next_passenger = passenger
                    next_action = action_type
                    next_floor = floor
                    
            if next_passenger is None:
                break  # Shouldn't happen for valid states
                
            # Add movement cost
            total_cost += min_dist
            
            # Add boarding/departing cost
            total_cost += 1
            
            # Update current position
            current_pos = next_floor
            
            # Remove processed passenger
            unserved_passengers.remove((next_passenger, next_action))
            
            # Remove any other passengers we can serve at this floor
            new_unserved = []
            for passenger, action_type in unserved_passengers:
                target_floor = self.origins[passenger] if action_type == "origin" else self.destinations[passenger]
                if target_floor == current_pos:
                    total_cost += 1  # Board/depart action
                else:
                    new_unserved.append((passenger, action_type))
            unserved_passengers = new_unserved
            
        return total_cost
        
    def _floor_distance(self, floor1, floor2):
        """Calculate minimal number of moves between two floors."""
        if floor1 == floor2:
            return 0
            
        # We can use BFS to find shortest path in the floor graph
        visited = set()
        queue = [(floor1, 0)]
        
        while queue:
            current, dist = queue.pop(0)
            if current == floor2:
                return dist
                
            if current in visited:
                continue
            visited.add(current)
            
            # Check all adjacent floors
            for f1, f2 in self.above_relations:
                if f1 == current and f2 not in visited:
                    queue.append((f2, dist + 1))
                elif f2 == current and f1 not in visited:
                    queue.append((f1, dist + 1))
                    
        return float('inf')  # No path found (shouldn't happen in valid problems)
