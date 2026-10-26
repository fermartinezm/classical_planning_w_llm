from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class miconic18Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers by considering the necessary moves to their origins and destinations, boarding, and departing actions.

    # Assumptions
    - Each unserved passenger must be boarded (if not already) and departed at their destination.
    - The elevator can move directly between any two floors in one action.
    - Boarding and departing each require one action per passenger.
    - Moving to a floor requires one action per unique floor visited.

    # Heuristic Initialization
    - Extracts the origin and destination floors for each passenger from static facts.
    - Stores these in dictionaries for quick lookup during state evaluation.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Current Elevator Position**: Determine the current floor of the elevator from the state.
    2. **Unserved Passengers**: Identify passengers not yet served.
    3. **Unboarded Passengers**: Among unserved, identify those not yet boarded.
    4. **Origins and Destinations**:
       - Collect origins of unboarded passengers.
       - Collect destinations of all unserved passengers.
    5. **Move Actions**:
       - Count unique origins not at the current floor (moves to origins).
       - Count unique destinations not in origins (moves from origins to destinations).
    6. **Action Counts**:
       - Board each unboarded passenger (one action each).
       - Depart each unserved passenger (one action each).
    7. **Total Estimate**: Sum move actions, board actions, and depart actions.
    """

    def __init__(self, task):
        self.origin = {}
        self.destin = {}
        for fact in task.static:
            parts = fact.strip('()').split()
            if parts[0] == 'origin':
                passenger = parts[1]
                floor = parts[2]
                self.origin[passenger] = floor
            elif parts[0] == 'destin':
                passenger = parts[1]
                floor = parts[2]
                self.destin[passenger] = floor

    def __call__(self, node):
        state = node.state
        current_floor = None
        # Extract current lift position
        for fact in state:
            if fact.startswith('(lift-at'):
                current_floor = fact.strip('()').split()[1]
                break
        
        unserved = []
        unboarded = []
        for passenger in self.origin:
            # Check if passenger is served
            if f'(served {passenger})' not in state:
                unserved.append(passenger)
                # Check if passenger is boarded
                if f'(boarded {passenger})' not in state:
                    unboarded.append(passenger)
        
        # Calculate unique origin and destination floors
        unique_origins = set(self.origin[p] for p in unboarded)
        unique_destinations = set(self.destin[p] for p in unserved)
        
        # Compute move actions
        moves_origins = len([o for o in unique_origins if o != current_floor]) if current_floor else 0
        moves_destinations = len([d for d in unique_destinations if d not in unique_origins])
        
        # Board and depart actions
        boards = len(unboarded)
        departs = len(unserved)
        
        return moves_origins + moves_destinations + boards + departs
