from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by stripping parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern (supports wildcards '*').
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (miconic) domain.

    Summary
    -------
    This heuristic estimates the number of actions needed to reach the goal state,
    which, per the domain and goal definition, is achieved when a specific `origin`
    fact holds between a passenger-2 (floor or task origin) and a passenger (lift location).

    The domain models passengers moving through floors via a lift, with peculiar relationships:
    - `origin (?floor - passenger-2, ?passenger - passenger)` means the passenger currently 
      occupies the "origin" spot represented by a floor or prior passenger.
    - `served` facts describe adjacency or connectivity between passengers and floors.
    - `lift-at ?passenger` indicates the current position of the lift (on a passenger).
    - `boarded ?passenger` indicates if a passenger is inside the lift.

    The actions allow movement "up" and "depart" passengers, changing the `lift-at`, `boarded`,
    and `origin` facts accordingly. The goal is to change the origin such that a certain passenger-2
    is linked to a desired passenger.

    Assumptions
    -----------
    - The chain of served relations among passengers and floors represents valid lift stops or connections.
    - Exactly one `lift-at` passenger is true at any time.
    - The goal is a single `origin` fact to be established.
    - Actions `up` and `depart` move the lift and passengers along the chain, advancing towards the goal.
    - Boarding/departing costs are accounted for by counting how many passengers need to move or be boarded/departed.
    - We ignore possible cycles or complex graph structures since adjacency is given through served and we assume connectedness.

    Heuristic Initialization
    ------------------------
    - Extract the goal passenger-2 and passenger from the goal condition (`origin <passenger-2> <passenger>`).
    - Build adjacency mappings between passengers and floors based on the static `served` facts,
      which describe connectivity needed for lift moves.
    - Extract currently boarded passengers, lift position, and current origins from the state.
    - Precompute data structures mapping which passengers/floors serve which others to estimate distances.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify the target `origin` fact from the goal (the pair (goal_floor, goal_passenger)).

    2. Find the current position of the lift (`lift-at ?passenger`).

    3. Identify the passenger currently associated with the goal_floor via `origin` (where the floor is involved),
       and passengers currently boarded.

    4. The heuristic estimates how many moves are required to get the lift and passengers from the current
       lift position and boarding to achieve the goal `origin` fact.

    5. To estimate this:
       - Compute shortest "served" adjacency paths from current lift location to the target passenger where the lift should be.
       - Compute if boarding or departing actions are required to move the passengers properly.
       - The lift movement actions roughly correspond to actions needed to traverse the served edges.
       - Boarding and departing represent additional steps (each boarding or departing is one action).

    6. The heuristic sums:
       - The minimal number of `up` (lift moves) actions needed to reach the target passenger.
       - The number of passengers that need to be boarded or departed to achieve the goal `origin`.
       - Boarding and departing count as separate actions.
       - If the `origin` fact is already satisfied, heuristic is zero.

    7. Return this summed value as heuristic estimate.

    Note: This heuristic is not guaranteed admissible but aims to minimize node expansions by guiding the search towards aligning the lift position and passenger boarding matching the goal `origin`.
    """

    def __init__(self, task):
        # Extract goal: only one origin fact assumed as goal
        goal_origins = [g for g in task.goals if get_parts(g)[0] == "origin"]
        assert len(goal_origins) == 1, "Expected exactly one origin fact in goal"
        _, goal_floor, goal_passenger = get_parts(goal_origins[0])
        self.goal_floor = goal_floor
        self.goal_passenger = goal_passenger

        # Static served facts define adjacency (passenger, passenger, floor), we build adjacency graph of passengers and floors
        # Key insight: served facts link (passenger served passenger floor)
        self.adjacency = {}  # dict passenger -> set of passengers reachable directly

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "served":
                p_from, p_to, floor = parts[1], parts[2], parts[3]
                # Establish adjacency for movement - the lift moves among passengers
                # Add edge p_from -> p_to and p_to -> p_from to represent bidirectional moves (assuming served defines bidirectional connectivity)
                self.adjacency.setdefault(p_from, set()).add(p_to)
                self.adjacency.setdefault(p_to, set()).add(p_from)

        # Save all passengers for sanity checks
        self.all_passengers = set()
        for fact in task.facts:
            parts = get_parts(fact)
            if len(parts) > 1 and parts[0] in ("boarded", "lift-at", "origin", "served"):
                self.all_passengers.update(parts[1:])

    def __call__(self, node):
        state = node.state

        # Quick check: if goal origin fact is already true, heuristic = 0
        goal_origin_fact = f"(origin {self.goal_floor} {self.goal_passenger})"
        if goal_origin_fact in state:
            return 0

        # Find current lift position (lift-at ?passenger)
        lift_at_passengers = [p for fact in state for p in get_parts(fact)[1:] if fact.startswith("(lift-at")]
        # More direct:
        lift_pos = None
        for fact in state:
            if fact.startswith("(lift-at"):
                parts = get_parts(fact)
                lift_pos = parts[1]
                break
        assert lift_pos is not None, "Lift position unknown in state"

        # Identify which passengers are currently boarded
        boarded_passengers = set()
        for fact in state:
            if fact.startswith("(boarded"):
                parts = get_parts(fact)
                boarded_passengers.add(parts[1])

        # Identify current origin mapping (only one origin is assumed per passenger-2)
        current_origins = {}  # floor (passenger-2) -> passenger
        for fact in state:
            if fact.startswith("(origin"):
                parts = get_parts(fact)
                floor, passenger = parts[1], parts[2]
                current_origins[floor] = passenger

        # The heuristic is to estimate how many actions are needed to get:
        # (origin goal_floor goal_passenger)
        # which means to have goal_floor linked to goal_passenger.
        # This implies moving the lift to goal_passenger and setting origin accordingly.

        # Step 1: Estimate lift moves needed from current lift position to goal passenger
        dist = self._shortest_path(lift_pos, self.goal_passenger, self.adjacency)

        if dist is None:
            # No path found in adjacency graph (unlikely if domain is well-defined)
            # Return a large heuristic but finite
            dist = 1000

        # Step 2: Estimate if boarding or departing needed

        # Conditions for boarding/departing:
        # We know from the domain and code that:
        # - Only one passenger can be boarded at time (from actions: boarded indicates passenger inside lift)
        # - The origin links a passenger-2 (floor) to a passenger
        # We want to have the origin fact (goal_floor, goal_passenger).

        # Count the number of passengers that must be boarded or departed to transform current state origin to goal_origin
        # We know in goal only one origin fact should hold (assumed).

        # If different origin passenger currently for goal_floor, we need an "up" action to move that origin passenger out and replace it.
        current_origin_passenger = current_origins.get(self.goal_floor, None)

        # Count boarded passenger changes:
        # - If goal_passenger is not boarded but some other is boarded, we need at least one depart + one board action.
        # - If no passenger is boarded and goal_passenger is not boarded, one board needed.
        # - If goal passenger is already boarded, no boarding needed.
        boarding_actions = 0
        if self.goal_passenger in boarded_passengers:
            boarding_actions = 0  # already boarded, good
        else:
            if len(boarded_passengers) == 0:
                boarding_actions = 1  # board goal_passenger
            else:
                # Some other passenger boarded, need to depart first then board goal_passenger
                boarding_actions = 2

        # Step 3: Align origin with goal
        # If current origin passenger differs from goal passenger for goal_floor, 
        # assume at least one depart to remove old origin and one boarding of new origin (counted above).

        origin_change_actions = 0
        if current_origin_passenger != self.goal_passenger:
            # changing origin requires transition onboard/offboard
            # We consider that boarding_actions count covers this transition,
            # so no extra here to avoid double-counting
            origin_change_actions = 0
        else:
            # Origin passenger equals goal, only boarding/departing if any to fix lift pos
            origin_change_actions = 0

        # Step 4: Sum estimates
        # - Dist: number of lift move actions needed (each 'up' action)
        # - Boarding actions: boarding/departing needed for passengers (each action counts as 1)
        # We assume depart and board both need 1 action each

        heuristic_value = dist + boarding_actions + origin_change_actions

        return heuristic_value

    def _shortest_path(self, start, goal, adjacency):
        """
        Compute shortest path length between start and goal in adjacency graph using BFS.
        Returns None if no path exists.
        """
        if start == goal:
            return 0

        from collections import deque

        visited = set([start])
        queue = deque([(start, 0)])

        while queue:
            current, dist = queue.popleft()
            for neighbor in adjacency.get(current, []):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return None
