from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(lift-at passenger10)".
    - `args`: The expected pattern with wildcards `*` allowed.
    - Returns `True` if all parts in fact match corresponding pattern args, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (miconic) domain.

    Summary
    -------
    This heuristic estimates the number of actions needed to achieve the goal condition,
    which is to have the relation (origin <passenger-2> <passenger>) hold as specified by the goal.
    Essentially, it tries to estimate how many "depart" and "up" actions remain until the lift
    reaches the desired passenger and the origin relation is updated.

    Assumptions
    -----------
    - The goal is always of the form (origin passenger-2 passenger).
    - The "lift-at" predicate indicates the current position of the lift, represented by a passenger.
    - Passengers can be "boarded" (inside the lift).
    - The "served" predicate represents adjacency relations (or floors served between passengers).
    - Movement is done via "depart" (disembarking a boarded passenger and boarding the one where the lift currently is)
      and "up" (moving the lift along a path, updating boarded passengers and origins accordingly).
    - We ignore exact floor numbers and treat the connectivity via served predicates 
      as a graph to estimate the minimal steps remaining.

    Heuristic Initialization
    ------------------------
    - Extract goal facts to identify the target origin relation (passenger-2 and passenger).
    - From static facts, build a graph mapping each passenger to passengers they can serve, along with floors.
      This is derived from the "served" predicates: served(?v3 ?v4 ?v5) means passenger ?v3 serves passenger ?v4.
    - This static graph represents possible lift moves "up" and transitions via served relations.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify the current lift locations ("lift-at" facts) and which passengers are boarded.
       Because multiple "lift-at" facts can exist (multiple lift locations?), treat all as potential current lift positions.
       Usually there should be exactly one lift position, but we consider multiple conservatively.

    2. Extract the goal origin relation (passenger-2, passenger) that must be established.

    3. Understand the current origin relation from the state (origin ?passenger-2 ?passenger?).

    4. If current origin already matches the goal origin, heuristic is 0 (goal reached).

    5. Otherwise, estimate a minimal number of actions as follows:

       - We want to move from current lift position(s) to the passenger specified as the goal origin's passenger.

       - Because "depart" requires that the lift is at a passenger and boarded passenger can depart to next passenger served,
         the heuristic counts:

         a) The minimal graph distance (number of edges) from any current lift position to the goal passenger along the served graph.

         b) One "depart" action is needed to disembark the boarded passenger and board the lift passenger.

         c) One "up" action is needed for each "jump" along the served adjacency between passengers.

       - Also, consider if the passenger to serve is already boarded or not:
         - If the correct passenger is boarded, less actions are needed to "depart".
         - Otherwise, boarding and moving steps are needed.

    6. Heuristic value is the sum of estimated "up" moves plus one "depart" move,
       considering the shortest path in the passenger-served graph from lift location(s) to goal passenger.

    7. If no path is found (disconnected), return a high heuristic value (finite), e.g., 1000.

    This heuristic is not admissible but should guide greedy search efficiently by estimating
    distance to the goal passenger through served relations and expected transitions.

    """

    def __init__(self, task):
        # Extract goal origin relation (should be unique in goals).
        # Format: (origin passenger-2 passenger)
        self.goal_origin = None
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == "origin" and len(parts) == 3:
                self.goal_origin = (parts[1], parts[2])
                break

        # Build adjacency graph from static served relations:
        # key: passenger, value: set of served passengers connected.
        self.served_graph = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "served" and len(parts) == 4:
                # served ?v3 ?v4 ?v5 means passenger ?v3 serves passenger ?v4 at floor ?v5
                p_from = parts[1]
                p_to = parts[2]
                if p_from not in self.served_graph:
                    self.served_graph[p_from] = set()
                self.served_graph[p_from].add(p_to)

        # As an optimization, add reverse edges if the served relation might be bidirectional
        # But domain does not guarantee that, so we keep direction as is.

    def __call__(self, node):
        state = node.state

        # If goal already reached, heuristic is 0.
        if self.goal_origin is None or (f"(origin {self.goal_origin[0]} {self.goal_origin[1]})" in state):
            return 0

        goal_passenger = self.goal_origin[1]

        # Extract current lift positions (should be passenger names after "lift-at")
        lift_positions = {get_parts(fact)[1] for fact in state if fact.startswith("(lift-at")}

        if not lift_positions:
            # No lift-at info in state => Set heuristic high (finite).
            return 1000

        # Extract boarded passengers
        boarded_passengers = {get_parts(fact)[1] for fact in state if fact.startswith("(boarded")}

        # Extract current origin passenger-2 for comparison (could be none or different)
        current_origin = None
        for fact in state:
            if fact.startswith("(origin "):
                parts = get_parts(fact)
                if len(parts) == 3:
                    current_origin = (parts[1], parts[2])
                    break

        # If current origin matches goal origin, heuristic is 0
        if current_origin == self.goal_origin:
            return 0

        # We want to estimate how many actions to get the lift to the passenger indicated by goal_origin,
        # and update the origin relation accordingly via action sequences.

        # Heuristic logic:
        # - Count minimal number of "up" actions to move from any lift position to the goal passenger in served graph.
        # - Each "up" corresponds to changing lift-at passenger along an edge in served graph.
        # - One "depart" action is needed at the end to finalize boarding/disembarking.
        # - If the goal passenger is already boarded, potentially fewer steps.

        # We perform BFS in served_graph from each lift position to goal passenger to find shortest path length.

        from collections import deque

        def bfs(start, goal, graph):
            if start == goal:
                return 0
            visited = set()
            queue = deque([(start, 0)])
            while queue:
                current, dist = queue.popleft()
                if current == goal:
                    return dist
                if current in visited:
                    continue
                visited.add(current)
                for neighbor in graph.get(current, ()):
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))
            return None  # no path found

        # Find shortest distance from any lift position to goal passenger
        distances = []
        for pos in lift_positions:
            dist = bfs(pos, goal_passenger, self.served_graph)
            if dist is not None:
                distances.append(dist)
        if not distances:
            # No path to goal passenger found
            return 1000

        min_dist = min(distances)

        # Heuristic cost estimate:
        # - Each "up" action corresponds to one move along served edges (min_dist steps).
        # - One final "depart" action to update origin and disembark.
        # - If goal passenger is already boarded, no need for depart, heuristic could be just min_dist.
        #   But benefit is small, conservative estimate adds 1 always to guide search.

        # Check if goal passenger is boarded
        goal_boarded = goal_passenger in boarded_passengers

        heuristic_value = min_dist
        if not goal_boarded:
            heuristic_value += 1  # for depart action needed to board/disembark correctly

        # Guarantee heuristic is at least 1 if not at goal (helps search)
        heuristic_value = max(1, heuristic_value)

        return heuristic_value
