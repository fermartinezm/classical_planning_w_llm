from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (miconic) domain.

    Summary
    -------
    This heuristic estimates the number of actions required to achieve the goal of setting a new origin passenger 
    for a given passenger-2 object, in a context where passengers are boarded, lifted, and served along floors 
    via lift-at facts.

    Assumptions
    -----------
    - The goal is always of the form (origin passenger-2 passenger), i.e., setting a passenger as the origin for a given passenger-2.
    - The domain encodes an elevator-like service system where passengers can board, depart, move lifts (up), and passengers are "served" at floors.
    - Lifts correspond to passengers (strange typing from domain), and "lift-at" indicates the current passenger where the lift is at.
    - The main actions move "boards" of passengers through served relations along floors.
    - The central challenge is to reach the goal origin fact (origin passenger-2 passenger).
    - Changes of origin, boarding, and lift movements are necessary actions to reach the goal.
    - Heuristic does not need to be admissible but should estimate the number of needed actions efficiently.

    Heuristic Initialization
    ------------------------
    - Store the single goal origin fact: (origin p2 p).
    - Extract from static facts the "served" relations, building a graph (for each passenger, which passenger is served next and on what floor).
    - This can be used as a transport chain to move between passengers.
    - From the served facts, build a mapping that allows determination of lift movement paths.
    
    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify the goal origin fact: (origin P2_goal P_goal).
    2. Check if this goal fact is already true in the current state. Return 0 if so.
    3. Identify the current origin passenger for that passenger-2 (there is always exactly one origin fact per passenger-2).
       If no origin fact for the passenger-2 is present, consider this as "no origin set".
    4. Identify the lift positions (lift-at facts) in the current state.
    5. From "served" static facts, build a directed graph representing allowed served transitions between passengers.
    6. Model the problem as moving from the current origin passenger to the goal passenger along served edges, 
       simulating lift movement with a sequence of "up" and "depart" actions.
    7. The heuristic estimate is the minimal number of steps to achieve the goal:
       - Changing the origin from current_origin to goal_origin involves:
         * Moving the lift from current lift position(s) to the current origin passenger (if not already there) by moving along served edges.
         * Departing the current origin passenger.
         * Moving the lift along served edges to the goal_origin passenger.
         * Boarding the goal_origin passenger (implied by the action 'up' in the domain).
         * Final goal origin fact is achieved.
    8. Estimate the minimal path in the served graph from lift positions to current_origin, then from current_origin to goal_origin.
    9. Since multiple lifts can be at different passengers concurrently, the heuristic takes the minimal distance over all lift-at passengers.
    10. Each movement along served edges corresponds to an 'up' action; departing corresponds to a 'depart' action.
    11. Combine counts of movements and boarding/departing to estimate total steps:
        - Move lift to current_origin: cost = number of "served" edges traveled (each corresponds to an 'up' action)
        - Depart current_origin: cost = 1 'depart' action
        - Move lift to goal_origin: cost = number of "served" edges traveled
        - Boarding the goal_origin passenger is achieved by the "up" action arriving there (already counted in moves)
    12. Sum these actions for the heuristic estimate.
    """

    def __init__(self, task):
        """Extract goal origin fact and build served passenger graph from static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract the (origin P2 P) goal fact (expecting only one)
        origin_goals = [fact for fact in self.goals if fact.startswith("(origin ")]
        assert len(origin_goals) == 1, "Expect exactly one goal origin fact."
        _, p2_goal, p_goal = get_parts(origin_goals[0])
        self.p2_goal = p2_goal
        self.p_goal = p_goal

        # Extract all (served p1 p2 floor) facts -> build graph p1 -> p2
        # Create adjacency list for served passengers ignoring floor for shortest path calculation
        self.served_graph = {}  # {passenger: set_of_passengers_served_from_it}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "served":
                p_from = parts[1]
                p_to = parts[2]
                # Add the edge p_from -> p_to
                if p_from not in self.served_graph:
                    self.served_graph[p_from] = set()
                self.served_graph[p_from].add(p_to)

    def shortest_path_length(self, start_nodes, goal):
        """
        Compute the shortest path length (number of edges) from any node in start_nodes to goal
        in the served_graph using BFS. Returns None if no path exists.
        """
        from collections import deque

        visited = set()
        queue = deque()

        for start in start_nodes:
            if start == goal:
                return 0
            queue.append((start, 0))
            visited.add(start)

        while queue:
            node, dist = queue.popleft()
            neighbors = self.served_graph.get(node, set())
            for neigh in neighbors:
                if neigh == goal:
                    return dist + 1
                if neigh not in visited:
                    visited.add(neigh)
                    queue.append((neigh, dist + 1))
        return None  # No path found

    def __call__(self, node):
        """Estimate actions needed to achieve goal (origin p2 p)."""
        state = node.state

        # 1. Return 0 if goal fact is already true.
        goal_fact = f"(origin {self.p2_goal} {self.p_goal})"
        if goal_fact in state:
            return 0

        # 2. Find current origin fact for the passenger-2 p2_goal if any.
        current_origin = None
        origin_prefix = f"(origin {self.p2_goal} "
        for fact in state:
            if fact.startswith(origin_prefix):
                # Extract the current origin passenger.
                parts = get_parts(fact)
                if len(parts) == 3:
                    current_origin = parts[2]
                    break

        # If no origin currently set for p2_goal, model as if we start "off" the served graph,
        # so we consider departing unneeded, just need to board goal.
        # But from domain, origin references the linked passenger that is the origin passenger.

        # 3. Collect all passengers where a lift is currently at.
        lift_positions = set()
        lift_prefix = "(lift-at "
        for fact in state:
            if fact.startswith(lift_prefix) and fact.endswith(")"):
                parts = get_parts(fact)
                if len(parts) == 2:
                    lift_positions.add(parts[1])

        # Sanity: if no lift-at facts (should not happen), heuristic fallback large number.
        if not lift_positions:
            return 100000  # unreachable heuristic fallback

        # 4. If no current origin for p2_goal, assume we need to board the goal passenger at some lift position.
        # Find shortest path from any lift position to p_goal in served graph (lift movements).
        dist_lift_to_goal = self.shortest_path_length(lift_positions, self.p_goal)
        if dist_lift_to_goal is None:
            # No path to goal passenger => large number (hard to reach)
            return 100000

        # 5. If current_origin is None or same as goal passenger (goal origin already set to goal passenger):
        # If current_origin == p_goal but goal fact missing, minimal cost is zero? But goal fact is missing so means some action needed.
        # But the heuristic returns 0 only for goal fact. Otherwise try normal calculation:
        if current_origin is None:
            # No current origin for the passenger-2, so actions:
            # - Move lift to goal passenger: dist_lift_to_goal 'up' actions
            # - Boarding by 'up' action is done on last step (counted in dist)
            # So heuristic is dist_lift_to_goal (lift moves).
            # To be safe, we add 1 action for boarding implied by domain semantics (but 'up' already boards),
            # so no extra addition needed.
            return dist_lift_to_goal

        # 6. If current_origin is different from p_goal:
        # Actions roughly:
        # - Move lift from lift_positions to current_origin: cost = dist_lift_to_current_origin
        # - Depart current_origin: cost = 1
        # - Move lift from current_origin to p_goal: cost = dist_current_origin_to_p_goal
        # - Boarding at p_goal achieved by last move (up) counted already
        dist_lift_to_current = self.shortest_path_length(lift_positions, current_origin)
        if dist_lift_to_current is None:
            # Unreachable current origin, big penalty
            dist_lift_to_current = 100000

        dist_current_to_goal = self.shortest_path_length({current_origin}, self.p_goal)
        if dist_current_to_goal is None:
            # Unreachable path, big penalty
            dist_current_to_goal = 100000

        total_cost = 0
        # Move lift to current origin passenger if not already there
        # (if lift at current_origin, cost 0 for that move)
        total_cost += dist_lift_to_current
        # Depart action at current origin passenger
        total_cost += 1
        # Move lift from current origin passenger to goal passenger
        total_cost += dist_current_to_goal
        # Boarding goal passenger is done implicitly by last 'up' action, already counted in dist_current_to_goal
        return total_cost
