from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    Example: "(boarded passenger39)" -> ["boarded", "passenger39"]
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(boarded passenger39)".
    - `args`: The expected pattern components (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-Dependent Heuristic for the secret_domain (miconic).

    Summary
    -------
    This heuristic estimates the number of required actions to reach a goal state
    in the "miconic" domain.
    The problem involves passengers moving via a network of served floors (nodes)
    with lifts at certain passengers and boarding/departing operations.

    Assumptions
    -----------
    - The goal is to establish a particular (origin ?passenger-2 ?passenger) fact, i.e., to achieve the specific
      "origin" relationship for a given passenger-2 and passenger.
    - Actions either "depart" a boarded passenger if conditions on lift-at and served predicates hold
      or "up" move a boarded passenger up the served chain and update lifted passengers.
    - Lift-at facts represent where the lift currently is.
    - Boarding or unboarding changes the boarded facts accordingly.
    - Multiple lifts can exist at the same time on different passengers.
    - Static "served" facts encode the connectivity (edges) on floors that the lifts move across.
    - We assume that reaching the goal implies the origin fact for the goal passenger-2 and passenger must hold.

    Heuristic Initialization
    ------------------------
    - Extract the goal "origin" fact to identify the target passenger-2 and passenger.
    - Parse static "served" facts into a graph structure connecting passengers and floors.
      The served facts have the form `(served ?passenger1 ?passenger2 ?floor)` indicating a directed connection.
    - From the static graph, build adjacency maps for forward movement (for "up" action).
    - Extract sets of floors and passengers for quick lookup.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state is a goal state (i.e., goal origin fact present), heuristic = 0.

    2. Identify the target passenger-2 and passenger from the goal (only one goal origin fact given).

    3. Determine the current "origin" fact for that passenger-2 (who is the current origin passenger for that passenger-2).
       If it equals the goal passenger, then we have reached the goal.

    4. If not at goal origin, estimate how many "up" actions are required to move from current origin passenger towards the goal passenger:
       - We find a shortest path from current origin to goal passenger along the "served" graph.
         The path length is the number of "up" actions needed.

    5. Additionally, to depart the goal passenger from the lift, the passenger must be boarded and the lift at the correct passenger.
       Estimate whether the passenger is boarded. If not boarded, boarding is an extra action (which requires the lift and path to be there).
       Since boarding is done after lifts move and the passenger is served, we count roughly:
         - One depart action when conditions are met.
         - Possibly one boarding action if not boarded yet.
         - Lifts must be at the right place for boarding or departing; thus, movements via "up" are accounted.

    6. Consider lift positions: 
       - The heuristic will factor in how far the lift is from the current origin passenger.
       - Since multiple lifts exist, estimate the minimum distance from a lift-at passenger to the current origin passenger
         via reverse edges of served graph (to simulate how lifts can move).

    7. Sum up:
       - Lift moves to current origin passenger (if needed).
       - Boarding the current origin passenger (if not already boarded).
       - Number of "up" moves from current origin passenger to goal passenger along served graph.
       - Final depart action to achieve goal origin fact.
    
    8. The heuristic is a sum of these action counts, giving a rough but effective estimate.

    This heuristic is efficient and domain-dependent, exploiting the static connectivity structure and lift positions,
    and passenger boarded/origin state. It may not be admissible but guides greedy best-first search effectively.
    """

    def __init__(self, task):
        # Extract the goal origin fact (there should be exactly one)
        origin_goals = [fact for fact in task.goals if match(fact, "origin", "*", "*")]
        assert len(origin_goals) == 1, "This heuristic expects exactly one goal origin fact."
        goal_origin_fact = origin_goals[0]
        _, self.goal_p2, self.goal_p = get_parts(goal_origin_fact)

        # Build graph from static 'served' facts: served passenger1 -> passenger2 edges.
        # Format: (served ?p1 ?p2 ?floor)
        self.served_graph = {}  # adjacency list: p1 -> set of p2
        # Reverse graph for lift movement backward distance estimation lift_at -> origin passenger
        self.reverse_served_graph = {}

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "served" and len(parts) == 4:
                p1, p2 = parts[1], parts[2]
                self.served_graph.setdefault(p1, set()).add(p2)
                self.reverse_served_graph.setdefault(p2, set()).add(p1)

        # Extract set of floors (not used directly here but could help if needed)
        self.floors = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "served" and len(parts) == 4:
                self.floors.add(parts[3])

    def shortest_path_length(self, graph, start, goal):
        """Compute shortest path length from start to goal in the given adjacency graph.

        Returns the length if reachable, else returns None."""
        if start == goal:
            return 0
        visited = {start}
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            neighbors = graph.get(current, [])
            for n in neighbors:
                if n == goal:
                    return dist + 1
                if n not in visited:
                    visited.add(n)
                    queue.append((n, dist + 1))
        return None

    def min_distance_to(self, graph, sources, target):
        """Find minimum distance from any node in sources to target using BFS on graph.

        If target unreachable, return None."""
        if target in sources:
            return 0
        visited = set(sources)
        queue = [(s, 0) for s in sources]
        while queue:
            current, dist = queue.pop(0)
            neighbors = graph.get(current, [])
            for n in neighbors:
                if n == target:
                    return dist + 1
                if n not in visited:
                    visited.add(n)
                    queue.append((n, dist + 1))
        return None

    def __call__(self, node):
        state = node.state

        # If goal origin fact present, heuristic is zero.
        goal_fact = f"(origin {self.goal_p2} {self.goal_p})"
        if goal_fact in state:
            return 0

        # Find current origin for the goal passenger-2.
        # If no origin fact known for this passenger-2, we consider high heuristic.
        current_origins = [
            fact for fact in state if fact.startswith(f"(origin {self.goal_p2} ")
        ]
        if not current_origins:
            # No origin fact for this passenger-2, heuristic large but finite.
            # Estimate from lifts to goal passenger.
            lifts = [get_parts(fact)[1] for fact in state if fact.startswith("(lift-at")]
            # Distance from any lift to goal passenger via reverse served graph (lift moving backwards).
            dist_lift_to_goal = self.min_distance_to(
                self.reverse_served_graph, lifts, self.goal_p
            )
            dist_lift_to_goal = dist_lift_to_goal if dist_lift_to_goal is not None else 10
            # Assume at least one board + one depart + moves
            return dist_lift_to_goal + 2 + 5

        # Extract current origin passenger for goal_p2
        current_origin_fact = current_origins[0]
        _, _, current_origin = get_parts(current_origin_fact)  # third element is current origin passenger

        # If current origin equals goal passenger, we want to depart goal passenger
        # so heuristic counts boarding if needed + lift moves if needed + one depart action.
        lifts = [get_parts(fact)[1] for fact in state if fact.startswith("(lift-at")]

        # Check if goal passenger is boarded
        boarded_goal = f"(boarded {self.goal_p})" in state
        boarded_current_origin = f"(boarded {current_origin})" in state

        # Distance from any lift to current_origin passenger (lift moves to origin)
        dist_lift_to_origin = self.min_distance_to(
            self.reverse_served_graph, lifts, current_origin
        )
        if dist_lift_to_origin is None:
            # If unreachable, assign finite high cost
            dist_lift_to_origin = 10

        # Number of "up" moves needed from current origin to goal passenger in forward served graph
        dist_origin_to_goal = self.shortest_path_length(
            self.served_graph, current_origin, self.goal_p
        )
        if dist_origin_to_goal is None:
            # If unreachable, high cost but finite
            dist_origin_to_goal = 10

        # To reach goal by actions:
        # Step 1: Move lift from current location(s) to current origin passenger if not already there (dist_lift_to_origin moves)
        # Step 2: If current origin passenger is not boarded, need one boarding action
        # Step 3: "Up" actions to move boarded passenger(s) from current origin to goal passenger (dist_origin_to_goal times)
        # Step 4: If goal passenger not boarded yet, may need an additional boarding (because "up" moves board passengers),
        #          but according to domain "up" results in boarding next passenger, so assume boarding is covered in "up" moves except first.
        # Step 5: One "depart" action to finalize goal origin fact.

        # For safety, count boarding on current origin if not boarded
        boarding_actions = 0
        if not boarded_current_origin:
            boarding_actions = 1  # board current origin passenger before "up" moves

        # After "up" moves, goal passenger should be boarded (domain semantics), so no extra boarding needed at goal.

        # One depart action always needed to reach goal after boarding and moving.
        depart_actions = 1

        heuristic_value = dist_lift_to_origin + boarding_actions + dist_origin_to_goal + depart_actions

        return heuristic_value
