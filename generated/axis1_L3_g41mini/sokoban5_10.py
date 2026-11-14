from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (miconic) domain.

    Summary
    -------
    This heuristic estimates the number of actions required to achieve the goal by focusing
    on reconstructing the "origin" relation specified in the goal, which identifies a passenger-2
    entity's target passenger. The goal is reached once the goal 'origin' fact holds with the specified
    passenger-2 and passenger.

    To achieve this, passengers are moved through a chain of "served" relations (which represent a floor-based 
    connection between passengers) and are moved "up" via the lift, changing the "lift-at" and "boarded" predicates 
    accordingly, until the designated passenger-2's origin passenger matches the goal. The heuristic considers:
    - How many "origin" relationships need to be changed (i.e., the difference between current and goal origin).
    - Whether the lift is currently at the correct passenger to enable either "depart" or "up" actions.
    - The minimum number of "up" transitions (serving floors) needed to achieve the goal origin.
    - If passengers are boarded or not, indicating in-progress operations.

    Assumptions
    -----------
    - The goal condition is always a single "origin" fact: (origin passenger-2 passenger).
    - Passengers move along a graph defined by "served" relations on floors, which are static.
    - "lift-at" facts indicate the current passenger(s) where the lift(s) is/are located.
    - "boarded" facts indicate passengers currently inside the lift.
    - Actions "depart" and "up" move passengers along the chain defined by served relations and lift-at.
    - The heuristic ignores precise floor movement cost and treats all moves as unit cost.
    - This heuristic is not admissible but efficient and effective in guiding greedy best-first search.

    Heuristic Initialization
    ------------------------
    In the constructor:
    - Extract the single goal origin fact: identify the target passenger-2 and goal passenger.
    - Parse all static "served" facts and build a graph:
       Mapping served edges between passengers on floors, allowing navigation to estimate distances.
    - Store these relations in a dictionary keyed by passenger pairs to floors.
    
    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. Check if the goal origin fact holds in the current state. If yes, return 0 (goal reached).
    2. Extract the current origin fact for the concerned passenger-2 (if it exists).
       If no origin fact for that passenger-2, assume infinite distance (large heuristic).
    3. Build a graph of passengers connected by the served relation from static facts
       to estimate shortest passenger path distance from current origin passenger to goal origin passenger.
    4. Calculate the shortest distance (#served edges) between the current origin passenger and the goal origin passenger.
       If no path exists, return a large heuristic value.
    5. Determine if the lift is currently at the passenger representing the current "origin" passenger (where the lift is).
       If the lift is not at the correct passenger or if passenger is not boarded and the operation requires boarding,
       add +1 to simulate the effort of moving or boarding.
    6. Add additional costs for discrepancies such as:
       - If the passenger-2 has no boarded passengers but boarding is required.
       - If multiple lifts exist (multiple "lift-at" facts), no special weighting; heuristics scale naturally by paths.
    7. Return the sum of these costs as heuristic estimate.

    This heuristic produces finite values in solvable states and helps direct the search efficiently
    through counting the "travel" in the served graph and passenger boarding states, minimizing expanded nodes.
    """

    def __init__(self, task):
        self.goals = task.goals  # goal facts (expecting exactly one origin fact as goal)
        self.static = task.static  # static facts containing served relations

        # Extract the goal origin fact values passenger-2 and passenger:
        # Goal is of the form (origin passenger-2 passenger)
        goal_origin_facts = [fact for fact in self.goals if fact.startswith("(origin ")]
        assert len(goal_origin_facts) == 1, "Expected exactly one origin fact in goal"
        parts = get_parts(goal_origin_facts[0])
        _, self.goal_passenger2, self.goal_passenger = parts

        # Build a graph from the served facts: passenger -> list of connected passengers (neighbors)
        # served facts have form (served passenger passenger floor)
        # We model bidirectional edges ignoring floor (just count hops)
        self.served_graph = {}
        for fact in self.static:
            if fact.startswith("(served "):
                p1, p2, _floor = get_parts(fact)[1:]  # skip predicate
                self.served_graph.setdefault(p1, set()).add(p2)
                self.served_graph.setdefault(p2, set()).add(p1)

    def __call__(self, node):
        state = node.state

        # 1. Check if goal origin fact holds:
        goal_fact = f"(origin {self.goal_passenger2} {self.goal_passenger})"
        if goal_fact in state:
            return 0

        # 2. Find current origin fact for passenger-2 (should be unique)
        current_origin_fact = None
        prefix = f"(origin {self.goal_passenger2} "
        for fact in state:
            if fact.startswith(prefix):
                current_origin_fact = fact
                break

        # If no origin currently assigned for passenger-2, heuristic is large (goal far)
        if not current_origin_fact:
            return 1000  # large number as heuristic

        # Extract current origin passenger from fact
        current_origin_passenger = get_parts(current_origin_fact)[2]

        # 3. Estimate shortest path distance in served_graph from current origin passenger to goal passenger
        def bfs_distance(graph, start, goal):
            if start == goal:
                return 0
            visited = {start}
            queue = [(start, 0)]
            while queue:
                node, dist = queue.pop(0)
                for neighbor in graph.get(node, []):
                    if neighbor == goal:
                        return dist + 1
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            return None

        dist = bfs_distance(self.served_graph, current_origin_passenger, self.goal_passenger)
        if dist is None:
            # No path between current origin passenger and goal passenger
            return 1000

        # 4. Check passengers boarded and lift-at facts.
        boarded_passengers = {get_parts(f)[1] for f in state if f.startswith("(boarded ")}
        lift_positions = {get_parts(f)[1] for f in state if f.startswith("(lift-at ")}

        # 5. If the lift is not at current origin passenger, add 1 (cost to move lift)
        lift_at_current_origin = current_origin_passenger in lift_positions
        cost_to_move_lift = 0 if lift_at_current_origin else 1

        # 6. If no passenger boarded but dist > 0, boarding operation needed, add cost 1
        needs_boarding = (dist > 0 and current_origin_passenger not in boarded_passengers)

        cost_boarding = 1 if needs_boarding else 0

        heuristic_value = dist + cost_to_move_lift + cost_boarding

        return heuristic_value
