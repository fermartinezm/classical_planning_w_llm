from heuristics.heuristic_base import Heuristic
from task import Task


class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    Estimates the cost to reach the goal by summing the number of unserved
    passengers and the minimum vertical distance the lift needs to travel
    to cover all floors relevant to unserved passengers (their origin floors
    if unboarded, their destination floors if boarded) and the current lift floor.

    Assumptions:
    - The 'above' predicate defines a directed acyclic graph (DAG) where an edge
      from f_low to f_high exists if (above f_high f_low) is true. This graph
      represents the relative vertical positions of floors.
    - The floors form a connected component in this graph (when treated as undirected).
    - The static facts include all (destin p f) facts.
    - The state representation uses strings like '(predicate arg1 ...)' for facts.

    Heuristic Initialization:
    - Parses static facts to build a map from floor names to integer levels.
      This is done by treating (above f_high f_low) as an edge f_low -> f_high
      in a graph. A Breadth-First Search (BFS) is performed starting from the
      source nodes (floors with no incoming edges, representing the lowest floors)
      to assign levels based on the shortest distance from a source node.
    - Parses static facts to build a map from passenger names to their destination floors.
    - Collects all passenger and floor names mentioned in the task.

    Step-By-Step Thinking for Computing Heuristic:
    1. Get the current state from the node.
    2. Find the current floor of the lift by looking for the (lift-at f) fact.
    3. Identify the set of unserved passengers by checking which passengers do not
       have the (served p) fact in the state.
    4. If there are no unserved passengers, the goal is reached, return 0.
    5. If there are unserved passengers, identify the set of "relevant floors".
       A floor is relevant if it is the origin floor of an unboarded unserved passenger
       (found via (origin p f) facts in the state) or the destination floor of a
       boarded unserved passenger (found via (boarded p) facts in the state and
       pre-calculated passenger destinations).
    6. If the set of relevant floors is empty, it means all unserved passengers
       must be boarded and currently at their destination floor. The remaining cost
       is simply the number of such passengers (each needs one 'depart' action).
       Return the number of unserved passengers.
    7. If the set of relevant floors is not empty, create a set including the current
       lift floor and all relevant floors. Calculate the minimum and maximum
       floor levels among the floors in this combined set, using the pre-calculated
       floor level map. Floors for which a level was not assigned during initialization
       (e.g., due to disconnected graph) are ignored for the level calculation,
       falling back to a simpler heuristic if no levels are available.
    8. The estimated movement cost is the difference between the maximum and minimum
       levels found in step 7. This represents the minimum vertical span the lift
       must cover to potentially visit all necessary floors.
    9. The heuristic value is the sum of the number of unserved passengers and the
       estimated movement cost.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.goals = task.goals

        self.floor_level_map = {}  # Map floor_name -> level (int)
        self.passenger_destinations = {}  # Map passenger_name -> floor_name
        self.all_passengers = set()
        self.all_floors = set()

        above_facts = set()

        # Parse all facts (static and initial) to collect all objects and relevant predicates
        # task.facts contains all ground facts in the domain/problem, including types etc.
        # We need to parse task.static and task.initial_state specifically for predicate instances.
        all_facts_in_problem = set(task.static) | set(task.initial_state) # Use set for faster lookup if needed, though iteration is fine here

        for fact_str in all_facts_in_problem:
            parts = fact_str.strip('()').split()
            if not parts:
                continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'above':
                if len(args) == 2:
                    f_high, f_low = args
                    above_facts.add((f_high, f_low))
                    self.all_floors.add(f_high)
                    self.all_floors.add(f_low)
            elif predicate == 'destin':
                if len(args) == 2:
                    p, f_destin = args
                    self.passenger_destinations[p] = f_destin
                    self.all_passengers.add(p)
                    self.all_floors.add(f_destin) # Destinations are floors
            elif predicate == 'origin':
                 if len(args) == 2:
                     p, f_origin = args
                     self.all_passengers.add(p)
                     self.all_floors.add(f_origin) # Origins are floors
            elif predicate == 'lift-at':
                 if len(args) == 1:
                     self.all_floors.add(args[0])
            elif predicate in ['boarded', 'served']:
                 if len(args) == 1:
                     self.all_passengers.add(args[0])

        # Build floor level map using BFS on the 'above' graph
        # Graph nodes are floors, edge f_low -> f_high if (above f_high f_low)
        graph = {floor: [] for floor in self.all_floors}
        incoming_count = {floor: 0 for floor in self.all_floors}

        # Populate graph and incoming counts based on (above f_high f_low) facts
        for f_high, f_low in above_facts:
            # Ensure floors are in our collected set (should be if parsing all facts)
            if f_low in graph and f_high in graph:
                graph[f_low].append(f_high)
                incoming_count[f_high] += 1

        # Find source nodes (floors with no incoming edges - lowest floors)
        queue = [floor for floor, count in incoming_count.items() if count == 0]
        
        # If there are floors but no source nodes (e.g., cycle or empty above_facts for multiple floors),
        # assign level 0 to all floors as a fallback. This makes movement cost 0.
        if not queue and self.all_floors:
             for floor in self.all_floors:
                 self.floor_level_map[floor] = 0
        else:
            # BFS to assign levels (distance from a source node)
            # Level 0 for source nodes (lowest floors), Level 1 for floors immediately above, etc.
            level = 0
            visited = set()
            current_level_nodes = queue

            while current_level_nodes:
                next_level_nodes = []
                for floor in current_level_nodes:
                    if floor not in visited:
                        visited.add(floor)
                        self.floor_level_map[floor] = level
                        for neighbor in graph.get(floor, []):
                            incoming_count[neighbor] -= 1
                            if incoming_count[neighbor] == 0:
                                next_level_nodes.append(neighbor)
                level += 1
                current_level_nodes = next_level_nodes

            # Floors not reached by BFS (disconnected components) will not have a level.
            # This is handled in __call__ by checking if floor is in self.floor_level_map.


    def __call__(self, node):
        state = node.state  # frozenset of fact strings

        # Find current lift floor
        current_lift_floor = None
        # Convert state to set for faster lookups if needed, but iteration is fine for typical state sizes
        state_set = set(state)
        
        for fact_str in state_set:
            parts = fact_str.strip('()').split()
            if not parts: continue
            if parts[0] == 'lift-at' and len(parts) == 2:
                current_lift_floor = parts[1]
                break

        # Identify unserved passengers
        served_passengers = {p for fact_str in state_set for parts in [fact_str.strip('()').split()] if parts and parts[0] == 'served' and len(parts) == 2 for p in [parts[1]]}
        unserved_passengers = self.all_passengers - served_passengers

        # If goal reached, h = 0
        if not unserved_passengers:
            return 0

        # Identify relevant floors
        relevant_floors = set()
        boarded_passengers = {p for fact_str in state_set for parts in [fact_str.strip('()').split()] if parts and parts[0] == 'boarded' and len(parts) == 2 for p in [parts[1]]}
        origin_facts = {parts[1]: parts[2] for fact_str in state_set for parts in [fact_str.strip('()').split()] if parts and parts[0] == 'origin' and len(parts) == 3}

        for p in unserved_passengers:
            if p in boarded_passengers:
                # Unserved and boarded -> needs to go to destination
                if p in self.passenger_destinations:
                    relevant_floors.add(self.passenger_destinations[p])
            elif p in origin_facts:
                # Unserved and not boarded -> needs to be picked up at origin
                relevant_floors.add(origin_facts[p])
            # else: Unserved, not boarded, not at origin? Malformed state? Ignore for heuristic.

        # If no relevant floors, all unserved passengers must be boarded at their destination
        # (or malformed state). Cost is just the number of departures needed.
        if not relevant_floors:
             # This case implies unserved passengers are boarded and at their destination.
             # The remaining cost is one 'depart' action per such passenger.
             return len(unserved_passengers)

        # Calculate movement cost
        all_relevant_floors_including_current = relevant_floors.copy()
        if current_lift_floor: # Add current floor if known
             all_relevant_floors_including_current.add(current_lift_floor)

        # Filter floors to only include those for which we have a level
        floors_with_levels = {f for f in all_relevant_floors_including_current if f in self.floor_level_map}

        # If no relevant floors have assigned levels (e.g., disconnected graph component),
        # fall back to just counting passengers.
        if not floors_with_levels:
             return len(unserved_passengers)

        min_level = float('inf')
        max_level = float('-inf')
        for floor in floors_with_levels:
            level = self.floor_level_map[floor]
            min_level = min(min_level, level)
            max_level = max(max_level, level)

        movement_cost = max_level - min_level

        # Total heuristic = number of unserved passengers + estimated movement cost
        h_value = len(unserved_passengers) + movement_cost

        return h_value

