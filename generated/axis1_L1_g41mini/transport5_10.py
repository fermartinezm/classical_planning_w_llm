from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract components of a PDDL fact by removing surrounding parentheses and splitting.

    For example, "(sjuejzxf p1 l2)" -> ["sjuejzxf", "p1", "l2"]
    """
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (hrpmxpjg).

    Summary:
    --------
    This heuristic estimates the number of actions required to achieve the goal
    by roughly counting and analyzing the positional and container relationships 
    of objects in the state. Specifically, it focuses on the differences between
    the current "sjuejzxf" predicate extensions of the movable objects of type
    du66z5od, with respect to their goal locations.

    Assumptions:
    ------------
    - The main goal predicates are always (sjuejzxf <object_du66z5od> <location_x6ibpitl>)
      i.e., objects of type du66z5od located at specific locations.
    - Moving objects from one location to another requires sequences of:
        - Possibly moving locations (using miud1izi adjacency)
        - Changing the (sjuejzxf) relation gradually with arsslvyt actions
        - Using actions szsevy61 and tmoxe3x5 to put objects into or out of containers.
    - Container/capability relations (do9fmoyo and hab0vli4 predicates) affect possible moves.
    - Adjacency between locations (`miud1izi`) is symmetric; shortest path calculations
      can be used to estimate minimal move costs.
    - The heuristic does not need to be admissible but should provide informative cost estimates.

    Heuristic Initialization:
    -------------------------
    - Extract the goal's desired (sjuejzxf <du66z5od> <x6ibpitl>) predicates.
    - From static facts, build:
      * Location adjacency graph from (miud1izi loc1 loc2).
      * Container hierarchy graph from (hab0vli4 c1 c2) relations.
    - No dynamic predicate is stored, only static relations for efficient distance computation.

    Step-By-Step Thinking for Computing Heuristic:
    -----------------------------------------------
    1. Identify each goal object (du66z5od) and its target location (x6ibpitl) from the goal facts.
    
    2. For each such object:
       a) Find current location(s) from (sjuejzxf <obj> <loc>) predicates in the current state.
          If none exist, check if the object is in a container by presence of (un606moq) relations.
       
       b) Estimate cost to move this object to goal location:
          - If object is currently directly at some location (sjuejzxf), estimate shortest
            path distance on the location graph (miud1izi) between current and goal location.
            Each movement along adjacency likely costs at least one action (arsslvyt).
          - If object is inside some container/location, estimate cost to extract it (using
            szsevy61 and tmoxe3x5 actions), and then move it toward goal location.
            Include minimal expected number of container state changes (do9fmoyo, hab0vli4).
       
    3. Sum all individual objects' estimated costs to get a total heuristic estimate.
    
    4. For container/capability relations, incorporate a small cost per container transfer
       to capture the cost of changing container states (szsevy61 / tmoxe3x5) actions.
       
    5. If the state is a goal state, return heuristic 0.
    6. Always return a finite heuristic value, reflecting a rough but computationally
       lightweight sum of expected movements and container operations.

    This heuristic balances spatial and container complexity to guide search effectively.
    """

    def __init__(self, task):
        self.goals = task.goals

        # Build goal map: du66z5od object -> goal location (x6ibpitl)
        self.goal_locations = {}
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "sjuejzxf":
                obj = parts[1]  # du66z5od object
                loc = parts[2]  # x6ibpitl location
                self.goal_locations[obj] = loc

        # Build location adjacency graph from miud1izi (edges between locations)
        # bidirectional adjacency assumed
        self.location_graph = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "miud1izi":
                loc1, loc2 = parts[1], parts[2]
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)
        
        # Build container hierarchy graph from hab0vli4 predicate 
        # (map container c -> set of containers connected/above it)
        self.container_graph = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "hab0vli4":
                c1, c2 = parts[1], parts[2]
                self.container_graph.setdefault(c1, set()).add(c2)
                self.container_graph.setdefault(c2, set())  # ensure all containers keys included

    def shortest_path_length(self, graph, start, goal):
        """
        Compute shortest path length between start and goal in the adjacency graph.

        If no path exists, return a large number to reflect high cost.
        """
        if start == goal:
            return 0

        visited = {start}
        frontier = [(start, 0)]
        while frontier:
            current, dist = frontier.pop(0)
            neighbors = graph.get(current, set())
            for n in neighbors:
                if n == goal:
                    return dist + 1
                if n not in visited:
                    visited.add(n)
                    frontier.append((n, dist + 1))
        # no path found
        return 50  # arbitrary large cost

    def container_distance(self, cgraph, start, goal):
        """
        Estimate minimal container transition steps between containers start and goal.

        Containers here represent kyvrixb5 objects connected by hab0vli4.
        Uses BFS over container_graph. Return a large number if no path.
        """
        if start == goal:
            return 0

        visited = {start}
        frontier = [(start, 0)]
        while frontier:
            current, dist = frontier.pop(0)
            neighbors = cgraph.get(current, set())
            for n in neighbors:
                if n == goal:
                    return dist + 1
                if n not in visited:
                    visited.add(n)
                    frontier.append((n, dist + 1))
        return 20  # large cost if no path

    def __call__(self, node):
        state = node.state
        
        # Quick check: if goal reached, heuristic=0
        if self.goals <= state:
            return 0

        # Extract current sjuejzxf relations: object du66z5od to location x6ibpitl
        # and un606moq: object du66z5od inside container hctyyy8k
        sjuejzxf_pos = {}  # obj -> loc
        un606moq_pos = {}   # obj -> container
        do9f_pos = {}       # container -> kyvrixb5 (some higher container/state)
        for fact in state:
            p = get_parts(fact)
            if p[0] == "sjuejzxf":
                obj, loc = p[1], p[2]
                sjuejzxf_pos[obj] = loc
            elif p[0] == "un606moq":
                obj, cont = p[1], p[2]
                un606moq_pos[obj] = cont
            elif p[0] == "do9fmoyo":
                cont, kyv = p[1], p[2]
                do9f_pos[cont] = kyv

        total_cost = 0

        for obj, goal_loc in self.goal_locations.items():
            # Determine current position:
            if obj in sjuejzxf_pos:
                curr_loc = sjuejzxf_pos[obj]
                # Cost: shortest path between curr_loc and goal_loc on location graph
                dist_loc = self.shortest_path_length(self.location_graph, curr_loc, goal_loc)
                # Assume each edge movement is approx one arsslvyt action to change location reference
                cost = dist_loc

                # Also add cost for extracting from container if relevant:
                # If object also has a container (un606moq), estimate container transfer cost
                if obj in un606moq_pos:
                    container = un606moq_pos[obj]
                    # If container location differs, add cost to handle container changes:
                    # Goal container inferred from do9fmoyo predicate of goal location container(s).
                    # We approximate by container distance 0 since unknown target container: minimal increment
                    cost += 2  # approximate fixed cost for container manipulations
                total_cost += cost

            elif obj in un606moq_pos:
                # Object currently inside a container, but no direct location.
                container = un606moq_pos[obj]
                # Estimate cost to extract from container + move to goal location

                # First estimate container transfer cost from current container to any container with goal location
                # Since goal location container unknown, add a small fixed cost
                container_cost = 2

                # We try to get goal container(s) where sjuejzxf predicate is true for any object,
                # but without direct data, we rely on a fixed estimate.

                # Estimate moving from container to goal location:
                # Containers hold locations? Not directly stated, so estimate path length from one
                # of the container states in do9fmoyo to goal location roughly as 0-1 cost

                # Assume 1 step from container out to some location adjacent to goal_loc
                loc_move_cost = 1

                total_cost += container_cost + loc_move_cost

            else:
                # Object location unknown (not found in sjuejzxf or un606moq)
                # optimistic assumption: cost zero (or minimal 1)
                total_cost += 1

        # Additionally, penalize difference in container states for containers holding relevant objects,
        # measuring changes needed according to do9fmoyo and hab0vli4 predicates

        # We get all container-kyvrixb5 relations in current state
        current_container_positions = do9f_pos.copy()

        # We try to get target container positions from static hab0vli4 or from goal?
        # No explicit goal on containers, so we approximate cost per container as 0

        # Optionally add a small fixed overhead per container known in current_container_positions
        total_cost += len(current_container_positions)  # penalize some container repositioning

        return total_cost
