from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    Example: "(sjuejzxf p1 l8)" -> ["sjuejzxf", "p1", "l8"]
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(sjuejzxf p1 l8)".
    - `args`: The expected pattern with possible wildcards '*'.
    
    Returns True if the fact matches the pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the 'secret_domain' (hrpmxpjg).

    Summary
    -------
    This heuristic estimates the number of actions required to move all goal objects (du66z5od)
    to their required locations (x6ibpitl) as specified in the goal. It considers the current 
    locations of objects and the carriers (hctyyy8k) holding them or locations they reside at.

    It models:
    - The cost to move objects between positions on the lattice formed by 'miud1izi' adjacency facts.
    - The cost to transfer objects between carriers and locations, as described by operator effects.
    - The hierarchical "carrier-location" relations connected by 'hab0vli4' facts between kyvrixb5 (entities related to carriers).
    
    Assumptions
    -----------
    - Objects to move are of type du66z5od (e.g., p1, p2, ...).
    - Carriers (hctyyy8k) and locations (x6ibpitl) form a spatial network connected by 'miud1izi'.
    - The goal requires some (sjuejzxf obj loc) facts for du66z5od objects at particular locations.
    - The carrier-item relationships (do9fmoyo, un606moq) represent whether an item is "carried".
    - Transfers of objects to their goal locations must be done either by moving them locally on paths or moving them carried and transferring carriers.
    - The static facts define the adjacency of locations and hierarchical relations among kyvrixb5.
    - Movement between adjacent locations costs 1 action.
    - Transfers between carriers and locations cost 1 action each.
    - Goals involving (sjuejzxf obj loc) must be achieved exactly.
    - Heuristic is not required to be admissible but should be efficient and informative.

    Heuristic Initialization
    -------------------------
    - Extract all adjacency relations (miud1izi) between locations (x6ibpitl) and store them as an undirected graph.
    - Extract hierarchical relationships (hab0vli4) between kyvrixb5 objects, to approximate carrier transfers if needed.
    - Parse the goals for all du66z5od objects and their target locations.
    - Build efficient lookup structures for objects' current locations, which may be:
        * directly at locations via (sjuejzxf obj loc)
        * carried by carriers via (un606moq obj carrier) and (do9fmoyo carrier kyvrixb5)
    - Precompute shortest path distances between locations for fast estimation of local moves.

    Step-By-Step Thinking for Computing Heuristic
    ----------------------------------------------
    Given a state:
    
    1. Identify all goal objects and their goal locations from the goals.
    2. For each goal object:
       a. Determine its current holder/location:
          - If (sjuejzxf obj loc): the object's current place is 'loc' directly.
          - Else if (un606moq obj carrier) and (do9fmoyo carrier kyv): the item is carried by 'carrier' which is associated with a kyvrixb5 entity.
       b. Determine the target location (goal_loc) from the goal.
       
    3. If the object is currently at the goal location (in terms of sjuejzxf predicate), cost is zero for that object.
    
    4. If the object is carried by a carrier:
       - Estimate cost to move the carrier to the goal location adjacently.
       - Include cost for dropping off the object (modeled as 1 action).
       - Distance between carrier's location and the goal location is considered.
       
    5. If the object is at a location but not the goal location:
       - Compute shortest path distance between current location and goal location.
       - The cost is roughly the number of moves required along 'miud1izi' edges.
       - Estimate number of moves as the shortest path length.
       
    6. Distance between kyvrixb5 entities (in the chain of do9fmoyo and hab0vli4) is approximated using the hab0vli4 hierarchy as a graph.
       - This allows approximate cost for moving carriers between hierarchical nodes.
       
    7. Sum all estimated costs for all objects still not at their goal positions.
    
    8. The heuristic returns the total sum.
    
    This approximation:
    - Approximates paths over adjacent locations as linear costs.
    - Treats picking up/dropping off as fixed small costs (1 each).
    - Ignores complex synchronization but provides informative heuristic guidance that correlates well with actions needed.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract all adjacency (miud1izi) pairs between locations (x6ibpitl).
        # Build undirected graph for shortest path computation.
        self.adj_graph = {}
        for fact in self.static:
            if match(fact, "miud1izi", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.adj_graph.setdefault(loc1, set()).add(loc2)
                self.adj_graph.setdefault(loc2, set()).add(loc1)

        # Extract hierarchy of kyvrixb5 connected by hab0vli4 edges (treated as undirected).
        self.kyv_graph = {}
        for fact in self.static:
            if match(fact, "hab0vli4", "*", "*"):
                _, k1, k2 = get_parts(fact)
                self.kyv_graph.setdefault(k1, set()).add(k2)
                self.kyv_graph.setdefault(k2, set()).add(k1)

        # Extract all goal objects (du66z5od) and their required goal locations (x6ibpitl).
        self.goal_obj_loc = {}
        for fact in self.goals:
            if match(fact, "sjuejzxf", "*", "*"):
                _, obj, loc = get_parts(fact)
                self.goal_obj_loc[obj] = loc

        # Precompute shortest paths cache for location pairs on demand.
        self.location_distance_cache = {}

        # Precompute shortest paths cache for kyvrixb5 pairs on demand.
        self.kyv_distance_cache = {}

    def shortest_path_length(self, graph, start, goal):
        """Compute shortest path length between start and goal in an undirected graph."""
        if start == goal:
            return 0
        queue = [(start, 0)]
        visited = {start}
        while queue:
            current, dist = queue.pop(0)
            for neighbor in graph.get(current, []):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        # No path found, return a large number
        return 1000

    def distance_between_locations(self, loc1, loc2):
        """Get cached distance or compute shortest path length in location adjacency graph."""
        key = (loc1, loc2)
        if key in self.location_distance_cache:
            return self.location_distance_cache[key]
        dist = self.shortest_path_length(self.adj_graph, loc1, loc2)
        self.location_distance_cache[key] = dist
        self.location_distance_cache[(loc2, loc1)] = dist
        return dist

    def distance_between_kyv(self, k1, k2):
        """Get cached distance or compute shortest path length in kyv hierarchy graph."""
        key = (k1, k2)
        if key in self.kyv_distance_cache:
            return self.kyv_distance_cache[key]
        dist = self.shortest_path_length(self.kyv_graph, k1, k2)
        self.kyv_distance_cache[key] = dist
        self.kyv_distance_cache[(k2, k1)] = dist
        return dist

    def __call__(self, node):
        state = node.state

        # If goal state (all goals met), heuristic = 0
        if self.goals <= state:
            return 0

        # Helper to find current location or carrier of an object.
        def find_obj_current_holder(obj):
            """
            Returns:
            - ('location', loc) if object is at a location (sjuejzxf obj loc)
            - ('carrier', carrier, kyv) if object is carried by carrier which influences kyvrixb5 node
            - None if not found (should not happen in valid state)
            """
            # Check if object is at location:
            for fact in state:
                if match(fact, "sjuejzxf", obj, "*"):
                    _, _, loc = get_parts(fact)
                    return ("location", loc)

            # Check if object is carried by du66z5od carrier (un606moq obj carrier)
            for fact in state:
                if match(fact, "un606moq", obj, "*"):
                    _, _, carrier = get_parts(fact)
                    # find kyvrixb5 node carrier is associated with (do9fmoyo carrier kyv)
                    for fact2 in state:
                        if match(fact2, "do9fmoyo", carrier, "*"):
                            _, _, kyv = get_parts(fact2)
                            return ("carrier", carrier, kyv)
            return None

        # Helper to find location of a carrier (hctyyy8k) in the state.
        def find_carrier_location(carrier):
            """
            The carrier location is found by (sjuejzxf carrier loc) in the state.
            Returns the location string or None.
            """
            for fact in state:
                if match(fact, "sjuejzxf", carrier, "*"):
                    _, _, loc = get_parts(fact)
                    return loc
            return None

        # Constants for action costs (approximate)
        MOVE_COST = 1          # move between adjacent locations
        PICK_OR_DROP_COST = 1  # pick up or drop one object
        CARRIER_TRANSFER_COST = 1  # move carrier between kyvrixb5 nodes, approximate

        total_cost = 0

        # Compute cost per goal object
        for obj, goal_loc in self.goal_obj_loc.items():
            current_info = find_obj_current_holder(obj)
            if current_info is None:
                # If missing info, assume large cost
                total_cost += 10
                continue

            if current_info[0] == "location":
                current_loc = current_info[1]
                if current_loc == goal_loc:
                    # Already at goal location, no cost
                    continue
                # Cost: shortest path moving object alone + drop/pick actions:
                dist = self.distance_between_locations(current_loc, goal_loc)
                # At least dist moves + 2 actions (pick then drop)
                cost = dist * MOVE_COST + 2 * PICK_OR_DROP_COST
                total_cost += cost

            elif current_info[0] == "carrier":
                carrier, carrier_kyv = current_info[1], current_info[2]
                # The object is carried by a carrier located at some location.
                carrier_loc = find_carrier_location(carrier)
                if carrier_loc is None:
                    # If unknown carrier location, large penalty estimate
                    total_cost += 10
                    continue

                # Compute cost to move carrier from carrier_loc to goal_loc
                dist = self.distance_between_locations(carrier_loc, goal_loc)
                # Approximate cost for moving carrier in kyvrixb5 graph to goal kyvrixb5 node
                # Find kyv for goal_loc by reasoning: goal_loc is x6ibpitl, kyv is kyvrixb5, 
                # but no direct mapping, so use heuristic: minimum distance from carrier_kyv to every kyv
                # To approximate, assume carrier must transfer from carrier_kyv to kyv controlling goal's carrier
                # But goal does not mention kyv for goal_loc. We approximate distance from carrier_kyv to same kyv.
                # To keep complexity low, omit kyv distance and approximate with location distance only.

                # Cost components:
                # - Move carrier between locations (carrier_loc to goal_loc)
                # - Drop off object cost (1)
                # Total approximate cost:
                cost = dist * MOVE_COST + PICK_OR_DROP_COST
                total_cost += cost

        return total_cost
