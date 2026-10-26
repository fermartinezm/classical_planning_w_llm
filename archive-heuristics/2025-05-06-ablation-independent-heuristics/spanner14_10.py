import collections
import math
from heuristics.heuristic_base import Heuristic
from task import Task

# Helper function to parse a PDDL fact string
def parse_fact(fact_str):
    """
    Parses a PDDL fact string into its predicate and objects.
    e.g., '(at bob shed)' -> ('at', ['bob', 'shed'])
    """
    # Removes surrounding brackets and splits by space
    parts = fact_str[1:-1].split()
    predicate = parts[0]
    objects = parts[1:]
    return predicate, objects

# Helper function to extract object name from "name - type" string
def parse_object_name(obj_str):
    """
    Extracts the object name from a string like "name - type".
    """
    return obj_str.split(" - ")[0]

# Helper function to extract object type from "name - type" string
def parse_object_type(obj_str):
    """
    Extracts the object type from a string like "name - type".
    """
    return obj_str.split(" - ")[1]

# BFS function to find shortest distances in an unweighted graph
def bfs(graph, start_node, all_nodes):
    """
    Performs Breadth-First Search to find shortest distances from a start node
    to all other nodes in an unweighted graph.

    Args:
        graph: An adjacency list representation of the graph {node: [neighbor1, ...]}
        start_node: The node to start the BFS from.
        all_nodes: A set of all possible nodes in the graph.

    Returns:
        A dictionary mapping each node to its shortest distance from the start_node.
        Returns math.inf for unreachable nodes.
    """
    distances = {node: math.inf for node in all_nodes}
    if start_node in all_nodes: # Ensure start_node is a valid node in the graph
        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node has neighbors in the graph
            if current_node in graph:
                for neighbor in graph[current_node]:
                    # Check if neighbor is a valid node and hasn't been visited yet (distance is inf)
                    if neighbor in all_nodes and distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
    return distances

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the spanner domain.

    Summary:
    Estimates the number of actions (walk, pickup, tighten) required to reach
    the goal state where all specified nuts are tightened. The heuristic
    components are the number of loose goal nuts (tighten actions), the number
    of spanners that need to be picked up (pickup actions), and an estimate
    of the walk actions required. The walk action estimate is the maximum
    shortest distance from the man's current location to any location where
    work is needed (either a loose goal nut location or a usable spanner
    location if spanners need picking up).

    Assumptions:
    - The domain is STRIPS with typing.
    - Links between locations are bidirectional (the PDDL defines them one-way,
      but typical interpretations and example instances suggest bidirectionality
      for movement). The heuristic models them as bidirectional.
    - Goal nuts are always at some location in the initial state and remain there.
    - Usable spanners needed for tightening are either carried or at some location.
    - Problem instances are solvable if there are enough usable spanners in total
      and required locations are reachable.

    Heuristic Initialization:
    In the constructor, the heuristic pre-processes static information:
    1. Identifies all objects and their types (man, spanner, nut, location)
       from task.objects.
    2. Extracts location objects to form the set of all possible locations.
    3. Builds an undirected graph representing the connections between locations
       based on the static '(link ?l1 ?l2)' facts. This graph is used for
       shortest path calculations (BFS). All identified locations are nodes
       in this graph.
    4. Identifies the set of goal nuts from the task definition's goals.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Parse the current state facts to quickly access information like object
       locations, carried items, and usable status. Find the man's current
       location. If the man's location is unknown or not a valid location,
       return infinity.
    2. Compute shortest path distances from the man's current location to all
       other known locations using BFS on the pre-built graph.
    3. Identify the set of goal nuts that are currently loose in the state
       (i.e., are goal nuts but the '(tightened ...)' fact is not in the state).
       Let N_loose be the count of these nuts. If N_loose is 0, the goal is
       reached, return 0.
    4. Identify usable spanners currently carried by the man. Let k_carrying
       be the count.
    5. Identify usable spanners currently at locations. Let usable_spanners_at_locs
       be the set of these spanners.
    6. Calculate the total number of usable spanners available in the state:
       all_usable_spanners_in_state = carried_usable_spanners | usable_spanners_at_locs.
       Let N_usable_total = len(all_usable_spanners_in_state). If N_loose > N_usable_total, the
       problem is likely unsolvable from this state (not enough spanners total),
       return infinity.
    7. Identify the set of locations where loose goal nuts are currently located
       (RequiredNutLocs). Iterate through loose_goal_nuts_in_state and use the
       parsed at_facts. Ensure the location is a valid node. If N_loose > 0 but
       no loose goal nuts are found at any known location in the state, return
       infinity (indicates an issue with the state representation or problem
       instance).
    8. Determine the set of locations where work is required (RequiredWorkLocs).
       This initially includes all locations in RequiredNutLocs.
    9. If the number of loose goal nuts (N_loose) is greater than the number
       of usable spanners the man is carrying (k_carrying), the man needs to
       pick up spanners. In this case, identify the locations of usable spanners
       currently at locations (RequiredSpannerLocs) using the parsed at_facts
       and usable_spanners_at_locs. Ensure locations are valid nodes. Add these
       locations to RequiredWorkLocs. If spanners are needed but no usable
       spanners are found at any known location, return infinity.
    10. If RequiredWorkLocs is empty (which should only happen if N_loose is 0,
        handled in step 3), return infinity as a safeguard.
    11. Calculate the maximum shortest distance from the man's current location
        to any location in RequiredWorkLocs. Iterate through RequiredWorkLocs,
        look up the distance computed in step 2. If any required location is
        unreachable (distance is infinity), return infinity. Otherwise, find
        the maximum distance.
    12. The heuristic value is calculated as:
        N_loose (estimated tighten actions)
        + max(0, N_loose - k_carrying) (estimated pickup actions)
        + maximum shortest distance to a required work location (estimated walk actions).
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goal_nuts = set()
        self.locations = set()
        self.spanners = set()
        self.nuts = set()
        self.man = None

        # Map object names to types
        obj_types = {}
        for obj_str in task.objects:
            name = parse_object_name(obj_str)
            obj_type = parse_object_type(obj_str)
            obj_types[name] = obj_type
            if obj_type == 'location':
                self.locations.add(name)
            elif obj_type == 'spanner':
                self.spanners.add(name)
            elif obj_type == 'nut':
                self.nuts.add(name)
            elif obj_type == 'man':
                self.man = name # Assuming only one man

        # Build location graph from static links
        self.location_graph = {loc: [] for loc in self.locations}
        for fact_str in task.static:
            predicate, objects = parse_fact(fact_str)
            if predicate == 'link' and len(objects) == 2:
                loc1, loc2 = objects
                # Ensure locations exist in our set of locations before adding link
                if loc1 in self.locations and loc2 in self.locations:
                    self.location_graph[loc1].append(loc2)
                    self.location_graph[loc2].append(loc1) # Assume bidirectional links

        # Identify goal nuts
        for goal_fact_str in task.goals:
            predicate, objects = parse_fact(goal_fact_str)
            if predicate == 'tightened' and len(objects) == 1:
                self.goal_nuts.add(objects[0])

    def __call__(self, node):
        state = node.state

        # Parse state facts for quick lookup
        state_facts = set(state)
        at_facts = {} # obj -> loc
        carrying_facts = {} # man -> set of spanners
        usable_facts = set() # usable spanners

        man_location = None

        for fact_str in state_facts:
            predicate, objects = parse_fact(fact_str)
            if predicate == 'at' and len(objects) == 2:
                obj, loc = objects
                at_facts[obj] = loc
                if obj == self.man:
                    man_location = loc
            elif predicate == 'carrying' and len(objects) == 2:
                m, s = objects
                if m == self.man:
                    if self.man not in carrying_facts:
                         carrying_facts[self.man] = set()
                    carrying_facts[self.man].add(s)
            elif predicate == 'usable' and len(objects) == 1:
                usable_facts.add(objects[0])

        # 1. Find the man's current location
        # Already done during state parsing

        if man_location is None or man_location not in self.locations:
             # Man's location is unknown or not a valid location node
             return math.inf

        # 2. Compute shortest path distances from man's location
        distances = bfs(self.location_graph, man_location, self.locations)

        # 3. Identify loose goal nuts in the current state
        loose_goal_nuts_in_state = {
            n for n in self.goal_nuts
            if f'(tightened {n})' not in state_facts
        }
        N_loose = len(loose_goal_nuts_in_state)

        if N_loose == 0:
            return 0 # Goal reached

        # 4. Identify usable spanners carried by the man
        carried_spanners_by_man = carrying_facts.get(self.man, set())
        carried_usable_spanners = {
            s for s in carried_spanners_by_man
            if s in usable_facts
        }
        k_carrying = len(carried_usable_spanners)

        # 5. Identify usable spanners at locations
        usable_spanners_at_locs = {
            s for s in self.spanners
            if s in at_facts and at_facts[s] in self.locations and s in usable_facts
        }

        # 6. Calculate total usable spanners and check solvability
        all_usable_spanners_in_state = carried_usable_spanners | usable_spanners_at_locs
        N_usable_total = len(all_usable_spanners_in_state)

        if N_loose > N_usable_total:
            # Not enough usable spanners in the entire state to tighten all loose goal nuts
            return math.inf

        # 7. Identify locations of loose goal nuts
        required_nut_locs = {at_facts[n] for n in loose_goal_nuts_in_state if n in at_facts and at_facts[n] in self.locations}

        # Check if loose goal nuts are not at any known location (shouldn't happen in valid instances)
        if N_loose > 0 and not required_nut_locs:
             return math.inf # Should not occur in valid problems

        # 8. Determine locations where work is required
        required_work_locs = set(required_nut_locs)

        # 9. If spanners need picking up, add spanner locations to required work locations
        if N_loose > k_carrying:
            required_spanner_locs = {at_facts[s] for s in usable_spanners_at_locs if s in at_facts and at_facts[s] in self.locations}

            # Check if spanners are needed but none are available at locations
            if N_loose > k_carrying and not required_spanner_locs:
                 return math.inf # Need spanners, but none are at locations

            required_work_locs.update(required_spanner_locs)

        # 10. Check if required work locations are identified (should be if N_loose > 0)
        if not required_work_locs:
             # This case should ideally be covered by N_loose == 0 check,
             # or the checks for empty required_nut_locs/required_spanner_locs
             # if N_loose > 0 and spanners are needed.
             # Adding as a safeguard.
             return math.inf # Should not occur in valid problems with N_loose > 0

        # 11. Calculate maximum shortest distance to a required work location
        max_dist_to_work_locs = 0
        for loc in required_work_locs:
            if distances[loc] == math.inf:
                # A required location is unreachable
                return math.inf
            max_dist_to_work_locs = max(max_dist_to_work_locs, distances[loc])

        # 12. Calculate heuristic value
        h_value = N_loose + max(0, N_loose - k_carrying) + max_dist_to_work_locs

        return h_value
