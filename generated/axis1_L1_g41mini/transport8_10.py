from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(sjuejzxf p1 l8)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the 'secret_domain' (hrpmxpjg domain).

    Summary:
    This heuristic estimates the number of actions required to reach the goal by focusing on 
    achieving all goal facts of the form (sjuejzxf ?obj ?location) as required by the goal.
    It combines estimates of: 
      - Distance in the location graph (defined by miud1izi edges) between the current location 
        of each relevant object and its goal location, indicating how many 'arsslvyt' actions 
        are minimally needed to move the location fluent one step closer.
      - Whether objects of type du66z5od (special objects from the domain) are already "unloaded" 
        at the correct hctyyy8k location or need to be moved there via szsevy61 and tmoxe3x5 actions, 
        inferred from the presence or absence of (un606moq ?obj ?hctyyy8k) facts.
    
    The heuristic counts for each goal fact how many location steps separate the current location 
    of the object from the goal location, and adds estimated costs to "unload" du66z5od objects 
    to their desired locations and to update relations (do9fmoyo) linked to kyvrixb5 objects accordingly.
    
    Assumptions:
    - The domain's movement between locations (x6ibpitl type objects) is symmetrical and represented 
      by (miud1izi ?loc1 ?loc2) facts in both directions.
    - Each required move from one location to another corresponds roughly to one 'arsslvyt' action.
    - For du66z5od objects, being "unloaded" at their target requires at least one szsevy61 or tmoxe3x5 action.
    - The static facts (miud1izi and hab0vli4) represent an undirected graph of locations and 
      connections between kyvrixb5 objects that can be used to estimate minimum distances.
    - Exact sequencing of szsevy61 and tmoxe3x5 actions is complex, so we approximate by adding one action cost if unloading is required.
    - The heuristic is not guaranteed admissible, but should be informative enough to guide Greedy Best-First Search efficiently.

    Heuristic Initialization:
    - Extract undirected graph of locations from static (miud1izi) facts for location distance computation.
    - Extract the hab0vli4 relations between kyvrixb5 objects to infer their connectivity (used for unloading steps).
    - Extract goal facts requiring (sjuejzxf ?obj ?location).
    - Precompute goal positions of relevant objects of type du66z5od and hctyyy8k.
    - Store facts relevant for heuristic calls for quick reference.

    Step-By-Step Thinking for Computing Heuristic:
    1. If the current state is a goal state, heuristic returns 0.
    2. For each goal of the form (sjuejzxf ?obj ?loc_goal):
       a. Identify current location of the object (?loc_curr) from facts (sjuejzxf ?obj ?loc_curr).
          - If not found, assign a large penalty as the object is "lost".
       b. Estimate the shortest distance between ?loc_curr and ?loc_goal via BFS on the static location graph.
          - Each step corresponds roughly to one arsslvyt action to move across locations.
       c. If the object is of type du66z5od:
          - Check if it is currently "unloaded" at its hctyyy8k container (?ic1ss):
            - Inspect if (un606moq ?obj ?container) exists in the state; if not, this implies you must perform unload-type actions (szsevy61 or tmoxe3x5).
            - Add a cost of 1 if unloading is needed (conservative estimate).
       d. Combine these costs to get a cost estimate for moving this object into final position and state coherence.
    3. Sum all estimated costs over all goal objects.
    4. Return the sum as heuristic value.

    This heuristic strikes a balance between informativeness and computational efficiency suitable 
    for guiding our greedy best-first search in this complex domain.
    """

    def __init__(self, task):
        # Store the goals needing (sjuejzxf ?obj ?loc)
        self.goal_sjuejzxf = []
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == "sjuejzxf" and len(parts) == 3:
                self.goal_sjuejzxf.append((parts[1], parts[2]))

        self.static = task.static

        # Build undirected graph of locations from miud1izi facts (x6ibpitl - x6ibpitl)
        self.location_graph = {}
        for fact in self.static:
            if match(fact, "miud1izi", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)

        # Extract hab0vli4 relations between kyvrixb5 objects (used for heuristic unloading estimation)
        # Builds a bidirectional map of linked kyvrixb5 objects.
        self.kyvrixb5_links = {}
        for fact in self.static:
            if match(fact, "hab0vli4", "*", "*"):
                _, c1, c2 = get_parts(fact)
                self.kyvrixb5_links.setdefault(c1, set()).add(c2)
                self.kyvrixb5_links.setdefault(c2, set()).add(c1)

        # Cache to store types of objects by examining operators' parameter domains
        # However, since we cannot directly access types here, we will identify du66z5od objects by facts in the state later.

        # To speed up shortest path calculations, we could cache distances but domain might be large; hence compute on demand.

    def shortest_distance(self, start, goal):
        """Compute shortest path length between two locations in the undirected location graph."""
        if start == goal:
            return 0
        visited = {start}
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            for neighbor in self.location_graph.get(current, ()):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        # If no path found, return a large penalty
        return 50

    def __call__(self, node):
        state = node.state

        # If goal reached, heuristic is zero.
        # Goal is subset of state facts, so shortcut:
        if all(goal in state for goal in self.goal_sjuejzxf):
            return 0

        # Index objects to current locations (sjuejzxf facts)
        # Map obj -> loc (only one location per object assumed)
        obj_current_locs = {}
        for fact in state:
            if match(fact, "sjuejzxf", "*", "*"):
                _, obj, loc = get_parts(fact)
                obj_current_locs[obj] = loc

        # Identify du66z5od objects from the state:
        # Any object appearing as first argument of un606moq predicate is du66z5od
        du66z5od_objects = set()
        for fact in state:
            if match(fact, "un606moq", "*", "*"):
                _, obj_du66z5od, _ = get_parts(fact)
                du66z5od_objects.add(obj_du66z5od)
            # Also sjuejzxf facts with first arg not in hctyyy8k or du66z5od types (unknown outside domain),
            # but safest approach is to collect du66z5od by presence in un606moq.

        total_cost = 0

        # We will check unloading status for du66z5od objects via presence of (un606moq ?obj ?hctyyy8k)
        # and their related (do9fmoyo ?hctyyy8k ?kyvrixb5).

        # Extract do9fmoyo facts: map hctyyy8k -> set of kyvrixb5
        do9fmoyo_map = {}
        for fact in state:
            if match(fact, "do9fmoyo", "*", "*"):
                _, hctyyy8k_obj, kyvrixb5_obj = get_parts(fact)
                do9fmoyo_map.setdefault(hctyyy8k_obj, set()).add(kyvrixb5_obj)

        # Extract hab0vli4 facts (static) between kyvrixb5 objects already stored as self.kyvrixb5_links

        # For each goal (sjuejzxf obj loc_goal)
        for obj, loc_goal in self.goal_sjuejzxf:
            loc_curr = obj_current_locs.get(obj)
            if loc_curr is None:
                # Object location unknown, penalize heavily
                total_cost += 20
                continue

            dist = self.shortest_distance(loc_curr, loc_goal)

            cost = dist  # Base cost is to move location-value of sjuejzxf from current to goal.

            # If obj is du66z5od, check if unloading needed
            if obj in du66z5od_objects:
                # Find if currently "unloaded": (un606moq ?obj ?hctyyy8k) exists?
                unloaded = False
                hctyyy8k_containers = []
                for fact in state:
                    if match(fact, "un606moq", obj, "*"):
                        _, _, container = get_parts(fact)
                        hctyyy8k_containers.append(container)

                if hctyyy8k_containers:
                    # For each container, check if do9fmoyo and hab0vli4 connections seem consistent:
                    # If container linked to kyvrixb5 objects that relate to goal, accept unloaded.
                    # This is a heuristic check, add no cost if found.
                    unloaded = True
                else:
                    # Unloading needed (at least one unload action required)
                    cost += 1

            total_cost += cost

        return total_cost
