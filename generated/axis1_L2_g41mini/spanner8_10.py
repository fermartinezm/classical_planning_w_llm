from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove parentheses and split a fact string into its parts."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with wildcards.

    Args:
        fact (str): A fact string, e.g. "(qkdx0xe2 obj1 loc1)".
        *args: Pattern parts to match, e.g. "qkdx0xe2", "*", "loc1".

    Returns:
        bool: True if all parts match the pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain (du770mtw).

    Summary:
    This heuristic estimates the number of actions required to reach the goal state,
    which requires certain objects of type wqid8mog to have the predicate (ne6sl557 ?obj).
    The domain's only goal predicate is of the form (ne6sl557 ?ris62).

    Assumptions:
    - The goal is to make all wqid8mog objects in the goal have the predicate (ne6sl557).
    - Such objects start having (of6wn8m4) and must lose this and gain (ne6sl557).
    - The only action producing (ne6sl557) is f7y6nzq3, which requires the following preconditions:
        * (ndg963qz ?cz2xr), some du770mtw object with that predicate
        * (qkdx0xe2 ?n54xx ?ts6t4), a pt4b08k4 object linked to an mz6r1d30 object
        * (fqyrzl30 ?n54xx ?cz2xr), linking pt4b08k4 and du770mtw object
        * (of6wn8m4 ?ris62), the wqid8mog object to "process"
        * (qkdx0xe2 ?ris62 ?ts6t4), linking the wqid8mog object to an mz6r1d30 object
    - The action also deletes (of6wn8m4 ?ris62) and (ndg963qz ?cz2xr), a kind of resource.
    - Other two actions move predicates qkdx0xe2 between mz6r1d30 objects and relate pt4b08k4 and du770mtw.
      We interpret these as "transport" or "connect" steps needed to enable action f7y6nzq3.

    Heuristic Initialization:
    - Extract the set of wqid8mog objects needed to be processed for goal (those for which (ne6sl557 ?ris62) must hold).
    - Extract static "lpmls56r" relations, representing adjacency between mz6r1d30 objects (locations).
    - Extract all du770mtw objects that can be "ndg963qz", i.e. can act as the resource for f7y6nzq3.
    - Build adjacency graph on mz6r1d30 objects via lpmls56r to allow shortest-path calculations.
    - This heuristic ignores exact assignments of pt4b08k4 objects and instead estimates costs based on distances and available resources.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify which wqid8mog objects in the goal do not yet have (ne6sl557).
       These are the ones still to be "processed".
    2. For each such wqid8mog object:
       - Find its current mz6r1d30 location from (qkdx0xe2 ?ris62 ?ts6t4).
       - Check if it currently has (of6wn8m4), if not, it cannot be processed yet by f7y6nzq3 action.
    3. Identify available du770mtw "resources" (those having (ndg963qz)) for performing f7y6nzq3.
    4. Also extract current locations of du770mtw resources from (qkdx0xe2 ?cz2xr ?ts6t4).
    5. Extract locations of pt4b08k4 objects with (fqyrzl30 ?n54xx ?cz2xr), linking pt4b08k4 to du770mtw.
    6. The core effect of f7y6nzq3 requires syncing three locations:
       - A du770mtw resource at some location ?cz2xr with (ndg963qz)
       - wqid8mog ?ris62 at an mz6r1d30 ?ts6t4
       - A pt4b08k4 ?n54xx that must be at ?cz2xr and linked to ?ts6t4.
    7. The qkdx0xe2 predicate can be moved on adjacency graph (mz6r1d30 nodes) by nm6lzp2x and jql8txb3 actions.
       This can be interpreted as "transport" steps or state changes for these objects.
    8. We approximate the heuristic as follows:
       - For each wqid8mog to process:
         * Estimate minimal cost to bring a du770mtw resource (ndg963qz) and a pt4b08k4 object to adjacent or correct locations to enable f7y6nzq3.
         * Approximate the minimal path length on the adjacency graph between resource location and wqid8mog location.
       - Sum those costs plus one action (f7y6nzq3) per processed wqid8mog.
    9. We use shortest paths on the adjacency graph (mz6r1d30 objects) to estimate minimal movement actions needed.
    10. Since resources (ndg963qz) are consumed per use, the heuristic penalizes for number of such resources.
        When not enough resources exist, the heuristic returns a large value.
    11. The heuristic is 0 if all goal wqid8mog have (ne6sl557).

    The heuristic is not guaranteed to be admissible but aims to be informative and efficient,
    supporting greedy best-first search.

    Implementation notes:
    - We precompute adjacency graph from static facts.
    - We implement a simple shortest-path lookup using BFS on the mz6r1d30 graph.
    - For efficiency, we do not consider full exact matching of pt4b08k4 objects; instead, assume at least one pt4b08k4 is adjacent.
    - Count the minimal resource movements plus f7y6nzq3 action per wqid8mog.

    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract adjacency graph of mz6r1d30 objects from static lpmls56r predicate
        self.adj = {}
        for fact in self.static:
            if match(fact, "lpmls56r", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.adj.setdefault(loc1, set()).add(loc2)
                self.adj.setdefault(loc2, set()).add(loc1)

        # Extract all pt4b08k4 objects from static by scanning all facts mentioning them
        # Not strictly necessary for heuristic as we treat them abstractly.

        # Extract all du770mtw objects (for resource usage) and wqid8mog objects needed in goal
        self.goal_wqid8mog = set()
        for goal in self.goals:
            if match(goal, "ne6sl557", "*"):
                _, obj = get_parts(goal)
                self.goal_wqid8mog.add(obj)

    def shortest_path_length(self, start, goal):
        """Compute shortest path length between two mz6r1d30 nodes in adjacency graph using BFS."""
        if start == goal:
            return 0
        visited = {start}
        frontier = [(start, 0)]
        while frontier:
            current, dist = frontier.pop(0)
            for neighbor in self.adj.get(current, ()):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append((neighbor, dist + 1))
        # No path found: return a large number
        return 1000

    def __call__(self, node):
        state = node.state

        # If goal reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # Extract current (ne6sl557 ?wqid8mog)
        achieved_wqid8mog = {get_parts(fact)[1] for fact in state if match(fact, "ne6sl557", "*")}

        # Wqid8mog objects still to process (goal but not achieved)
        to_process = self.goal_wqid8mog - achieved_wqid8mog
        if not to_process:
            return 0

        # Extract (of6wn8m4 ?wqid8mog), wqid8mog available for processing
        available_wqid8mog = {get_parts(fact)[1] for fact in state if match(fact, "of6wn8m4", "*")}

        # Extract all du770mtw objects with (ndg963qz ?obj)
        available_resources = {get_parts(fact)[1] for fact in state if match(fact, "ndg963qz", "*")}

        # Locations of wqid8mog: from (qkdx0xe2 ?ris62 ?ts6t4)
        wqid8mog_loc = {}
        for fact in state:
            if match(fact, "qkdx0xe2", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                # Object type unknown here; filter wqid8mog only
                if obj in self.goal_wqid8mog or obj in to_process or obj in available_wqid8mog:
                    wqid8mog_loc[obj] = loc

        # Locations of du770mtw resources: from (ndg963qz ?cz2xr) and (qkdx0xe2 ?cz2xr ?loc)
        # Note: ndg963qz is a predicate on du770mtw object
        # First find which have ndg963qz:
        # Then find their locations via qkdx0xe2 predicates

        resource_locations = {}
        for resource in available_resources:
            # location of resource:
            # find fact (qkdx0xe2 resource ?loc)
            for fact in state:
                if match(fact, "qkdx0xe2", resource, "*"):
                    resource_locations[resource] = get_parts(fact)[2]
                    break

        # Extract pt4b08k4 object locations from (fqyrzl30 ?n54xx ?cz2xr)
        # We treat pt4b08k4 abstractly, assume at least one such object exists
        # and can be moved along adjacency graph similarly.

        # For heuristic, we consider for each wqid8mog to process the minimal cost:
        # - One action f7y6nzq3 for processing it.
        # - The minimal transportation cost to bring:
        #   * a resource (ndg963qz) at some du770mtw object
        #   * a pt4b08k4 object adjacent to both wqid8mog location and resource location
        # In practice, we approximate minimal distances on adjacency graph between
        # resource location and wqid8mog location.

        # If no available resources, return a large number (unsolvable)
        if not available_resources:
            return 10000

        # Heuristic cost accumulator
        total_cost = 0

        # For each wqid8mog to process:
        for wqid_obj in to_process:
            # Must be available (of6wn8m4) to be processed, else more cost since of6wn8m4 must be obtained first
            has_of6wn8m4 = (wqid_obj in available_wqid8mog)

            # Location of wqid8mog (mz6r1d30)
            w_loc = wqid8mog_loc.get(wqid_obj, None)
            if w_loc is None:
                # If no location known, heuristic large penalty
                total_cost += 50
                continue

            # Compute minimal distance from any resource location to w_loc on adjacency graph
            min_dist = min(
                self.shortest_path_length(r_loc, w_loc) for r_loc in resource_locations.values()
            )

            # Include cost of at least one transport action for pt4b08k4 assumed approximately as min_dist as well
            # Because nm6lzp2x can move qkdx0xe2 predicates between mz6r1d30 along lpmls56r edges,
            # we assume it costs one action per step per object (resource and pt4b08k4).

            # Cost estimate:
            # - Distance for resource qkdx0xe2 movement
            # - Distance for pt4b08k4 qkdx0xe2 movement (approx same as resource)
            # - 1 action for f7y6nzq3 to produce (ne6sl557)
            # - If wqid8mog not (of6wn8m4), add 1 action penalty (to get it ready, overapproximation)

            cost = min_dist * 2 + 1  # transportation + f7y6nzq3

            if not has_of6wn8m4:
                cost += 1

            total_cost += cost

        # Also factor in that each f7y6nzq3 consumes one resource (ndg963qz)
        # If not enough resources to process all wqid8mog, add penalty
        if len(available_resources) < len(to_process):
            total_cost += (len(to_process) - len(available_resources)) * 10

        return int(total_cost)
