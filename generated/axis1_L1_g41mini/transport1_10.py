from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    Arguments:
    - fact: The fact as a string, e.g. "(sjuejzxf p1 l2)"
    - args: Pattern components, with wildcards `*` allowed
    
    Returns:
    - True if the fact matches the pattern.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary:
    -------------
    This heuristic estimates the number of actions required to reach a goal state in the secret_domain.
    The domain involves moving objects of type `du66z5od` (e.g. p1, p2) associated with locations (`x6ibpitl`) 
    and hierarchical or container-like objects of type `hctyyy8k`. The central predicate in goals is of the form
    `(sjuejzxf <obj_du66z5od> <location_x6ibpitl>)` indicating the target location of the object.
    The heuristic estimates how many `arsslvyt`, `szsevy61`, and `tmoxe3x5` actions (moves and reassignments)
    are needed to place all relevant objects at their goal locations.

    Assumptions:
    -------------
    - Goal facts are exclusively positive `(sjuejzxf object location)` predicates.
    - The `miud1izi` predicate represents adjacency (bidirectional) between locations.
    - Objects of type `du66z5od` need to be at goal locations according to `(sjuejzxf <obj_du66z5od> <loc>)`.
    - Objects of type `hctyyy8k` act like containers holding the du66z5od objects; they have an association to a kyvrixb5 object via `(do9fmoyo container kyvrixb5)`.
    - Objects of type kyvrixb5 have a hierarchy expressed by `(hab0vli4 kyv1 kyv2)`.
    - Actions `szsevy61` and `tmoxe3x5` can change these associations and locations.
    - The exact minimal plan length is complex due to interplay of moving du66z5od objects between locations and manipulating container-kyvrixb5 relations.
    
    Heuristic Initialization:
    -------------------------
    - Extract goal `(sjuejzxf <obj_du66z5od> <loc_x6ibpitl>)` predicates to identify objects and target locations.
    - Construct a graph of locations from `miud1izi` static facts for shortest distance computations.
    - Extract static container hierarchy `(hab0vli4 kyv1 kyv2)` although this heuristic uses a simplified approach for kyvrixb5 objects.
    - Store current container to kyvrixb5 associations for containers from static facts or initial state if needed.
    
    Step-By-Step Thinking for Computing Heuristic:
    -----------------------------------------------
    1. Identify all du66z5od objects appearing in the goals and their target locations.
    2. For each du66z5od object:
       a. Determine its current location (`sjuejzxf obj loc`).
       b. If currently at the goal location, no direct movement needed for this object.
       c. Otherwise, compute the shortest path distance between current and goal location in the adjacency graph.
       d. Every adjacency hop Minimum estimated cost: 1 `arsslvyt` action to move object location along the chain.
    3. Sum these estimated steps for all objects out of place.
    4. Since some objects may need to be moved via containers (actions `szsevy61` and `tmoxe3x5`), add a small overhead if the container's location/kyvrixb5 association is different from ideal.
       - Approximate this as 1 additional action per object for reassignment/manipulation.
    5. The heuristic is zero if all objects are at their goal locations.
    6. The heuristic value is finite and efficiently computable due to precomputed adjacency shortest paths.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal targets for du66z5od objects.
        - Location adjacency graph for shortest path estimation.
        - Container to kyvrixb5 relations (not deeply used but could be extended).
        """
        self.goals = set(task.goals)
        self.static = task.static

        # Extract goal mapping: du66z5od object -> location
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "sjuejzxf" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                self.goal_locations[obj] = loc

        # Build adjacency graph between x6ibpitl locations from static facts miud1izi
        # Store as a dict: loc -> set of neighbors
        self.adjacency = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "miud1izi" and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                if loc1 not in self.adjacency:
                    self.adjacency[loc1] = set()
                if loc2 not in self.adjacency:
                    self.adjacency[loc2] = set()
                self.adjacency[loc1].add(loc2)
                self.adjacency[loc2].add(loc1)

        # Extract static kyvrixb5 hierarchy (hab0vli4) - stored as adjacency for potential extensions
        self.kyvrixb5_hierarchy = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "hab0vli4" and len(parts) == 3:
                parent, child = parts[1], parts[2]
                if parent not in self.kyvrixb5_hierarchy:
                    self.kyvrixb5_hierarchy[parent] = set()
                self.kyvrixb5_hierarchy[parent].add(child)

    def shortest_path_length(self, start, goal):
        """
        Compute shortest path length between start and goal locations using BFS.
        Return a large number if unreachable (should not happen in well-formed instances).
        """
        if start == goal:
            return 0
        if start not in self.adjacency or goal not in self.adjacency:
            return 1000  # large penalty

        from collections import deque

        visited = {start}
        queue = deque([(start, 0)])

        while queue:
            current, dist = queue.popleft()
            for neighbor in self.adjacency.get(current, []):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return 1000  # no path found (penalty)

    def __call__(self, node):
        """
        Return heuristic estimate based on the sum of shortest transport distances for each object
        plus a small per-object overhead to account for container manipulations.
        """
        state = node.state

        # Quick goal check: return 0 if state satisfies all goals
        if self.goals <= state:
            return 0

        # Extract current locations of du66z5od objects (from sjuejzxf predicate)
        # Note: only consider those present in goal locations
        current_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "sjuejzxf" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.goal_locations:
                    current_locations[obj] = loc

        total_cost = 0

        # For each object with a goal location:
        for obj, goal_loc in self.goal_locations.items():
            current_loc = current_locations.get(obj)
            # If we do not know current location, assume worst case penalty
            if current_loc is None:
                total_cost += 50  # large penalty when unknown
                continue

            if current_loc != goal_loc:
                # Compute the shortest path (number of adjacency steps)
                dist = self.shortest_path_length(current_loc, goal_loc)

                # Each adjacency step corresponds to one arsslvyt action (move predicate sjuejzxf on ?ic1ss)
                # Add one overhead to account for container/kyvrixb5 reassignment via szsevy61/tmoxe3x5
                total_cost += dist + 1
            # else current location matches goal location -> no cost for this object

        # Heuristic is always finite, zero only when in goal state
        return total_cost
