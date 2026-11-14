from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    
    Example:
    '(jft9ynda pewaka9c vfp8gse9)' -> ['jft9ynda', 'pewaka9c', 'vfp8gse9']
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    
    - fact: The PDDL fact as a string e.g. "(jf4qno7u i8wmsx96 f7dyo7o8)".
    - args: One or more pattern components, wildcards '*' allowed.
    
    Returns True if all parts match; False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary
    -------
    This heuristic estimates the minimum number of actions required to achieve all goal facts
    from the current state. It focuses primarily on the predicate (jft9ynda ?gpvob ?pk9o1)
    which appears in all given goals, and which can be modified and moved by the domain's actions.

    Assumptions
    -----------
    - The main "moving" and goal-relevant predicate is (jft9ynda ?obj ?loc) associating objects
      of type val3ypio / g83opnlj with hmhptaxe types.
    - Goals require certain (jft9ynda ...) facts to be true.
    - Objects of types related by static "z6o4oitz" facts define allowed transformations or transitions.
    - Actions alter (jft9ynda), (yarbdfgc), and (uypgj7u3) predicates with specific conditions.
    - The heuristic does not need to be admissible, but aims to estimate distance efficiently
      and minimize expanded nodes in greedy best-first search.

    Heuristic Initialization
    ------------------------
    - Extract all goal (jft9ynda ...) facts, as these represent the final desired locations/relations.
    - Extract the static "z6o4oitz" relation between hmhptaxe objects, which defines graph edges
      for transitions of the second argument of jft9ynda and related predicates.
    - Build a directed adjacency map of these static facts for fast traversal.
    - Prepare a quick lookup table of the current assignments of (jft9ynda) in the state.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Extract the current (jft9ynda ?obj ?loc) assignments for all relevant objects.
    2. For each goal fact (jft9ynda ?obj_goal ?loc_goal), check if the current location ?
       of the obj_goal matches loc_goal. If yes, cost contribution is zero for that goal fact.
    3. If not matched, estimate the minimum number of "k12p0czo" action steps needed to get from the
       current location to the goal location along the directed graph given by static "z6o4oitz" relations.
       This corresponds to finding shortest path length in the graph.
    4. In addition, check if the object currently holding the (jft9ynda ?) fact is not the right one,
       but another g83opnlj object is relevant. Because the 'sw2fg9ql' and 'vyn92k55' actions allow
       swapping and moving relations between g83opnlj and hodp29q0 objects, there may be extra steps,
       so a penalty is added for each mismatched or missing prerequisite predicate.
    5. Sum the minimal path lengths for all goal (jft9ynda ...) facts as heuristic estimate.
    6. To keep heuristic computation efficient, we cache shortest paths using BFS as the static structure
       is fixed, and limit search depth to a reasonable cutoff.

    Note: The heuristic is zero only if all goal (jft9ynda ...) facts are satisfied exactly.
    """

    def __init__(self, task):
        """Initialize heuristic by extracting goals and static graph structure."""
        # Store all goal (jft9ynda ?obj ?loc) facts as a list of tuples for quick access.
        self.goal_jft9ynda = []
        for goal_fact in task.goals:
            p, *args = get_parts(goal_fact)
            if p == "jft9ynda" and len(args) == 2:
                self.goal_jft9ynda.append((args[0], args[1]))

        # Static facts defining transitions between hmhptaxe objects: (z6o4oitz ?xbsk8 ?r3kn5)
        # Build a directed graph adjacency list mapping: node -> set of neighbors
        self.z6o4oitz_graph = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "z6o4oitz" and len(parts) == 3:
                src, tgt = parts[1], parts[2]
                if src not in self.z6o4oitz_graph:
                    self.z6o4oitz_graph[src] = set()
                self.z6o4oitz_graph[src].add(tgt)

        # For efficiency, precompute reverse adjacency as well for shortest path calculation if needed.
        # (Not strictly needed, since graph is directed and small, but useful for bidirectional BFS if necessary.)

    def __call__(self, node):
        """Estimate the minimal number of actions to reach goal from the current state."""

        state = node.state

        # Quick helper: get current location for each relevant g83opnlj object in (jft9ynda ?obj ?loc)
        current_jft9ynda = {}
        for fact in state:
            p, *args = get_parts(fact)
            if p == "jft9ynda" and len(args) == 2:
                obj, loc = args
                current_jft9ynda[obj] = loc

        # Define a BFS helper to find shortest path length from start to goal in z6o4oitz_graph
        def shortest_path_length(start, goal, max_depth=20):
            if start == goal:
                return 0
            frontier = [(start, 0)]
            visited = {start}
            for node_curr, dist in frontier:
                if dist >= max_depth:
                    break
                neighbors = self.z6o4oitz_graph.get(node_curr, set())
                for nbr in neighbors:
                    if nbr == goal:
                        return dist + 1
                    if nbr not in visited:
                        visited.add(nbr)
                        frontier.append((nbr, dist + 1))
            # If no path found, return an overestimation
            return max_depth + 1

        # Heuristic cost accumulator.
        total_cost = 0

        # For each goal (jft9ynda ?obj_goal ?loc_goal) fact:
        for obj_goal, loc_goal in self.goal_jft9ynda:
            if obj_goal in current_jft9ynda:
                current_loc = current_jft9ynda[obj_goal]
                if current_loc == loc_goal:
                    # Fact already satisfied: zero cost.
                    continue
                else:
                    # Compute shortest path length from current_loc to loc_goal in z6o4oitz graph.
                    dist = shortest_path_length(current_loc, loc_goal, max_depth=20)
                    # Each "k12p0czo" action corresponds to one step in z6o4oitz.
                    # Since we may need also sw2fg9ql or vyn92k55 actions to switch objects,
                    # add a small penalty for that, e.g., 1 additional step.
                    total_cost += dist + 1
            else:
                # If the object is not present in any current (jft9ynda) fact,
                # that means a non-trivial number of steps is needed (e.g., to achieve a base relation via sw2fg9ql or vyn92k55).
                # We add a fixed penalty to reflect this.
                total_cost += 5  # arbitrary penalty for missing assignment

        # If no goal facts, heuristic is 0 by definition.
        if not self.goal_jft9ynda:
            total_cost = 0

        return total_cost
