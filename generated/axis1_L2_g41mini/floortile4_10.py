from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(ypt9mrgu pjr7u2i2 j0on2kgr)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain (hi4l5zmb).

    Summary:
    ----------
    This heuristic estimates the number of actions required to achieve all goal facts of the form (cb6fw2xd ?loc ?obj)
    in the secret_domain. The domain appears to involve moving some entities of type `k9b7lknr` around objects of types
    `nh1sy8v9` (locations or places) and associating `qekv4465` objects with these `nh1sy8v9` objects.

    The key goal atoms are predicates of type "cb6fw2xd" binding an nh1sy8v9 object and a qekv4465 object.
    Reaching these goals requires repositioning actors (k9b7lknr) and relevant objects from their current states and locations
    across the nh1sy8v9 objects.

    Assumptions:
    ------------
    - The goal consists only of facts with predicate `cb6fw2xd` relating an nh1sy8v9 object to a qekv4465 object.
    - Achieving one goal fact mostly requires placing the right qekv4465 object onto the correct nh1sy8v9 object, 
      potentially by moving k9b7lknr actors that "carry" or "own" the qekv4465 values.
    - The predicates `ypt9mrgu ?k9b7lknr ?nh1sy8v9` track where actors currently are.
    - The predicates `fguw2142 ?nh1sy8v9` denote a special "free" or "active" nh1sy8v9 object (likely like a marker or a currently controlled location).
    - Movement and repositioning actions must move actors (k9b7lknr) among nh1sy8v9 objects and arrange qekv4465 objects appropriately.
    - Static predicates `ga4l9do0`, `s4jrpk2b`, `vjcqgdgn`, `vom4pfig` encode static adjacency or relation information between nh1sy8v9 objects.
    - Each goal fact will require at least one corresponding action to establish `cb6fw2xd` relation if missing.
    - Actions `wvswfzmn` change qp2t6u4s (binding k9b7lknr and qekv4465) facts, representing the moving/assigning of qekv4465 objects for an actor.

    Heuristic Initialization:
    -------------------------
    - Extract all goal facts with predicate `cb6fw2xd` as target pairs (nh1sy8v9, qekv4465).
    - Parse static facts to build adjacency maps for the relations `ga4l9do0`, `s4jrpk2b`, `vjcqgdgn`, and `vom4pfig`.
      These relations represent different types of adjacency or connectivity among nh1sy8v9 objects.
    - Store the static facts of predicate `fezojtj3` indicating which qekv4465 objects are considered "active" or "free".
    - Store all relevant facts and objects to allow quick checking of current positions and states.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. Identify all goal (cb6fw2xd ?loc ?obj) pairs.
    2. For each goal pair:
       - Check if the fact is already true in the current state; if yes, no action is required.
       - Otherwise, count at least one action needed to create this fact.
       - To create the fact, the actor (k9b7lknr) that carries ?obj (qekv4465) must be where the ?loc (nh1sy8v9) is or must be moved appropriately.
    3. For qekv4465 objects that are currently assigned to an actor (have qp2t6u4s ?actor ?obj), but are not at the goal location, 
       the actor will require a sequence of movements:
         - Move the actor from its current nh1sy8v9 location to the goal nh1sy8v9 location.
         - Possibly perform actions to re-assign or update the cb6fw2xd fact.
    4. Actors can move only to related nh1sy8v9 objects according to static adjacency predicates; the shortest path length between current and goal locations
       of actors (for each relevant goal) is estimated via a heuristic graph search on the adjacency derived from static facts.
    5. Each step movement of an actor and each re-assignment or action to produce the goal `cb6fw2xd` fact is counted as one action.
    6. Summing all individual minimal estimates for achieving each goal fact yields the heuristic value.
    7. The heuristic is zero if and only if all goal facts are true.
    
    This heuristic is not guaranteed to be admissible, but attempts to realistically estimate the number of moves needed by leveraging static adjacency and current actor/object positions.

    """

    def __init__(self, task):
        """
        Initialize the heuristic with static knowledge and goal conditions.
        """
        self.goals = set(task.goals)
        self.static = task.static  # Static facts never changing.
        self.facts = task.facts

        # Extract goal cb6fw2xd facts: pairs (nh1sy8v9_obj, qekv4465_obj)
        self.goal_pairs = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "cb6fw2xd":
                loc, obj = parts[1], parts[2]
                self.goal_pairs.append((loc, obj))

        # Build adjacency graphs from static facts for nh1sy8v9 objects.
        # We'll build one combined adjacency set for shortest path computations.
        self.adjacency = dict()  # nh1sy8v9 -> set of nh1sy8v9 neighbors

        def add_edge(a, b):
            self.adjacency.setdefault(a, set()).add(b)
            self.adjacency.setdefault(b, set()).add(a)

        # Relations indicating adjacency (undirected for heuristic)
        adjacency_predicates = ["ga4l9do0", "s4jrpk2b", "vjcqgdgn", "vom4pfig"]

        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in adjacency_predicates:
                n1, n2 = parts[1], parts[2]
                add_edge(n1, n2)

        # Extract set of qekv4465 objects that satisfy fezojtj3 predicate (free/active qekv4465)
        self.free_qekv4465 = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "fezojtj3":
                self.free_qekv4465.add(parts[1])

    def _shortest_path_length(self, start, goal):
        """
        Compute approximate shortest path length (in number of steps) between two nh1sy8v9 objects using BFS.
        Returns a large number if no path found (treated as unreachable).
        """
        if start == goal:
            return 0
        visited = {start}
        frontier = [start]
        distance = 0
        while frontier:
            next_frontier = []
            distance += 1
            for node in frontier:
                for neighbor in self.adjacency.get(node, ()):
                    if neighbor == goal:
                        return distance
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
        # No path found, return large number as penalty
        return 50

    def __call__(self, node):
        """
        Compute the heuristic estimate from the current node.

        Approach:
        - If all goals are already satisfied, return 0.
        - For each goal (loc, obj):
            - Check if (cb6fw2xd loc obj) is in the current state. If yes, cost 0 for this goal.
            - Otherwise:
                - Find which actor currently holds obj (i.e. which ?en18p satisfies qp2t6u4s ?en18p ?obj).
                  If none, heuristic assumes actor must pick up obj first (add small fixed cost).
                - Find current location of that actor by ypt9mrgu predicate.
                - Estimate cost to move actor from current location to goal location (loc) using shortest path length.
                - Add 1 action to generate (cb6fw2xd loc obj) fact.
                - Sum these costs over all goals.
        """
        state = node.state

        # Quick goal check: if all goal facts true, heuristic is zero.
        if self.goals <= state:
            return 0

        # Build lookup for actor carrying qekv4465 objects: qp2t6u4s ?actor ?qekv4465
        obj_to_actor = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "qp2t6u4s":
                actor, obj = parts[1], parts[2]
                obj_to_actor[obj] = actor

        # Build lookup for actor locations: ypt9mrgu ?actor ?loc
        actor_to_loc = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "ypt9mrgu":
                actor, loc = parts[1], parts[2]
                actor_to_loc[actor] = loc

        total_cost = 0

        for (goal_loc, goal_obj) in self.goal_pairs:
            goal_fact = f"(cb6fw2xd {goal_loc} {goal_obj})"
            if goal_fact in state:
                # Goal fact already reached, no cost.
                continue

            # Cost components:
            cost = 0

            # Find actor currently holding goal_obj
            actor = obj_to_actor.get(goal_obj)
            if actor is None:
                # Heuristic penalty:
                # Actor must pick up obj before moving it. Add pickup + assignment cost.
                # Assuming minimal cost 2 for pickup/reassign.
                cost += 2
                # Try to locate any actor that could pick it up by adjacency or ignore location cost here.
                # This unknown adds uncertainty.
                # Can't estimate movement cost reliably, skip to simpler fallback.
                # Adding an arbitrary penalty to reflect the effort.
                cost += 5
                total_cost += cost
                continue

            # Find actor's current location
            actor_loc = actor_to_loc.get(actor)
            if actor_loc is None:
                # Unlikely but in case actor location unknown, add penalty
                cost += 10
                total_cost += cost
                continue

            # Estimate shortest path (number of moves) between actor location and goal location
            dist = self._shortest_path_length(actor_loc, goal_loc)
            cost += dist

            # Add 1 to create the cb6fw2xd fact, assuming one action needed after arrival.
            cost += 1

            total_cost += cost

        return total_cost
