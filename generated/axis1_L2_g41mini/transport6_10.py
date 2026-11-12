from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    Example: "(jft9ynda pewaka9c dj7sycy5)" -> ["jft9ynda", "pewaka9c", "dj7sycy5"]
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - fact: The complete fact as a string, e.g., "(jft9ynda pewaka9c dj7sycy5)".
    - args: The expected pattern (wildcards '*' allowed).
    Returns True if the fact matches the pattern; False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain (mtfkspuf) domain.

    Summary:
    This heuristic estimates the number of actions needed to achieve all goal conditions,
    focusing primarily on the "jft9ynda" predicates that appear in the goals.
    This predicate relates an object of type hodp29q0 to an object of type hmhptaxe,
    which seems central for achieving the goals.
    
    Actions "k12p0czo" and the paired relations z6o4oitz define a form of
    'movement' or 'change of association' between hmhptaxe objects.
    Actions "sw2fg9ql" and "vyn92k55" modify "yarbdfgc", "jft9ynda", "uypgj7u3" and "jf4qno7u".
    The heuristic uses static mappings from "z6o4oitz" as a graph for path computation.
    It estimates the minimal steps required for each goal fact by summing:
      - the minimum number of "k12p0czo" hops to transform the current hmhptaxe object
        associated with a hodp29q0 object to the goal hmhptaxe object,
      - plus cost estimates for required changes in r86varlt associations (through jf4qno7u),
      - plus the involvement of g83opnlj and hodp29q0 objects needed for "sw2fg9ql" and "vyn92k55".

    Assumptions:
    - The main goal facts are on predicate "jft9ynda" linking hodp29q0 and hmhptaxe objects.
    - The hierarchy and chains created by "z6o4oitz" are static and define a graph structure.
    - The facts "yarbdfgc" and "jf4qno7u" link hodp29q0 and r86varlt objects, which affect 
      whether state updates through actions "sw2fg9ql"/"vyn92k55" are needed.
    - The relations (predicates) in the goal are a subset of the predicates used in the domain,
      mainly "jft9ynda".

    Heuristic Initialization:
    - Extract all goal facts.
    - Build a bidirectional adjacency graph from static facts "z6o4oitz" on hmhptaxe objects.
    - Index current associations for "jft9ynda" per hodp29q0 object.
    - Collect "yarbdfgc" and "jf4qno7u" facts for quick lookup.
    - Collect "uypgj7u3" and "jft9ynda" facts involving g83opnlj and hodp29q0 for heuristic reference.

    Step-By-Step Thinking for Computing Heuristic:
    1. For each goal fact (mostly "jft9ynda ?hodp29q0 ?hmhptaxe_goal"):
       a. Find the current associated hmhptaxe object for the hodp29q0 object.
          If none, assume max cost (no path).
       b. Compute shortest path length from current hmhptaxe to goal hmhptaxe via the z6o4oitz graph.
          This estimates minimal required "k12p0czo" actions.
    2. For each "yarbdfgc" and "jf4qno7u" fact involving hodp29q0 and r86varlt objects,
       estimate if value changes are needed (since the actions "sw2fg9ql" / "vyn92k55" update these).
       This increments heuristic cost because these require corresponding actions.
    3. Include estimated costs related to coordinating g83opnlj and hodp29q0 interactions,
       especially where "uypgj7u3" and "jft9ynda" facts mismatch goal.
    4. Sum all these costs as the output heuristic value; zero if all goals already met.
    5. The heuristic is non-admissible but tries to reflect the minimal combined number
       of required coordination and transformation actions.

    The heuristic is efficient by precomputing static graphs and using BFS for path length estimates;
    it avoids expensive search or relaxations.
    """

    def __init__(self, task):
        """Initialize heuristic by processing static facts and goals."""
        self.goals = task.goals  # Set of goal facts (strings)
        self.static = task.static
        self.static_z6o4oitz_graph = {}

        # Build undirected graph from static facts "z6o4oitz ?a ?b"
        # We treat "z6o4oitz" as edges on hmhptaxe objects for shortest path calculations.
        for fact in self.static:
            if match(fact, "z6o4oitz", "*", "*"):
                _, a, b = get_parts(fact)
                self.static_z6o4oitz_graph.setdefault(a, set()).add(b)
                self.static_z6o4oitz_graph.setdefault(b, set()).add(a)

        # Extract goal associations: relevant for "jft9ynda" predicates
        # Index goal jft9ynda by hodp29q0-object -> desired hmhptaxe-object
        self.goal_jft9ynda = {}
        for goal_fact in self.goals:
            if match(goal_fact, "jft9ynda", "*", "*"):
                _, hodp29q0_obj, hmhptaxe_obj = get_parts(goal_fact)
                self.goal_jft9ynda[hodp29q0_obj] = hmhptaxe_obj

    def shortest_path_length(self, graph, start, goal):
        """
        Compute shortest path length between start and goal nodes in undirected graph using BFS.
        Returns integer length if path exists, else a large number as penalty.
        """
        if start == goal:
            return 0
        visited = {start}
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            for neighbor in graph.get(current, []):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        # No path found; return penalty
        return 10**6

    def __call__(self, node):
        """Estimate number of actions to reach goal from given state."""
        state = node.state

        # If goal is already reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # Extract current "jft9ynda" mappings: hodp29q0 -> hmhptaxe
        current_jft9ynda = {}
        for fact in state:
            if match(fact, "jft9ynda", "*", "*"):
                _, hodp29q0_obj, hmhptaxe_obj = get_parts(fact)
                current_jft9ynda[hodp29q0_obj] = hmhptaxe_obj

        # Extract current "yarbdfgc" facts: hodp29q0 -> r86varlt
        current_yarbdfgc = {}
        for fact in state:
            if match(fact, "yarbdfgc", "*", "*"):
                _, hodp29q0_obj, r86varlt_obj = get_parts(fact)
                current_yarbdfgc[hodp29q0_obj] = r86varlt_obj

        # Extract current "jf4qno7u" facts: r86varlt -> r86varlt (mapping)
        current_jf4qno7u = {}
        for fact in state:
            if match(fact, "jf4qno7u", "*", "*"):
                _, r86a, r86b = get_parts(fact)
                current_jf4qno7u[r86a] = r86b

        # Extract goal "yarbdfgc" and "jf4qno7u" facts if any (not common from given info)
        goal_yarbdfgc = {}
        goal_jf4qno7u = {}
        for fact in self.goals:
            if match(fact, "yarbdfgc", "*", "*"):
                _, hodp29q0_obj, r86varlt_obj = get_parts(fact)
                goal_yarbdfgc[hodp29q0_obj] = r86varlt_obj
            elif match(fact, "jf4qno7u", "*", "*"):
                _, r86a, r86b = get_parts(fact)
                goal_jf4qno7u[r86a] = r86b

        # Extract current "uypgj7u3" facts for possible interactions hodp29q0 and g83opnlj
        current_uypgj7u3 = set()
        for fact in state:
            if match(fact, "uypgj7u3", "*", "*"):
                current_uypgj7u3.add(fact)

        # Cost accumulator
        cost = 0

        # --- 1. "jft9ynda" predicate: move the hodp29q0 object's hmhptaxe association ---
        for hodp29q0_obj, goal_hmhptaxe in self.goal_jft9ynda.items():
            current_hmhptaxe = current_jft9ynda.get(hodp29q0_obj)
            if not current_hmhptaxe:
                # No current association means likely many steps to achieve
                cost += 10  # Arbitrary penalty
                continue

            dist = self.shortest_path_length(
                self.static_z6o4oitz_graph, current_hmhptaxe, goal_hmhptaxe
            )
            cost += dist  # Estimate how many k12p0czo actions needed

            # If current association matches, zero cost here.

        # --- 2. Check necessary changes for "yarbdfgc" and linked "jf4qno7u" fact chains ---
        # Heuristic: Count how many r86varlt assignments differ between current and goal for each hodp29q0,
        # because sw2fg9ql and vyn92k55 depend on these relations.
        for hodp29q0_obj in set(current_yarbdfgc.keys()).union(goal_yarbdfgc.keys()):
            cur_r86 = current_yarbdfgc.get(hodp29q0_obj)
            goal_r86 = goal_yarbdfgc.get(hodp29q0_obj)
            if goal_r86 and cur_r86 != goal_r86:
                # Check associated jf4qno7u mappings for r86varlt involved
                # Count a cost for changing the r86varlt pointer chain.
                # We estimate number of hops to convert cur_r86 to goal_r86 on jf4qno7u mapping.
                # jf4qno7u form a relation r86varlt -> r86varlt (like a chain)
                # We try to find distance in this chain from cur_r86 to goal_r86:
                dist = 0
                # Follow chain from cur_r86 up to 10 steps to find goal_r86
                r = cur_r86
                found = False
                for i in range(10):
                    if r == goal_r86:
                        dist = i
                        found = True
                        break
                    r = current_jf4qno7u.get(r)
                    if r is None:
                        break
                if not found:
                    dist = 5  # fallback penalty
                cost += 2 + dist  # Cost for swap actions (sw2fg9ql/vyn92k55) plus chain hops

        # --- 3. Interaction of g83opnlj and hodp29q0 through uypgj7u3 / jft9ynda ---
        # Heuristic: number of goal jft9ynda on g83opnlj objects minus those currently holding uypgj7u3
        # This reflects potential need for coordination actions.
        # Extract current jft9ynda for g83opnlj to hmhptaxe:
        current_jft9ynda_g83 = {}
        for fact in state:
            if match(fact, "jft9ynda", "*", "*"):
                pred, g_obj, hm_obj = get_parts(fact)
                # g_obj could be g83opnlj based on type in domain
                current_jft9ynda_g83[g_obj] = hm_obj

        # Extract goal jft9ynda on g83opnlj, if any
        goal_jft9ynda_g83 = {}
        for fact in self.goals:
            if match(fact, "jft9ynda", "*", "*"):
                pred, g_obj, hm_obj = get_parts(fact)
                # g_obj is g83opnlj type if it's not a hodp29q0 key in goal_jft9ynda
                if g_obj not in self.goal_jft9ynda:
                    goal_jft9ynda_g83[g_obj] = hm_obj

        # Count difference in those needing to switch hmhptaxe association
        for g_obj, goal_hmhptaxe in goal_jft9ynda_g83.items():
            cur_hmhptaxe = current_jft9ynda_g83.get(g_obj)
            if cur_hmhptaxe != goal_hmhptaxe:
                # Estimate cost similarly as above (1 per needed reassociation)
                cost += 1

        # Count how many uypgj7u3 facts differ by facts involved in goal
        # This approximates number of vyn92k55 actions needed (cost ~1 per difference)
        goal_uypgj7u3 = set()
        for fact in self.goals:
            if match(fact, "uypgj7u3", "*", "*"):
                goal_uypgj7u3.add(fact)

        uypgj7u3_diff = (goal_uypgj7u3 - current_uypgj7u3) | (current_uypgj7u3 - goal_uypgj7u3)
        cost += len(uypgj7u3_diff)

        return cost
