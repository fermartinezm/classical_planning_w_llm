from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing
    parentheses and splitting the string.
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    Wildcards '*' are supported.

    - fact: a string, e.g. "(qkdx0xe2 gpegssq9 w5eewe3t)"
    - args: pattern parts, e.g. "qkdx0xe2", "*", "w5eewe3t"
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary:
    ----------
    This heuristic estimates the number of actions required to achieve 
    the goal conditions in the secret_domain domain, guiding a greedy 
    best-first search. The goal requires one or more wqid8mog-type objects 
    to satisfy (ne6sl557 ?ris62).

    Assumptions:
    ------------
    - The unique goal predicates are of form (ne6sl557 ?ris62) to mark 
      success for objects of type wqid8mog.
    - Achieving (ne6sl557 ?ris62) requires applying the 'f7y6nzq3' action,
      which consumes a fact (ndg963qz ?cz2xr) true for one du770mtw object,
      and requires appropriate qkdx0xe2 and fqyrzl30 relations.
    - Intermediate movements and transfers of qkdx0xe2 facts among mz6r1d30 
      objects (via nm6lzp2x actions) and linking pt4b08k4 and du770mtw objects 
      to mz6r1d30 objects form the core of progress toward the goal.
    - Static facts (lpmls56r) define connectivity graph edges among mz6b08k4 objects,
      which can be used to compute shortest paths needed for the transitions.
    - There is only one (ndg963qz ?cz2xr) fact true at a time (capacity of the 
      relevant du770mtw object).

    Heuristic Initialization:
    --------------------------
    - Extract all mz6r1d30 objects (nodes in a connectivity graph).
    - From static facts (lpmls56r ?a ?b), build a bidirectional graph describing 
      connectivity among mz6r1d30 nodes.
    - Extract all (ndg963qz ?cz2xr) objects (these represent the "containers" 
      or key 'du770mtw' objects).
    - Extract the goal's wqid8mog objects requiring (ne6sl557 ?ris62).
    - This heuristic assumes moving qkdx0xe2 facts across the mz6r1d30 graph is 
      the principal cost driver.
    - Precompute shortest path distances (in terms of edge hops) on the connectivity 
      graph for fast distance queries.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. For each goal object of type wqid8mog requiring (ne6sl557 ?ris62),
       check if it already satisfies the goal predicate in the current state.
       - If yes, no further cost for that object.
       - If no, estimate how many actions to enable 'f7y6nzq3' for it.
       
    2. To apply action 'f7y6nzq3' for a goal object ris62:
       - There must be a du770mtw object cz2xr with (ndg963qz ?cz2xr) true.
       - The qkdx0xe2 facts must link ris62 and cz2xr to the same mz6r1d30 object ts6t4.
       - Also, the goal object ris62 must have (of6wn8m4 ris62) true.
       - The pt4b08k4 object n54xx that holds a fqyrzl30 fact linking it to cz2xr.
       - The ris62 and n54xx must share the qkdx0xe2 fact at ts6t4.
       
    3. Strategy to estimate cost:
       - Identify current qkdx0xe2 facts for the ris62 object: which mz6r1d30 node is it on?
       - Identify current qkdx0xe2 facts for the du770mtw container cz2xr (with ndg963qz true).
       - Identify the pt4b08k4 object n54xx that links via fqyrzl30 to cz2xr.
         (If multiple, pick one closest in graph sense.)
       - Check their current locations (mz6r1d30 nodes).
       - Use the precomputed shortest paths:
         * Number of nm6lzp2x actions to move ris62's qkdx0xe2 fact nodes along the graph 
           to align with cz2xr and n54xx at the same node.
         * Possibly one jql8txb3 action to link fqyrzl30 for n54xx with cz2xr.
         * Possibly one f7y6nzq3 action to conclude.
         
    4. Combine the costs for all unsatisfied goal wqid8mog objects.
       - Since f7y6nzq3 consumes (ndg963qz ?cz2xr) making it false,
         we consider the number of goal objects and assume sequential steps.

    5. Return the sum of estimated action counts as the heuristic value.
       Return 0 if the goal is already satisfied.

    Note:
    - This heuristic does not guarantee admissibility but will guide the search efficiently.
    - The heuristic is 0 for states satisfying all goal predicates.
    - Costs are based on distances in connectivity graph plus minimal fixed costs.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        self.facts = task.facts

        # Extract mz6r1d30 nodes from static facts appearing in connectivity predicates (lpmls56r).
        self.nodes = set()
        self.edges = dict()

        for fact in self.static:
            if match(fact, "lpmls56r", "*", "*"):
                parts = get_parts(fact)
                a, b = parts[1], parts[2]
                self.nodes.add(a)
                self.nodes.add(b)
                self.edges.setdefault(a, set()).add(b)
                self.edges.setdefault(b, set()).add(a)

        # Precompute shortest path distances between all pairs using BFS on undirected graph.
        self.distances = self._precompute_distances()

        # Extract all du770mtw objects with (ndg963qz ?cz2xr) fact in static or dynamic.
        # We get static facts only for connectivity, but ndg963qz changes dynamically, so initialized here as empty.
        self.du770mtw_objects = set()
        for fact in self.facts:
            if match(fact, "ndg963qz", "*"):
                self.du770mtw_objects.add(get_parts(fact)[1])
        for fact in self.static:
            if match(fact, "ndg963qz", "*"):
                self.du770mtw_objects.add(get_parts(fact)[1])

        # Extract all pt4b08k4 and wqid8mog objects appearing in relevant predicates.
        # fqyrzl30 links pt4b08k4 and du770mtw, of6wn8m4 and ne6sl557 use wqid8mog.
        self.pt4b08k4_objects = set()
        self.wqid8mog_objects = set()
        for fact in self.facts | self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "fqyrzl30":
                self.pt4b08k4_objects.add(parts[1])
            elif pred in ("of6wn8m4", "ne6sl557"):
                self.wqid8mog_objects.add(parts[1])

    def _precompute_distances(self):
        """
        Compute shortest path distances between all nodes in self.nodes
        using BFS on undirected graph self.edges.
        Returns: dict of dicts: distances[src][dst] = shortest_path_length
        """
        distances = dict()
        for node in self.nodes:
            distances[node] = self._bfs_distances(node)
        return distances

    def _bfs_distances(self, start):
        """
        BFS to compute shortest distances from start to all reachable nodes.
        """
        dist = {start: 0}
        queue = [start]
        while queue:
            current = queue.pop(0)
            for neighbor in self.edges.get(current, []):
                if neighbor not in dist:
                    dist[neighbor] = dist[current] + 1
                    queue.append(neighbor)
        return dist

    def __call__(self, node):
        state = node.state

        # Check if goal is already reached: all (ne6sl557 ?ris62) true.
        if self.goals <= state:
            return 0

        # Gather facts for queries:
        # - ndg963qz true container(s) (likely one).
        # - qkdx0xe2 facts mapping objects to mz6r1d30 nodes.
        # - fqyrzl30 linking pt4b08k4 to du770mtw.
        # - of6wn8m4 true wqid8mog.
        # - ne6sl557 goal predicate might be missing or partially true.

        # Identify container cz2xr with ndg963qz fact true in current state.
        containers = [get_parts(f)[1] for f in state if match(f, "ndg963qz", "*")]
        # For heuristic, pick first container or None if none exists
        container = containers[0] if containers else None

        # Map qkdx0xe2 facts: object -> mz6r1d30 node they currently occupy.
        # There might be multiple such facts per object; we assume one per object.
        # We build multi-mapping as qkdx_map[obj] = set(locations)
        qkdx_map = {}
        for fact in state:
            if match(fact, "qkdx0xe2", "*", "*"):
                parts = get_parts(fact)
                obj = parts[1]
                loc = parts[2]
                qkdx_map.setdefault(obj, set()).add(loc)

        # Map fqyrzl30 facts: pt4b08k4 object -> set(du770mtw objects)
        fqyrzl30_map = {}
        for fact in state:
            if match(fact, "fqyrzl30", "*", "*"):
                parts = get_parts(fact)
                pt_obj = parts[1]
                du_obj = parts[2]
                fqyrzl30_map.setdefault(pt_obj, set()).add(du_obj)

        # Identify wqid8mog objects that have of6wn8m4 true in current state.
        of6wn8m4_true = {get_parts(f)[1] for f in state if match(f, "of6wn8m4", "*")}

        # Helper function to get shortest distance between two mz6r1d30 nodes.
        def dist(n1, n2):
            if n1 in self.distances and n2 in self.distances[n1]:
                return self.distances[n1][n2]
            return 10**6  # Large number if unreachable.

        total_estimated_cost = 0

        # For each goal wqid8mog object requiring ne6sl557, estimate cost if not reached yet.
        for fact in self.goals:
            if not match(fact, "ne6sl557", "*"):
                continue
            goal_obj = get_parts(fact)[1]
            if fact in state:
                # Goal achieved for this object, no cost.
                continue

            # To achieve ne6sl557(goal_obj), f7y6nzq3 must be done.

            # Check that ris62 (=goal_obj) has of6wn8m4 true:
            if goal_obj not in of6wn8m4_true:
                # Missing prerequisite: must produce of6wn8m4 first.
                # No direct action provided for changing of6wn8m4,
                # assume 1 action to produce it as approximation.
                cost_of6wn8m4 = 1
            else:
                cost_of6wn8m4 = 0

            # We need to unify qkdx0xe2 facts for:
            # ris62, container du770mtw with ndg963qz, and pt4b08k4 linked via fqyrzl30.

            # Candidate pt4b08k4 objects linked to container:
            candidate_pt4b08k4 = [pt for pt, dus in fqyrzl30_map.items() if container in dus]

            if not candidate_pt4b08k4 or container is None:
                # Cannot form precondition for f7y6nzq3; heuristic high.
                total_estimated_cost += 50 + cost_of6wn8m4
                continue

            # We take the first pt4b08k4 for estimation.
            pt_obj = candidate_pt4b08k4[0]

            # Get current qkdx0xe2 locations for ris62, container, pt_obj.
            ris62_locs = qkdx_map.get(goal_obj, set())
            container_locs = qkdx_map.get(container, set())
            pt_obj_locs = qkdx_map.get(pt_obj, set())

            # If any qkdx0xe2 fact is missing, cost is high.
            if not ris62_locs or not container_locs or not pt_obj_locs:
                total_estimated_cost += 30 + cost_of6wn8m4
                continue

            # We want to bring all three to the same mz6r1d30 node ts6t4.
            # For estimation, find minimal sum of pairwise distances over all triples.
            min_total_moves = 10**6
            for ris_loc in ris62_locs:
                for cont_loc in container_locs:
                    for pt_loc in pt_obj_locs:
                        # Cost to bring ris62 to meeting node + pt_obj to meeting node + container to meeting node.
                        # Because container has ndg963qz and only one such object we focus on aligning ris62 and pt_obj to container loc.

                        # Treated as moving ris62 and pt_obj qkdx0xe2 nodes to container_loc:
                        move_r = dist(ris_loc, cont_loc)
                        move_p = dist(pt_loc, cont_loc)
                        total_moves = move_r + move_p
                        if total_moves < min_total_moves:
                            min_total_moves = total_moves
            # Estimated nm6lzp2x action cost = min_total_moves

            # Need one jql8txb3 action to create (fqyrzl30 ?n54xx ?cz2xr) if not already linked
            # But since fqyrzl30 links pt_obj to container in current state, no action needed.

            # One f7y6nzq3 action to produce ne6sl557.

            # Total estimated cost for this goal object:
            # nm6lzp2x moves + 1 f7y6nzq3 + possibly 1 of6wn8m4 production + slight slack

            estimated_cost = min_total_moves + 1 + cost_of6wn8m4

            total_estimated_cost += estimated_cost

        # Return heuristic value at least 1 if not goal (to avoid zero when no info)
        return max(total_estimated_cost, 1)
