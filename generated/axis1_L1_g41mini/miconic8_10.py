from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Check if a PDDL fact matches a given pattern with possible wildcards '*'.

    Args:
        fact (str): A PDDL fact string, e.g. '(predicate arg1 arg2)'.
        pattern (str): A pattern to match against fact parts.

    Returns:
        bool: True if all parts match the pattern with wildcards, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (ud3uf11z).

    Summary
    -------
    Estimates the number of actions required to make all target objects (nxbd1hc3)
    satisfy the goal predicate (a7sb56om ?obj). The domain involves objects of types
    nxbd1hc3 and t4rdxh32 and actions that manipulate predicates:
    - kut2ksnk(?praz5, ?w3okg): connects objects prav5 to w3okg (nxbd1hc3 to t4rdxh32)
    - klw11kjf(?praz5, ?w3okg): another kind of connection between nxbd1hc3 and t4rdxh32
    - acovodrf(?vwxtg, ?zrba6): connectivity or adjacency between t4rdxh32 objects
    - f2xo7hb6(?praz5), a7sb56om(?praz5): progress predicates on nxbd1hc3 objects
    - omggbn9z(?w3okg): marks "current" or "occupied" t4rdxh32 nodes

    The goal is to have (a7sb56om ?praz5) true for certain objects paz5.

    Assumptions
    -----------
    - To achieve (a7sb56om ?praz5), a chain of actions is required:
       1) Move "omggbn9z" from one t4rdxh32 to another using acovodrf links (qs8pdrqj, i0veo5kv).
       2) Apply diwlmgg9 on a ?praz5 connected with kut2ksnk to the current omggbn9z node to set f2xo7hb6(?praz5).
       3) Apply y7nik9q2 on a ?praz5 connected with klw11kjf to the current omggbn9z node and with f2xo7hb6(?praz5) true; which removes f2xo7hb6 and sets a7sb56om(?praz5).
    - The current "omggbn9z" marker moves along a graph of t4rdxh32 nodes connected via acovodrf.
    - To service each nxbd1hc3 object (to get it to a7sb56om), the omggbn9z marker must reach appropriate nodes connected via kut2ksnk and klw11kjf.
    - There is only a single omggbn9z mark in the state (the "robot" or agent position).
    - The cost for moving between nodes is counted as one action per edge traverse.
    - For each nxbd1hc3 object to reach the goal, two actions on that object are needed (diwlmgg9 and y7nik9q2).
    - The heuristic ignores order dependencies but includes minimal estimated travel steps along the connectivity graph.

    Heuristic Initialization
    ------------------------
    - Extract all t4rdxh32 objects and store their adjacencies based on acovodrf static facts as a graph.
    - Extract kut2ksnk and klw11kjf links from static facts, for fast lookup of which nxbd1hc3 objects are connected to which t4rdxh32 nodes.
    - Extract from the goal the set of nxbd1hc3 objects that require (a7sb56om ?praz5) to be true.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1) Identify the current location of the unique agent (unique t4rdxh32 with (omggbn9z ?w3okg)).
    2) Determine which nxbd1hc3 objects still need to reach the goal (those with goal (a7sb56om ?praz5) but not yet true in the state).
    3) For each such object:
       a) Identify the pair of t4rdxh32 nodes connected by kut2ksnk and klw11kjf predicates for that object.
          These represent the nodes where diwlmgg9 and y7nik9q2 must be applied.
       b) From the current omggbn9z position, estimate shortest path distances (min number of moves) along the acovodrf graph to the kut2ksnk node, then to the klw11kjf node.
       c) Add 2 actions for diwlmgg9 and y7nik9q2 for the object.
       d) Sum these to get the cost for servicing that object.
    4) Use a greedy approach to compute a minimum approximate route:
       - Start from current agent position,
       - Pick next object with minimal travel cost,
       - Move the agent last known position to klw11kjf node of the serviced object,
       - Accumulate total estimated cost over all objects.
    5) Return the sum of all costs as heuristic estimate.

    This heuristic is not guaranteed admissible but balances accurate path cost estimates
    and operation counts for efficient greedy best-first search node expansions.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Build adjacency graph of t4rdxh32 objects from acovodrf (undirected).
        self.adjacency = {}
        self.t4rdxh32_objects = set()
        for fact in self.static:
            if match(fact, "acovodrf", "*", "*"):
                parts = get_parts(fact)
                src, tgt = parts[1], parts[2]
                self.t4rdxh32_objects.add(src)
                self.t4rdxh32_objects.add(tgt)
                self.adjacency.setdefault(src, set()).add(tgt)
                self.adjacency.setdefault(tgt, set()).add(src)

        # Map from nxbd1hc3 object to their kut2ksnk and klw11kjf connected nodes (t4rdxh32)
        self.obj_to_kut2ksnk = {}  # For diwlmgg9
        self.obj_to_klw11kjf = {}  # For y7nik9q2

        for fact in self.static:
            if match(fact, "kut2ksnk", "*", "*"):
                parts = get_parts(fact)
                obj, loc = parts[1], parts[2]
                self.obj_to_kut2ksnk[obj] = loc
            elif match(fact, "klw11kjf", "*", "*"):
                parts = get_parts(fact)
                obj, loc = parts[1], parts[2]
                self.obj_to_klw11kjf[obj] = loc

        # Extract the set of nxbd1hc3 objects that must satisfy (a7sb56om ?praz5) in goal.
        self.goal_nxbd_objects = {
            get_parts(goal)[1] for goal in self.goals if match(goal, "a7sb56om", "*")
        }

    def shortest_path_length(self, start, goal):
        """Compute shortest path length between two t4rdxh32 nodes using BFS on adjacency graph.

        Args:
            start (str): start node.
            goal (str): goal node.

        Returns:
            int: minimum number of moves along acovodrf edges, or large number if no path.
        """
        if start == goal:
            return 0
        if start not in self.adjacency or goal not in self.adjacency:
            # If nodes unknown, assign a large cost.
            return 1000

        from collections import deque

        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            current, dist = queue.popleft()
            for neighbor in self.adjacency.get(current, []):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        # No path found
        return 1000

    def __call__(self, node):
        state = node.state

        # If goal state, heuristic = 0
        if self.goals <= state:
            return 0

        # Find current agent position: unique ?w3okg with (omggbn9z ?w3okg)
        agent_positions = [get_parts(f)[1] for f in state if match(f, "omggbn9z", "*")]
        if not agent_positions:
            # No agent position -> large cost
            return 10000
        agent_pos = agent_positions[0]

        # Identify which objs still need to reach goal (a7sb56om not yet true)
        achieved_objs = {get_parts(f)[1] for f in state if match(f, "a7sb56om", "*")}
        objs_to_service = self.goal_nxbd_objects - achieved_objs

        # Track which objects are already f2xo7hb6 (mid-progress)
        mid_progress_objs = {get_parts(f)[1] for f in state if match(f, "f2xo7hb6", "*")}

        # We will attempt a simple greedy approximation:
        # For each remaining object, estimate the cost from current position visiting kut2ksnk_node,
        # then klw11kjf_node (the two locations needed), plus 2 actions for the two domain actions.

        total_cost = 0
        current_pos = agent_pos
        objs_left = set(objs_to_service)

        while objs_left:
            best_obj = None
            best_cost = None
            best_path_info = None

            for obj in objs_left:
                kut_node = self.obj_to_kut2ksnk.get(obj, None)
                klw_node = self.obj_to_klw11kjf.get(obj, None)

                if kut_node is None or klw_node is None:
                    # Missing static links, assign high cost
                    continue

                cost_to_kut = self.shortest_path_length(current_pos, kut_node)
                cost_kut_to_klw = self.shortest_path_length(kut_node, klw_node)

                if cost_to_kut >= 1000 or cost_kut_to_klw >= 1000:
                    # Unreachable or unknown, ignore this object for now
                    continue

                # Additional action costs:
                # - Move actions: cost_to_kut + cost_kut_to_klw
                # - Apply diwlmgg9 (1 action) + apply y7nik9q2 (1 action) = 2 actions total per object
                # However, if object is mid-progress (has f2xo7hb6), we assume diwlmgg9 done,
                # so only y7nik9q2 (1 action) required, and only need to move to klw_node from current_pos.
                if obj in mid_progress_objs:
                    # The f2xo7hb6 is true --> only need to move to klw11kjf node and apply y7nik9q2 (1 action).
                    cost = self.shortest_path_length(current_pos, klw_node) + 1
                else:
                    cost = cost_to_kut + cost_kut_to_klw + 2

                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_obj = obj
                    best_path_info = (kut_node, klw_node)

            if best_obj is None:
                # If no reachable object, break to avoid infinite loop
                break

            # Accumulate cost and update current position to klw_node of serviced object
            total_cost += best_cost
            current_pos = best_path_info[1]
            objs_left.remove(best_obj)

        return total_cost
