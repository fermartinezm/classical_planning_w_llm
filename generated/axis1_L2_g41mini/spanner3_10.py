from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(qkdx0xe2 obj1 obj2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (du770mtw).

    Summary
    -------
    This heuristic estimates the number of actions required to achieve the goals
    of the secret_domain by considering how to make all wqid8mog objects satisfy
    the `(ne6sl557 ?ris62)` predicate (the goal), starting from the current state.

    Assumptions
    -----------
    - The goal predicates are `(ne6sl557 ?ris62)` for some wqid8mog objects.
    - To achieve `(ne6sl557 ?ris62)` for a wqid8mog object, the `f7y6nzq3` action must be applied:
      it requires:
        * a du770mtw object with `(ndg963qz ?cz2xr)` true,
        * the `wqid8mog` object has `(of6wn8m4 ?ris62)` true initially,
        * `(qkdx0xe2 ?ris62 ?ts6t4)` true for some mz6r1d30 object ?ts6t4,
        * `(fqyrzl30 ?n54xx ?cz2xr)` true for some pt4b08k4 ?n54xx,
        * `(qkdx0xe2 ?n54xx ?ts6t4)` true,
      and after `f7y6nzq3` the wqid8mog object loses `(of6wn8m4)` and gains `(ne6sl557)`.
    - Objects of type mz6r1d30 are linked by `(lpmls56r ?from ?to)`, which form chains.
    - The `nm6lzp2x` actions can move pt4b08k4 objects along these chains,
      affecting which `(qkdx0xe2 ?n54xx ?ts6t4)` facts hold.
    - The `jql8txb3` action can move `(fqyrzl30 ?n54xx ?cz2xr)` from one du770mtw to another,
      but requires `(qkdx0xe2 ?cz2xr ?ts6t4)` and `(qkdx0xe2 ?n54xx ?ts6t4)` to hold for the same ?ts6t4.
    - A du770mtw object can only have `(ndg963qz)` true for exactly one object at a time,
      since applying `f7y6nzq3` removes this predicate.
    - The heuristic is a non-admissible estimate that tries to reflect:
       * For each wqid8mog object not yet `(ne6sl557)`, estimate the steps needed to
         move it from `(of6wn8m4)` and current `(qkdx0xe2 ?ris62 ?ts6t4)`
         along chains of mz6r1d30 objects via `nm6lzp2x` actions until it can
         be "processed" by an `f7y6nzq3` action using some du770mtw with `(ndg963qz)`.
       * For each pt4b08k4 object `(n54xx)`, estimate how to move its `(fqyrzl30 ?n54xx ?cz2xr)`
         if needed to a du770mtw with `(ndg963qz)`.
       * The heuristic sums cost estimates for these adjustments plus a cost for `f7y6nzq3` executions,
         weighted to prioritize completing all needed `(ne6sl557)` facts.

    Heuristic Initialization
    ------------------------
    - Extract all static `(lpmls56r ?from ?to)` links to build chains between mz6r1d30 objects.
    - From static facts, build adjacency maps to quickly estimate shortest distances along chains.
    - Extract goal wqid8mog objects (those appearing in `(ne6sl557 ?ris62)` goals).
    - Record all pt4b08k4 and du770mtw objects for lookup.
    
    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify all wqid8mog goal objects for which `(ne6sl557 ?ris62)` must hold (those in goals).
    2. For each goal wqid8mog object:
       a) Check if it is already `(ne6sl557)`. If yes, no cost for it.
       b) Else, it must be processed by an `f7y6nzq3` action:
          - The wqid8mog object must have `(of6wn8m4)` true (otherwise unreachable goal).
          - It needs a du770mtw object with `(ndg963qz)` true.
          - It needs a pt4b08k4 object `(n54xx)` such that:
            * `(fqyrzl30 ?n54xx ?cz2xr)` on that du770mtw object,
            * `(qkdx0xe2 ?n54xx ?ts6t4)` and `(qkdx0xe2 ?ris62 ?ts6t4)` for the same chain node ?ts6t4.
          - If these conditions are not met, estimate:
            i) Actions to move `(fqyrzl30 ?n54xx ?cz2xr)` to the right du770mtw `(ndg963qz)`.
            ii) Actions to move the pt4b08k4 object along the chain via `nm6lzp2x` actions to match the mz6r1d30 node of the wqid8mog chain node.
          - Estimate minimal steps using chain distances.
          - Check if a du770mtw with `(ndg963qz)` is present. If not, estimate cost for moving/changing it.
    3. Add 1 action cost for the required `f7y6nzq3` execution per object.
    4. Sum these for all wqid8mog goal objects not already `(ne6sl557)`.
    5. Return the sum as the heuristic estimate.

    The heuristic thus approximates the necessary movements and final processing steps,
    trading accuracy for efficiency and strong guidance towards goal completion.
    """

    def __init__(self, task):
        # Store goals
        self.goals = task.goals

        # Static facts for chain extraction
        static_facts = task.static

        # Extract lpmls56r edges representing a chain/graph between mz6r1d30 objects
        self.chain_graph = {}
        for fact in static_facts:
            if match(fact, "lpmls56r", "*", "*"):
                _, from_node, to_node = get_parts(fact)
                self.chain_graph.setdefault(from_node, []).append(to_node)

        # To allow bidirectional search for shortest chain paths, build reversed edges too
        for from_node, to_nodes in list(self.chain_graph.items()):
            for to_node in to_nodes:
                self.chain_graph.setdefault(to_node, []).append(from_node)

        # Extract wqid8mog, pt4b08k4, du770mtw objects from task facts and static (object typing not always explicit)
        # Instead, extract by observed predicates from operators or goals/facts if possible
        # Since example states and goals exist as facts, scan to identify objects by predicate usage

        # Set of all mz6r1d30 objects (chain nodes) from chain_graph keys and values
        self.mz6r1d30_objects = set(self.chain_graph.keys())
        for tos in self.chain_graph.values():
            self.mz6r1d30_objects.update(tos)

        # Helper sets to cache known objects by type from initial state and goals
        self.pt4b08k4_objects = set()
        self.du770mtw_objects = set()
        self.wqid8mog_objects = set()

        # Scan initial state and goals for objects used as types from predicates
        for fact in task.facts:
            pred, *args = get_parts(fact)
            if pred == "fqyrzl30":  # (fqyrzl30 ?n54xx - pt4b08k4 ?cz2xr - du770mtw)
                if len(args) >= 2:
                    self.pt4b08k4_objects.add(args[0])
                    self.du770mtw_objects.add(args[1])
            elif pred == "qkdx0xe2":
                # qkdx0xe2 can be (d11dxhbr,mz6r1d30) or (pt4b08k4,mz6r1d30) or (wqid8mog,mz6r1d30)
                # Different first args' types unknown by parsing, but we can extract all involved objects
                # Will extract wqid8mog as those appearing also in (of6wn8m4 ?ris62) or (ne6sl557 ?ris62)
                pass

            elif pred == "of6wn8m4" or pred == "ne6sl557":
                if len(args) >= 1:
                    self.wqid8mog_objects.add(args[0])
            elif pred == "ndg963qz":
                if len(args) >= 1:
                    self.du770mtw_objects.add(args[0])

        # Some objects like qkdx0xe2 first argument are of different types:
        # but we focus only on:
        # - wqid8mog for goal targets (ne6sl557)
        # - pt4b08k4 for carrier objects (n54xx)
        # - du770mtw for location holders (cz2xr)
        # We identify all pt4b08k4 and du770mtw seen in fqyrzl30 and ndg963qz facts.

        # Build cache for shortest chain distances between mz6r1d30 objects using BFS
        self._chain_dist_cache = {}

    def _chain_distance(self, start, goal):
        """
        Compute shortest distance between two mz6r1d30 objects along the chain graph.
        Caches results for efficiency.
        """
        if (start, goal) in self._chain_dist_cache:
            return self._chain_dist_cache[(start, goal)]

        if start == goal:
            self._chain_dist_cache[(start, goal)] = 0
            return 0

        visited = {start}
        frontier = [(start, 0)]
        while frontier:
            current, dist = frontier.pop(0)
            for neighbor in self.chain_graph.get(current, []):
                if neighbor == goal:
                    self._chain_dist_cache[(start, goal)] = dist + 1
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append((neighbor, dist + 1))
        # No path found: return a large distance to avoid infinite
        self._chain_dist_cache[(start, goal)] = 1000
        return 1000

    def __call__(self, node):
        state = node.state

        # If goal reached, heuristic = 0
        if self.goals <= state:
            return 0

        # Map to quickly find:
        # (qkdx0xe2 ?obj ?mz6r1d30): which mz6r1d30 an object (du770mtw, pt4b08k4, wqid8mog) is at
        obj_pos = {}
        for fact in state:
            if match(fact, "qkdx0xe2", "*", "*"):
                pred, obj, pos = get_parts(fact)
                obj_pos[obj] = pos

        # Which pt4b08k4 objects currently are associated with which du770mtw objects via (fqyrzl30 ?pt4b08k4 ?du770mtw)
        fqyrzl30_map = {}
        for fact in state:
            if match(fact, "fqyrzl30", "*", "*"):
                pred, pt_obj, du_obj = get_parts(fact)
                fqyrzl30_map[pt_obj] = du_obj

        # Which du770mtw object currently has (ndg963qz)
        ndg_obj = None
        for fact in state:
            if match(fact, "ndg963qz", "*"):
                ndg_obj = get_parts(fact)[1]
                break

        # Identify which wqid8mog objects are already (ne6sl557)
        ne6sl557_objs = set()
        for fact in state:
            if match(fact, "ne6sl557", "*"):
                ne6sl557_objs.add(get_parts(fact)[1])

        # Which wqid8mog objects currently have (of6wn8m4)
        of6wn8m4_objs = set()
        for fact in state:
            if match(fact, "of6wn8m4", "*"):
                of6wn8m4_objs.add(get_parts(fact)[1])

        # For each goal wqid8mog object, estimate cost to get to (ne6sl557)
        total_cost = 0

        # Cost weights (tuned heuristically)
        COST_MOVE_CHAIN = 1     # cost per nm6lzp2x action to move pt4b08k4 object along chain
        COST_MOVE_FQYR = 2      # cost per jql8txb3 action to move (fqyrzl30 ?) between du770mtw objects
        COST_F7Y6NZQ3 = 3       # cost to apply f7y6nzq3 action once (to achieve ne6sl557 for one wqid8mgo)
        COST_CHANGE_NDG = 2     # cost to setup/restore ndg963qz on needed du770mtw object

        for wobj in self.wqid8mog_objects:
            if wobj in ne6sl557_objs:
                # Already done: no cost
                continue

            # Must have (of6wn8m4) to be processed, else unreachable goal state => assign high cost
            if wobj not in of6wn8m4_objs:
                # No (of6wn8m4) means cannot apply f7y6nzq3: large penalty
                total_cost += 50
                continue

            # Positions:
            # The wqid8mog object's mz6r1d30 position (ts6t4) used in qkdx0xe2(wobj, ts6t4)
            wobj_pos = obj_pos.get(wobj, None)
            if wobj_pos is None:
                # Not located on a chain node: large penalty
                total_cost += 50
                continue

            # Find the pt4b08k4 object (n54xx) and du770mtw (cz2xr) that will be used to process this wobj:
            # The f7y6nzq3 precondition requires:
            #   - (ndg963qz ?cz2xr)
            #   - (fqyrzl30 ?n54xx ?cz2xr)
            #   - (qkdx0xe2 ?n54xx ?ts6t4)
            #   - (qkdx0xe2 ?ris62 ?ts6t4)

            # We need to find a pt4b08k4 object whose (fqyrzl30 ?n54xx ?cz2xr) matches the ndg du770mtw,
            # and whose qkdx0xe2 position matches wobj_pos

            # If ndg_obj is None, then no du770mtw with ndg963qz currently => need to account cost to set one (assume 1 action)
            ndg_cost = COST_CHANGE_NDG if ndg_obj is None else 0

            best_obj_cost = 1000  # start large, find minimal cost pt4b08k4 candidate

            # Search candidates for pt4b08k4 objects
            for pt_obj in self.pt4b08k4_objects:
                # Positions:
                # pt_obj position on chain node
                pt_pos = obj_pos.get(pt_obj, None)
                if pt_pos is None:
                    continue
                # pt_obj fqyrzl30 du770mtw
                pt_du = fqyrzl30_map.get(pt_obj, None)
                if pt_du is None:
                    continue

                # Distance along chain for pt_obj to wobj: must have same chain position (qkdx0xe2 ?n54xx ?ts6t4 and ?ris62 ?ts6t4)
                # If pt_pos != wobj_pos, must move pt_obj along chain by nm6lzp2x actions
                dist_chain = self._chain_distance(pt_pos, wobj_pos)

                # If pt_obj is at different du770mtw ?cz2xr than the ndg_obj, must move (fqyrzl30) via jql8txb3 actions
                fqyrzl30_move_cost = 0
                if ndg_obj is not None and pt_du != ndg_obj:
                    fqyrzl30_move_cost = COST_MOVE_FQYR

                cost = dist_chain * COST_MOVE_CHAIN + fqyrzl30_move_cost

                if cost < best_obj_cost:
                    best_obj_cost = cost

            # If no pt4b08k4 candidate configurations found, assign a large penalty
            if best_obj_cost == 1000:
                best_obj_cost = 50

            # Additionally, if ndg_obj is None (no available ndg963qz), add cost for changing it (for simplicity, count once per wobj)
            cost_for_wobj = best_obj_cost + ndg_cost + COST_F7Y6NZQ3
            total_cost += cost_for_wobj

        # Heuristic estimate should be finite for solvable states and 0 for goals
        # Return the computed cost as heuristic value
        return total_cost
