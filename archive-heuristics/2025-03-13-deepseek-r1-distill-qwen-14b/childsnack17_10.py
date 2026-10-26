import re

from heuristics.heuristic_base import Heuristic


class ChildsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children by calculating the required steps for each child based on their current state.

    # Assumptions:
    - Each child must be served exactly one sandwich.
    - Allergic children require a gluten-free sandwich.
    - Sandwiches are made from available ingredients in the kitchen.
    - Each sandwich must be placed on a tray before being served.

    # Heuristic Initialization
    - Extracts static facts about allergic children, their waiting places, and the gluten-free status of ingredients.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each child, check if they are already served. If yes, no actions needed.
    2. For each unserved child:
       a. Add 4 actions (make, put on tray, move, serve).
       b. If the tray is already at the child's waiting place, subtract 1 action.
    3. Sum the actions for all children to get the total heuristic value.
    """

    def __init__(self, task):
        super().__init__(task)
        self.allergic_children = set()
        self.waiting_places = dict()
        self.children = set()
        # Extract static facts
        for fact in task.static:
            if fact.startswith('(allergic_gluten '):
                c = self._get_arg(fact, 1)
                self.allergic_children.add(c)
                self.children.add(c)
            elif fact.startswith('(not_allergic_gluten '):
                c = self._get_arg(fact, 1)
                self.children.add(c)
            elif fact.startswith('(waiting '):
                c, p = self._get_args(fact, 2)
                self.waiting_places[c] = p

    def __call__(self, node):
        state = node.state
        total = 0
        for c in self.children:
            if f'(served {c})' in state:
                continue
            # Base actions: make, put on tray, move, serve (4 actions)
            actions = 4
            # Check if the tray is already at the child's place
            tray_at_place = False
            for fact in state:
                if fact.startswith('(at tray') and self._get_arg(fact, -1) == self.waiting_places[c]:
                    tray_at_place = True
                    break
            if tray_at_place:
                actions -= 1
            total += actions
        return total

    def _get_arg(self, fact, index):
        return fact[1:-1].split()[index]

    def _get_args(self, fact, num_args):
        return fact[1:-1].split()[:num_args]
