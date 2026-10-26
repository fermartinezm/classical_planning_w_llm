(define (domain r9oudocu)

(:requirements :strips)

(:predicates (dhjk1eyy ?x)
             (xk87au5b ?x)
             (hxtpdpff)
             (q71n8mtz ?x)
             (x272hpoe ?x ?y))

(:action oecve6pr
  :parameters (?ob)
  :precondition (and (dhjk1eyy ?ob) (xk87au5b ?ob) (hxtpdpff))
  :effect (and (q71n8mtz ?ob) (not (dhjk1eyy ?ob)) (not (xk87au5b ?ob))
               (not (hxtpdpff))))

(:action dgdzvgpm
  :parameters  (?ob)
  :precondition (q71n8mtz ?ob)
  :effect (and (dhjk1eyy ?ob) (hxtpdpff) (xk87au5b ?ob)
               (not (q71n8mtz ?ob))))

(:action m82i1lr3
  :parameters  (?ob ?ob2)
  :precondition (and (dhjk1eyy ?ob2) (q71n8mtz ?ob))
  :effect (and (hxtpdpff) (dhjk1eyy ?ob) (x272hpoe ?ob ?ob2)
               (not (dhjk1eyy ?ob2)) (not (q71n8mtz ?ob))))

(:action fpk054nz
  :parameters  (?ob ?ob2)
  :precondition (and (x272hpoe ?ob ?ob2) (dhjk1eyy ?ob) (hxtpdpff))
  :effect (and (q71n8mtz ?ob) (dhjk1eyy ?ob2)
               (not (x272hpoe ?ob ?ob2)) (not (dhjk1eyy ?ob)) (not (hxtpdpff)))))
