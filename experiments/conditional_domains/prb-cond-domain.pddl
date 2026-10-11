(define (domain shape-stacking)
    (:requirements :strips :typing :conditional-effects :equality :negative-preconditions :universal-preconditions)
    (:types
        block sphere - shape
    )
    (:predicates
        (is-directly-on-table ?s - shape)
        (is-clear-on-top ?s - shape)
        (shape-on-top-of-block ?s - shape ?b - block)
    )

    ;; Source-agnostic safe move to table.  The physical safety condition is
    ;; that the moved shape is clear.  The previous support is not a safety
    ;; precondition; it only determines which support becomes clear, so it is
    ;; handled by conditional effects over all possible old blocks.
    (:action move-to-table
        :parameters (?s - shape)
        :precondition (is-clear-on-top ?s)
        :effect (and
            (is-directly-on-table ?s)
            (forall (?old - block)
                (when (shape-on-top-of-block ?s ?old)
                    (and
                        (not (shape-on-top-of-block ?s ?old))
                        (is-clear-on-top ?old))))))

    ;; Source-agnostic safe move to a destination block.  Destination occupancy
    ;; remains a real safety precondition: moving onto a non-clear block is not
    ;; safe.  The old support is cleaned up conditionally, so the action works
    ;; even when the belief is uncertain about whether ?s started on the table or
    ;; on some block.
    (:action move-to-block
        :parameters (?s - shape ?b - block)
        :precondition (and
            (is-clear-on-top ?s)
            (is-clear-on-top ?b)
            (not (= ?s ?b)))
        :effect (and
            (not (is-directly-on-table ?s))
            (not (is-clear-on-top ?b))
            (shape-on-top-of-block ?s ?b)
            (forall (?old - block)
                (when (shape-on-top-of-block ?s ?old)
                    (and
                        (not (shape-on-top-of-block ?s ?old))
                        (is-clear-on-top ?old))))))
)
