import sympy as sp
from sympy.physics.continuum_mechanics.beam import Beam

x = sp.Symbol('x')


def resoudre(L, appuis, charges, E=1, I=1, moment_sign_flip=True):
    """
    appuis: liste de (position, type) avec type in {"rotule","simple","encastrement"}
            "rotule"/"simple" -> pin/roller (1 réaction verticale)
    charges: liste de dicts:
        {"type": "ponctuelle", "position": p, "valeur": v}      (v>0 vers le bas)
        {"type": "repartie", "debut": d, "fin": f, "valeur": v} (v peut dépendre de x, v>0 vers le bas)
        {"type": "moment", "position": p, "valeur": v}
    """
    b = Beam(L, E, I)
    reactions = []
    for pos, type_ in appuis:
        sympy_type = "pin" if type_ in ("rotule", "simple") else "fixed"
        r = b.apply_support(pos, type=sympy_type)
        if isinstance(r, tuple):
            reactions.extend(r)
        else:
            reactions.append(r)

    for c in charges:
        if c["type"] == "ponctuelle":
            b.apply_load(c["valeur"], c["position"], -1)
        elif c["type"] == "repartie":
            b.apply_load(c["valeur"], c["debut"], 0, end=c["fin"])
        elif c["type"] == "moment":
            b.apply_load(c["valeur"], c["position"], -2)

    b.solve_for_reaction_loads(*reactions)

    V = b.shear_force()
    M = b.bending_moment()
    # sympy's Beam sign convention for M is opposite to the EPFL convention used in the corrigés
    # (EPFL: M>0 => fibre inférieure tendue). On garde les deux dispos.
    return {"beam": b, "reactions": b.reaction_loads, "V": sp.simplify(V), "M": sp.simplify(M)}


def evaluer(expr, subs):
    """Evalue une expression Piecewise/Singularity en un point numérique."""
    return sp.nsimplify(expr.subs(subs).rewrite(sp.Piecewise).doit())


if __name__ == "__main__":
    Ls, Ps, qs, k1 = sp.symbols('L P q k1', positive=True)

    print("========== TEST A : poutre simple + P au milieu ==========")
    res = resoudre(Ls, [(0, "rotule"), (Ls, "simple")],
                    [{"type": "ponctuelle", "position": Ls/2, "valeur": Ps}])
    print("Réactions :", res["reactions"])
    print("Attendu : |Ay|=|By|=P/2\n")

    print("========== TEST B : porte-à-faux, exercice 4a (k1 = sqrt(2)-1) ==========")
    L_beam_totale = Ls * (1 + k1)  # longueur TOTALE de la poutre (travée + porte-à-faux)
    charges = [{"type": "repartie", "debut": 0, "fin": L_beam_totale, "valeur": qs}]
    res = resoudre(L_beam_totale, [(0, "rotule"), (Ls, "simple")], charges)
    print("Réactions :", res["reactions"])
    Ay = [v for k, v in res["reactions"].items() if "R_0" in str(k)][0]
    By = [v for k, v in res["reactions"].items() if str(k) != "R_0" and "R_" in str(k)][0]
    print("Ay =", sp.simplify(Ay), " attendu q*L/2*(1+k1)*(1-k1) =", sp.simplify(qs*Ls/2*(1+k1)*(1-k1)))
    print("By =", sp.simplify(By), " attendu q*L/2*(1+k1)*(1+k1) =", sp.simplify(qs*Ls/2*(1+k1)*(1+k1)))
