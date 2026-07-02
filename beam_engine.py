import sympy as sp
from sympy.physics.continuum_mechanics.beam import Beam

x = sp.Symbol('x')


def resoudre(L, appuis, charges, E=1, I=1):
    """
    appuis: liste de (position, type) avec type in {"rotule","simple","encastrement"}
    charges: liste de dicts:
        {"type": "ponctuelle", "position": p, "valeur": v}      (v>0 vers le bas)
        {"type": "repartie", "debut": d, "fin": f, "valeur": v} (v peut dépendre de x, v>0 vers le bas)
        {"type": "moment", "position": p, "valeur": v}

    Les réactions sont calculées par équilibre (somme des forces et des moments) résolu
    avec sp.solve — plus fiable que Beam.solve_for_reaction_loads qui échoue silencieusement
    dès qu'une charge n'a pas une position strictement égale à 0.
    """
    xi = sp.Symbol('xi_int')  # variable d'intégration dédiée (évite les conflits avec x)

    if len(appuis) == 1 and appuis[0][1] == "encastrement":
        # Poutre console : équilibre direct, pas besoin de résoudre un système
        pos_enc = appuis[0][0]
        somme_Fy = 0
        somme_M = 0  # moment par rapport à l'encastrement
        for c in charges:
            if c["type"] == "ponctuelle":
                somme_Fy += c["valeur"]
                somme_M += c["valeur"] * (c["position"] - pos_enc)
            elif c["type"] == "repartie":
                W = sp.integrate(c["valeur"].subs(x, xi) if hasattr(c["valeur"], "subs") else c["valeur"],
                                  (xi, c["debut"], c["fin"]))
                x_bar = sp.integrate((c["valeur"].subs(x, xi) if hasattr(c["valeur"], "subs") else c["valeur"]) * xi,
                                      (xi, c["debut"], c["fin"])) / W if W != 0 else (c["debut"] + c["fin"]) / 2
                somme_Fy += W
                somme_M += W * (x_bar - pos_enc)
            elif c["type"] == "moment":
                somme_M += c["valeur"]
        Ry = sp.simplify(-somme_Fy)
        M_enc = sp.simplify(-somme_M)
        reactions_valeurs = {sp.Symbol("R_" + str(pos_enc)): Ry, sp.Symbol("M_" + str(pos_enc)): M_enc}
        appui_positions = [(pos_enc, Ry, "force"), (pos_enc, M_enc, "moment")]
    else:
        assert len(appuis) == 2, "Le moteur gère : 2 appuis simples/rotules, OU 1 encastrement seul"
        xA, xB = appuis[0][0], appuis[1][0]
        RA, RB = sp.symbols('RA_ RB_')

        somme_Fy = 0
        somme_M_A = 0
        for c in charges:
            if c["type"] == "ponctuelle":
                somme_Fy += c["valeur"]
                somme_M_A += c["valeur"] * (c["position"] - xA)
            elif c["type"] == "repartie":
                v = c["valeur"]
                v_xi = v.subs(x, xi) if hasattr(v, "subs") else v
                W = sp.integrate(v_xi, (xi, c["debut"], c["fin"]))
                if W != 0:
                    x_bar = sp.integrate(v_xi * xi, (xi, c["debut"], c["fin"])) / W
                else:
                    x_bar = (c["debut"] + c["fin"]) / 2
                somme_Fy += W
                somme_M_A += W * (x_bar - xA)
            elif c["type"] == "moment":
                somme_M_A += c["valeur"]

        eq1 = sp.Eq(RA + RB - somme_Fy, 0)
        eq2 = sp.Eq(RB * (xB - xA) - somme_M_A, 0)
        sol = sp.solve([eq1, eq2], [RA, RB])
        RA_val = sp.simplify(sol[RA])
        RB_val = sp.simplify(sol[RB])
        reactions_valeurs = {sp.Symbol("R_" + str(xA)): RA_val, sp.Symbol("R_" + str(xB)): RB_val}
        appui_positions = [(xA, RA_val, "force"), (xB, RB_val, "force")]

    # --- Construction de V(x), M(x) via les fonctions de singularité de sympy,
    #     avec les réactions déjà connues (on n'appelle PAS Beam.solve_for_reaction_loads) ---
    L_beam = L
    b = Beam(L_beam, E, I)
    for pos, val, type_ in appui_positions:
        # convention empirique de sympy : il faut négliger le val "physique" (positif=vers le haut)
        # et le donner négatif pour que V(x)/M(x) sortent avec le bon signe (M>0 = fibre inf. tendue)
        if type_ == "force":
            b.apply_load(-val, pos, -1)
        else:
            b.apply_load(-val, pos, -2)

    for c in charges:
        if c["type"] == "ponctuelle":
            b.apply_load(c["valeur"], c["position"], -1)
        elif c["type"] == "repartie":
            b.apply_load(c["valeur"], c["debut"], 0, end=c["fin"])
        elif c["type"] == "moment":
            b.apply_load(c["valeur"], c["position"], -2)

    V = sp.simplify(b.shear_force())
    M = sp.simplify(b.bending_moment())

    return {"beam": b, "reactions": reactions_valeurs, "V": V, "M": M}


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
