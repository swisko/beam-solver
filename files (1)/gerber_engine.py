import sympy as sp
from beam_engine import resoudre

x = sp.Symbol('x')


def resoudre_gerber(segments, hinges_positions):
    """
    segments: liste de dicts par tronçon, dans l'ordre gauche->droite :
        {"x_debut":..., "x_fin":..., "appuis":[(position,type),...], "charges":[...]}
    hinges_positions: positions x des rotules internes séparant les segments
                       (len(segments) == len(hinges_positions) + 1)

    Méthode : une rotule non résolue est traitée comme un appui simple TEMPORAIRE
    pour permettre de résoudre le segment (le moteur d'équilibre donne alors directement
    la force réellement transmise à la rotule). Une rotule déjà résolue devient une
    charge ponctuelle connue pour le(s) segment(s) voisin(s). On répète jusqu'à ce que
    tout soit résolu (chaque segment ayant au max 2 appuis réels + rotules temporaires).
    """
    n = len(segments)
    assert len(hinges_positions) == n - 1

    forces_rotules = [None] * len(hinges_positions)
    resolus = [False] * n
    resultats = [None] * n

    def appuis_reels(i):
        return segments[i]["appuis"]

    def nb_inconnues(i):
        nb = sum(2 if t == "encastrement" else 1 for _, t in appuis_reels(i))
        if i > 0 and forces_rotules[i - 1] is None:
            nb += 1
        if i < n - 1 and forces_rotules[i] is None:
            nb += 1
        return nb

    iterations = 0
    while not all(resolus) and iterations < n + 3:
        iterations += 1
        progres = False
        for i in range(n):
            if resolus[i] or nb_inconnues(i) > 2:
                continue

            seg = segments[i]
            appuis = list(seg["appuis"])
            charges = list(seg["charges"])

            hinge_gauche_temp = (i > 0 and forces_rotules[i - 1] is None)
            hinge_droite_temp = (i < n - 1 and forces_rotules[i] is None)

            if hinge_gauche_temp:
                appuis.append((hinges_positions[i - 1], "simple"))
            elif i > 0:
                charges.append({"type": "ponctuelle", "position": hinges_positions[i - 1],
                                 "valeur": forces_rotules[i - 1]})

            if hinge_droite_temp:
                appuis.append((hinges_positions[i], "simple"))
            elif i < n - 1:
                charges.append({"type": "ponctuelle", "position": hinges_positions[i],
                                 "valeur": forces_rotules[i]})

            if len(appuis) == 1 and appuis[0][1] == "encastrement":
                res = resoudre(seg["x_fin"] - seg["x_debut"], appuis, charges)
            elif len(appuis) == 2:
                res = resoudre(seg["x_fin"], appuis, charges)
            else:
                raise ValueError(f"Segment {i}: {len(appuis)} appui(s) effectif(s), configuration non gérée "
                                  f"(il faut exactement 2 appuis/rotules, ou 1 encastrement)")

            resultats[i] = res
            resolus[i] = True
            progres = True

            if hinge_gauche_temp:
                key = [k for k in res["reactions"] if str(hinges_positions[i - 1]) in str(k)][0]
                forces_rotules[i - 1] = res["reactions"][key]
            if hinge_droite_temp:
                key = [k for k in res["reactions"] if str(hinges_positions[i]) in str(k)][0]
                forces_rotules[i] = res["reactions"][key]

        if not progres:
            break

    if not all(resolus):
        raise RuntimeError("Poutre Gerber non résolue : vérifie le nombre d'appuis/rotules "
                            "(probablement hyperstatique ou mécanisme).")

    return {"segments": resultats, "forces_rotules": forces_rotules, "hinges": hinges_positions}
