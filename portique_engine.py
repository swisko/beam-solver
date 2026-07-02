import sympy as sp
import numpy as np


# ============================================================
# Helpers communs (symboles de réaction, moment d'une force, etc.)
# ============================================================

def _symboles_appui(sup, suffixe=""):
    """Crée les symboles d'inconnues pour un appui, et retourne (liste_inconnues, info_pour_reconstruction)."""
    idx = sup["index"]
    if sup["type"] == "rotule":
        Rx, Ry = sp.symbols(f'Rx{idx}{suffixe} Ry{idx}{suffixe}')
        return [Rx, Ry], ("rotule", (Rx, Ry))
    elif sup["type"] == "simple":
        R = sp.Symbol(f'R{idx}{suffixe}')
        angle = sp.rad(sup.get("angle", 0))
        Rx, Ry = R * sp.sin(angle), R * sp.cos(angle)
        return [R], ("simple", (Rx, Ry))
    elif sup["type"] == "encastrement":
        Rx, Ry, M = sp.symbols(f'Rx{idx}{suffixe} Ry{idx}{suffixe} M{idx}{suffixe}')
        return [Rx, Ry, M], ("encastrement", (Rx, Ry, M))
    else:
        raise ValueError(f"Type d'appui inconnu: {sup['type']}")


def _moment_de(Fx, Fy, point_force, point_ref):
    """Moment de la force (Fx,Fy) appliquée en point_force, calculé par rapport à point_ref."""
    dx = point_force[0] - point_ref[0]
    dy = point_force[1] - point_ref[1]
    return dx * Fy - dy * Fx


def _somme_efforts(waypoints, reactions_def, charges, point_ref):
    """Somme Fx, Fy, M (par rapport à point_ref) de tous les appuis + charges donnés."""
    Fx_tot, Fy_tot, M_tot = 0, 0, 0
    for sup, (type_, syms) in reactions_def:
        pos = waypoints[sup["index"]]
        if type_ == "encastrement":
            Rx, Ry, M = syms
            M_tot += M
        else:
            Rx, Ry = syms
        Fx_tot += Rx
        Fy_tot += Ry
        M_tot += _moment_de(Rx, Ry, pos, point_ref)

    for c in charges:
        if c["type"] == "ponctuelle":
            Fx_tot += c["Fx"]
            Fy_tot += c["Fy"]
            M_tot += _moment_de(c["Fx"], c["Fy"], c["position"], point_ref)
        elif c["type"] == "repartie":
            p0 = np.array(waypoints[c["index_debut"]], dtype=float)
            p1 = np.array(waypoints[c["index_fin"]], dtype=float)
            L = np.linalg.norm(p1 - p0)
            Fx_c, Fy_c = c["qx"] * L, c["qy"] * L
            centre = tuple((p0 + p1) / 2)
            Fx_tot += Fx_c
            Fy_tot += Fy_c
            M_tot += _moment_de(Fx_c, Fy_c, centre, point_ref)
        elif c["type"] == "moment":
            M_tot += c["valeur"]

    return Fx_tot, Fy_tot, M_tot


def _avant(waypoints, i_troncon, s_val, position):
    """Est-ce que `position` se situe avant (ou sur) le point de coupe (tronçon i_troncon, abscisse s_val) ?"""
    px, py = position
    for k in range(len(waypoints) - 1):
        a = np.array(waypoints[k], dtype=float)
        b = np.array(waypoints[k + 1], dtype=float)
        Lk = np.linalg.norm(b - a)
        if Lk == 0:
            continue
        t = np.dot(np.array([px, py]) - a, (b - a) / Lk)
        proj = a + (b - a) / Lk * t
        if np.linalg.norm(proj - np.array([px, py])) < 1e-6 and -1e-6 <= t <= Lk + 1e-6:
            if k < i_troncon:
                return True
            elif k == i_troncon:
                return t <= s_val + 1e-9
            return False
    return False


def _construire_segments(waypoints, reactions_numeriques, charges):
    """
    reactions_numeriques: dict {index_waypoint: (Rx,Ry)} ou {index_waypoint: (Rx,Ry,M)} — valeurs NUMÉRIQUES connues.
    charges: charges propres à ce fragment.
    Retourne la liste des segments avec leur fonction N(s),V(s),M(s).
    """
    n = len(waypoints)
    segments = []

    for i in range(n - 1):
        p0 = np.array(waypoints[i], dtype=float)
        p1 = np.array(waypoints[i + 1], dtype=float)
        L = np.linalg.norm(p1 - p0)
        direction = (p1 - p0) / L
        normale = np.array([-direction[1], direction[0]])

        def N_V_M(s_val, i=i, p0=p0, direction=direction, normale=normale):
            point_coupe = p0 + direction * s_val
            Fx_cum, Fy_cum, M_cum = 0.0, 0.0, 0.0

            for idx, vals in reactions_numeriques.items():
                if idx <= i:
                    pos = waypoints[idx]
                    Rx, Ry = float(vals[0]), float(vals[1])
                    Fx_cum += Rx
                    Fy_cum += Ry
                    M_cum += _moment_de(Rx, Ry, pos, point_coupe)
                    if len(vals) == 3:
                        M_cum += float(vals[2])

            for c in charges:
                if c["type"] == "ponctuelle":
                    if _avant(waypoints, i, s_val, c["position"]):
                        Fx_cum += c["Fx"]
                        Fy_cum += c["Fy"]
                        M_cum += _moment_de(c["Fx"], c["Fy"], c["position"], point_coupe)
                elif c["type"] == "repartie":
                    if c["index_fin"] <= i or (c["index_debut"] <= i < c["index_fin"]):
                        pa = np.array(waypoints[c["index_debut"]], dtype=float)
                        pb_idx = min(c["index_fin"], i + 1)
                        pb = np.array(waypoints[pb_idx], dtype=float)
                        if pb_idx == i + 1 and c["index_fin"] > i:
                            pb = p0 + direction * s_val
                        Lc = np.linalg.norm(pb - pa)
                        Fx_c, Fy_c = c["qx"] * Lc, c["qy"] * Lc
                        centre = tuple((pa + pb) / 2)
                        Fx_cum += Fx_c
                        Fy_cum += Fy_c
                        M_cum += _moment_de(Fx_c, Fy_c, centre, point_coupe)
                elif c["type"] == "moment":
                    if _avant(waypoints, i, s_val, c["position"]):
                        M_cum += c["valeur"]

            N = -(Fx_cum * direction[0] + Fy_cum * direction[1])
            V = -(Fx_cum * normale[0] + Fy_cum * normale[1])
            M = M_cum
            return N, V, M

        segments.append({"p0": tuple(p0), "p1": tuple(p1), "L": L,
                          "direction": tuple(direction), "fonction": N_V_M})

    return segments


# ============================================================
# Cas simple : un seul corps rigide (pas de rotule interne)
# ============================================================

def resoudre_portique(waypoints, supports, charges):
    """
    waypoints: [(x0,y0), ..., (xn,yn)] — polyligne définissant la géométrie.
    supports: [{"index":i, "type":"rotule"/"simple"/"encastrement", "angle":a}, ...]
    charges: [{"type":"ponctuelle","position":(x,y),"Fx":..,"Fy":..},
              {"type":"repartie","index_debut":i,"index_fin":j,"qx":..,"qy":..},
              {"type":"moment","position":(x,y),"valeur":..}]
    """
    inconnues = []
    reactions_def = []
    for sup in supports:
        syms, info = _symboles_appui(sup)
        inconnues += syms
        reactions_def.append((sup, info))

    Fx_tot, Fy_tot, M_tot = _somme_efforts(waypoints, reactions_def, charges, (0, 0))
    sol = sp.solve([sp.Eq(Fx_tot, 0), sp.Eq(Fy_tot, 0), sp.Eq(M_tot, 0)], inconnues)
    if not sol:
        raise RuntimeError("Système non résolu — vérifie le nombre d'appuis (3 inconnues nécessaires).")

    reactions = {}
    reactions_numeriques = {}
    for sup, (type_, syms) in reactions_def:
        if type_ == "encastrement":
            Rx, Ry, M = syms
            vals = (float(sol[Rx]), float(sol[Ry]), float(sol[M]))
        else:
            Rx, Ry = syms
            Rx_val = float(Rx.subs(sol)) if hasattr(Rx, "subs") else float(sol.get(Rx, Rx))
            Ry_val = float(Ry.subs(sol)) if hasattr(Ry, "subs") else float(sol.get(Ry, Ry))
            vals = (Rx_val, Ry_val)
        reactions[f'appui_{sup["index"]}'] = vals
        reactions_numeriques[sup["index"]] = vals

    segments = _construire_segments(waypoints, reactions_numeriques, charges)
    return {"reactions": reactions, "segments": segments}


# ============================================================
# Cas multi-fragments : rotules internes (portiques à 3 articulations, etc.)
# ============================================================

def resoudre_portique_multi(fragments, hinges):
    """
    fragments: [{"waypoints":[...], "supports":[...], "charges":[...]}, ...]  (SANS rotules internes)
    hinges: [{"position":(x,y), "connexions":[(frag_index, waypoint_index), (frag_index2, waypoint_index2)]}, ...]

    Méthode (celle du corrigé EPFL) :
    1. Équilibre GLOBAL (somme de tous les fragments) → 3 équations.
    2. Pour chaque rotule : le moment de tout ce qui appartient à UN SEUL fragment
       (ses propres appuis + ses propres charges) par rapport à la rotule doit être nul
       (une rotule ne transmet aucun moment) → 1 équation par rotule.
    3. On résout le système complet (toutes les réactions de tous les fragments d'un coup).
    """
    inconnues = []
    reactions_def_par_fragment = []  # liste (parallèle à fragments) de reactions_def
    for frag in fragments:
        reactions_def = []
        for sup in frag["supports"]:
            syms, info = _symboles_appui(sup, suffixe=f'_f{len(reactions_def_par_fragment)}')
            inconnues += syms
            reactions_def.append((sup, info))
        reactions_def_par_fragment.append(reactions_def)

    equations = []

    # Équilibre global (tous fragments confondus)
    Fx_tot, Fy_tot, M_tot = 0, 0, 0
    for i, frag in enumerate(fragments):
        fx, fy, m = _somme_efforts(frag["waypoints"], reactions_def_par_fragment[i], frag["charges"], (0, 0))
        Fx_tot += fx
        Fy_tot += fy
        M_tot += m
    equations += [sp.Eq(Fx_tot, 0), sp.Eq(Fy_tot, 0), sp.Eq(M_tot, 0)]

    # Une équation par rotule : moment nul par rapport à la rotule, côté d'UN SEUL fragment
    for h in hinges:
        frag_idx, wp_idx = h["connexions"][0]  # on prend le premier fragment connecté (arbitraire)
        frag = fragments[frag_idx]
        _, _, m = _somme_efforts(frag["waypoints"], reactions_def_par_fragment[frag_idx],
                                  frag["charges"], h["position"])
        equations.append(sp.Eq(m, 0))

    sol = sp.solve(equations, inconnues)
    if not sol:
        raise RuntimeError("Système non résolu — vérifie appuis/rotules (isostaticité).")

    # Extraction des réactions numériques par fragment
    resultats = []
    for i, frag in enumerate(fragments):
        reactions = {}
        reactions_numeriques = {}
        for sup, (type_, syms) in reactions_def_par_fragment[i]:
            if type_ == "encastrement":
                Rx, Ry, M = syms
                vals = (float(sol[Rx]), float(sol[Ry]), float(sol[M]))
            else:
                Rx, Ry = syms
                Rx_val = float(Rx.subs(sol)) if hasattr(Rx, "subs") else float(sol.get(Rx, Rx))
                Ry_val = float(Ry.subs(sol)) if hasattr(Ry, "subs") else float(sol.get(Ry, Ry))
                vals = (Rx_val, Ry_val)
            reactions[f'appui_{sup["index"]}'] = vals
            reactions_numeriques[sup["index"]] = vals

        segments = _construire_segments(frag["waypoints"], reactions_numeriques, frag["charges"])
        resultats.append({"reactions": reactions, "segments": segments})

    return {"fragments": resultats}
