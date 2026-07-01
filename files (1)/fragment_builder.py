import numpy as np


def construire_fragments(noeuds, elements, hinges_ids, supports, charges):
    """
    noeuds: dict {id: (x,y)}
    elements: liste de (id_element, noeud_a, noeud_b) — chaque élément est un tronçon droit
    hinges_ids: set des ids de nœuds qui sont des rotules internes (ne transmettent pas de moment)
    supports: liste de dicts {"noeud": id, "type":"rotule"/"simple"/"encastrement", "angle":a}
    charges: liste de dicts, chacune référence soit un nœud (ponctuelle/moment) soit un élément (répartie)
             {"type":"ponctuelle","noeud":id,"Fx":..,"Fy":..}
             {"type":"moment","noeud":id,"valeur":..}
             {"type":"repartie","element":id,"qx":..,"qy":..}

    Regroupe les éléments en fragments rigides (Union-Find), séparés aux nœuds-rotules.
    Vérifie que chaque fragment forme une chaîne simple (polyligne), pas un embranchement
    (notre moteur ne gère pas les nœuds en T à l'intérieur d'un même fragment rigide).

    Retourne : (fragments, hinges) au format attendu par portique_engine.resoudre_portique_multi
    """
    # --- Union-Find pour regrouper les éléments connectés de façon rigide ---
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for eid, a, b in elements:
        parent.setdefault(eid, eid)

    # deux éléments partageant un nœud NON-rotule sont fusionnés dans le même fragment
    noeud_vers_elements = {}
    for eid, a, b in elements:
        noeud_vers_elements.setdefault(a, []).append(eid)
        noeud_vers_elements.setdefault(b, []).append(eid)

    for noeud, eids in noeud_vers_elements.items():
        if noeud not in hinges_ids and len(eids) > 1:
            for k in range(1, len(eids)):
                union(eids[0], eids[k])

    groupes = {}
    for eid, a, b in elements:
        racine = find(eid)
        groupes.setdefault(racine, []).append((eid, a, b))

    # --- Pour chaque groupe, vérifier que c'est une chaîne simple et ordonner les waypoints ---
    fragments = []
    element_vers_fragment_index = {}

    for racine, elems in groupes.items():
        degre = {}
        for eid, a, b in elems:
            degre[a] = degre.get(a, 0) + 1
            degre[b] = degre.get(b, 0) + 1
        extremites = [n for n, d in degre.items() if d == 1]
        if len(elems) == 1:
            extremites = [elems[0][1], elems[0][2]]
        if any(d > 2 for d in degre.values()):
            raise ValueError(
                f"Structure non supportée : un nœud a plus de 2 éléments rigidement connectés "
                f"(embranchement en T). Le solveur actuel ne gère que des chaînes simples "
                f"(polylignes) entre rotules. Ajoute une rotule à ce nœud pour séparer les branches."
            )
        if len(extremites) not in (1, 2):
            raise ValueError("Structure non supportée : fragment non linéaire détecté.")

        depart = extremites[0]
        chemin_noeuds = [depart]
        chemin_elements = []
        elems_restants = list(elems)
        courant = depart
        while elems_restants:
            trouve = False
            for k, (eid, a, b) in enumerate(elems_restants):
                if a == courant:
                    chemin_noeuds.append(b)
                    chemin_elements.append(eid)
                    courant = b
                    elems_restants.pop(k)
                    trouve = True
                    break
                elif b == courant:
                    chemin_noeuds.append(a)
                    chemin_elements.append(eid)
                    courant = a
                    elems_restants.pop(k)
                    trouve = True
                    break
            if not trouve:
                raise ValueError("Structure non supportée : fragment discontinu.")

        waypoints = [noeuds[n] for n in chemin_noeuds]

        frag_supports = []
        for sup in supports:
            if sup["noeud"] in chemin_noeuds:
                idx = chemin_noeuds.index(sup["noeud"])
                frag_supports.append({"index": idx, "type": sup["type"], "angle": sup.get("angle", 0)})

        frag_charges = []
        for c in charges:
            if c["type"] in ("ponctuelle", "moment") and c.get("noeud") in chemin_noeuds:
                idx = chemin_noeuds.index(c["noeud"])
                pos = waypoints[idx]
                if c["type"] == "ponctuelle":
                    frag_charges.append({"type": "ponctuelle", "position": pos, "Fx": c["Fx"], "Fy": c["Fy"]})
                else:
                    frag_charges.append({"type": "moment", "position": pos, "valeur": c["valeur"]})
            elif c["type"] == "repartie" and c.get("element") in chemin_elements:
                k = chemin_elements.index(c["element"])
                frag_charges.append({"type": "repartie", "index_debut": k, "index_fin": k + 1,
                                      "qx": c["qx"], "qy": c["qy"]})

        fragment_index = len(fragments)
        fragments.append({"waypoints": waypoints, "supports": frag_supports, "charges": frag_charges,
                           "_chemin_noeuds": chemin_noeuds})
        for eid in chemin_elements:
            element_vers_fragment_index[eid] = fragment_index

    # --- Construction des rotules (connexions entre fragments) ---
    hinges = []
    for h_noeud in hinges_ids:
        connexions = []
        for i, frag in enumerate(fragments):
            if h_noeud in frag["_chemin_noeuds"]:
                idx = frag["_chemin_noeuds"].index(h_noeud)
                connexions.append((i, idx))
        if len(connexions) >= 2:
            hinges.append({"position": noeuds[h_noeud], "connexions": connexions[:2]})
        # si connexions == 1 : rotule "morte" (extrémité libre avec rotule, ne change rien)

    for frag in fragments:
        del frag["_chemin_noeuds"]

    return fragments, hinges
