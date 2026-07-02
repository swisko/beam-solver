import streamlit as st
import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from beam_engine import resoudre, evaluer
from fragment_builder import construire_fragments
from portique_engine import resoudre_portique, resoudre_portique_multi

x = sp.Symbol('x')

st.set_page_config(page_title="StatiqueSolver", page_icon="📐", layout="wide")
st.title("📐 StatiqueSolver")

tab_poutre, tab_portique = st.tabs(["📏 Poutre simple", "🏗️ Structure libre (portique)"])

# ============================================================
# ONGLET 1 : POUTRE SIMPLE
# ============================================================
with tab_poutre:
    st.caption("Réactions, V(x), M(x) — poutre isostatique à 2 appuis. Accepte des nombres OU des variables (L, k1, q...)")

    mode_symbolique = st.toggle("🔤 Mode symbolique (variables type L, k1, q...)", value=False, key="toggle_sym")

    def parse(champ, defaut, key):
        txt = st.text_input(champ, value=defaut, key=key)
        try:
            return sp.sympify(txt)
        except Exception:
            st.error(f"Impossible d'interpréter « {txt} » — vérifie la syntaxe (ex: L/2, 2*L, k1*L)")
            st.stop()

    def tracer_diagramme_poutre(xs, Vs, Ms, points_symboliques=None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        ax1.plot(xs, Vs, color="#2563eb", linewidth=2)
        ax1.axhline(0, color="black", linewidth=0.8)
        ax1.fill_between(xs, Vs, 0, alpha=0.2, color="#2563eb")
        ax1.set_ylabel("V(x)")
        ax1.set_title("Effort tranchant")
        ax1.grid(alpha=0.3)

        ax2.plot(xs, Ms, color="#dc2626", linewidth=2)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.fill_between(xs, Ms, 0, alpha=0.2, color="#dc2626")
        ax2.set_ylabel("M(x)")
        ax2.set_xlabel("x")
        ax2.set_title("Moment fléchissant")
        ax2.grid(alpha=0.3)

        if points_symboliques:
            xticks, xticklabels = [], []
            for (xv, label_x, V_expr, M_expr) in points_symboliques:
                xticks.append(xv)
                xticklabels.append(f"${sp.latex(label_x)}$" if label_x is not None else f"{xv:.2g}")

                v_num = np.interp(xv, xs, Vs)
                ax1.plot([xv], [v_num], "o", color="#1e3a8a", markersize=4)
                if V_expr is not None:
                    V_lit = sp.simplify(V_expr.subs(x, label_x))
                    if V_lit != 0:
                        ax1.annotate(f"${sp.latex(V_lit)}$", (xv, v_num),
                                     textcoords="offset points", xytext=(0, 8 if v_num >= 0 else -16),
                                     fontsize=9, ha="center", color="#1e3a8a")

                m_num = np.interp(xv, xs, Ms)
                ax2.plot([xv], [m_num], "o", color="#991b1b", markersize=4)
                if M_expr is not None:
                    M_lit = sp.simplify(M_expr.subs(x, label_x))
                    if M_lit != 0:
                        ax2.annotate(f"${sp.latex(M_lit)}$", (xv, m_num),
                                     textcoords="offset points", xytext=(0, 8 if m_num >= 0 else -16),
                                     fontsize=9, ha="center", color="#991b1b")

            ax2.set_xticks(xticks)
            ax2.set_xticklabels(xticklabels)

        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Poutre")
    if mode_symbolique:
        L = parse("Longueur totale de la poutre", "L", "L_sym")
    else:
        L = st.number_input("Longueur totale de la poutre (m)", min_value=0.1, value=6.0, step=0.5, key="L_num")

    st.subheader("Appuis")
    col1, col2 = st.columns(2)
    with col1:
        if mode_symbolique:
            pos_A = parse("Position appui A", "0", "posA_sym")
        else:
            pos_A = st.number_input("Position appui A (m)", min_value=0.0, value=0.0, step=0.5, key="posA_num")
        type_A = st.selectbox("Type appui A", ["rotule", "simple"], key="typeA")
    with col2:
        if mode_symbolique:
            pos_B = parse("Position appui B", "L", "posB_sym")
        else:
            pos_B = st.number_input("Position appui B (m)", min_value=0.0, value=6.0, step=0.5, key="posB_num")
        type_B = st.selectbox("Type appui B", ["rotule", "simple"], key="typeB")

    st.subheader("Charges")
    if "charges" not in st.session_state:
        st.session_state.charges = []

    with st.expander("➕ Ajouter une charge", expanded=len(st.session_state.charges) == 0):
        type_charge = st.selectbox("Type de charge", ["ponctuelle", "repartie", "moment"], key="type_charge_poutre")
        if type_charge == "ponctuelle":
            if mode_symbolique:
                pos = parse("Position", "L/2", "pp_pos_sym")
                val = parse("Valeur (positif = vers le bas)", "P", "pp_val_sym")
            else:
                pos = st.number_input("Position (m)", min_value=0.0, value=3.0, key="p_pos")
                val = st.number_input("Valeur (kN, positif = vers le bas)", value=10.0, key="p_val")
            if st.button("Ajouter charge ponctuelle"):
                st.session_state.charges.append({"type": "ponctuelle", "position": pos, "valeur": val})
                st.rerun()
        elif type_charge == "repartie":
            if mode_symbolique:
                debut = parse("Début", "0", "rr_deb_sym")
                fin = parse("Fin", "L", "rr_fin_sym")
                val = parse("Intensité q (positif = vers le bas)", "q", "rr_val_sym")
            else:
                debut = st.number_input("Début (m)", min_value=0.0, value=0.0, key="r_deb")
                fin = st.number_input("Fin (m)", min_value=0.0, value=6.0, key="r_fin")
                val = st.number_input("Intensité q (kN/m, positif = vers le bas)", value=5.0, key="r_val")
            if st.button("Ajouter charge répartie"):
                st.session_state.charges.append({"type": "repartie", "debut": debut, "fin": fin, "valeur": val})
                st.rerun()
        else:
            if mode_symbolique:
                pos = parse("Position", "L/2", "mm_pos_sym")
                val = parse("Moment", "M0", "mm_val_sym")
            else:
                pos = st.number_input("Position (m)", min_value=0.0, value=3.0, key="m_pos")
                val = st.number_input("Moment (kN·m)", value=10.0, key="m_val")
            if st.button("Ajouter moment"):
                st.session_state.charges.append({"type": "moment", "position": pos, "valeur": val})
                st.rerun()

    if st.session_state.charges:
        st.write("**Charges actuelles :**")
        for i, c in enumerate(st.session_state.charges):
            c1, c2 = st.columns([5, 1])
            with c1:
                if c["type"] == "ponctuelle":
                    st.write(f"🔹 Ponctuelle : {c['valeur']} à x={c['position']}")
                elif c["type"] == "repartie":
                    st.write(f"🔹 Répartie : {c['valeur']} entre x={c['debut']} et x={c['fin']}")
                else:
                    st.write(f"🔹 Moment : {c['valeur']} à x={c['position']}")
            with c2:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.charges.pop(i)
                    st.rerun()

    st.divider()

    if st.button("🔍 Calculer", type="primary", use_container_width=True, key="calc_poutre"):
        if not st.session_state.charges:
            st.warning("Ajoute au moins une charge.")
        else:
            try:
                res = resoudre(L, [(pos_A, type_A), (pos_B, type_B)], st.session_state.charges)
                st.session_state.resultat = {
                    "V": res["V"], "M": res["M"], "reactions": res["reactions"],
                    "L": L, "pos_A": pos_A, "pos_B": pos_B, "mode_symbolique": mode_symbolique,
                    "charges_snapshot": list(st.session_state.charges),
                }
            except Exception as e:
                st.session_state.resultat = None
                st.error(f"Erreur de calcul : {e}")

    if st.session_state.get("resultat"):
        r = st.session_state.resultat
        V, M = r["V"], r["M"]
        reac_items = list(r["reactions"].items())
        L, pos_A, pos_B = r["L"], r["pos_A"], r["pos_B"]

        st.subheader("Réactions d'appuis")
        if r["mode_symbolique"]:
            st.latex(f"R_A = {sp.latex(sp.simplify(reac_items[0][1]))}")
            st.latex(f"R_B = {sp.latex(sp.simplify(reac_items[1][1]))}")
        else:
            rc1, rc2 = st.columns(2)
            with rc1:
                st.metric(f"Réaction en A (x={pos_A})", f"{abs(float(reac_items[0][1])):.2f} kN")
            with rc2:
                st.metric(f"Réaction en B (x={pos_B})", f"{abs(float(reac_items[1][1])):.2f} kN")

        if r["mode_symbolique"]:
            st.subheader("Formules littérales")
            st.write("Effort tranchant V(x) :")
            st.latex(f"V(x) = {sp.latex(sp.simplify(V))}")
            st.write("Moment fléchissant M(x) :")
            st.latex(f"M(x) = {sp.latex(sp.simplify(M))}")

            st.info("Donne des valeurs numériques aux symboles pour voir le diagramme :")
            symboles = sorted(V.free_symbols.union(M.free_symbols).union(sp.sympify(L).free_symbols) - {x}, key=str)
            valeurs_num = {}
            if symboles:
                cols = st.columns(min(len(symboles), 4))
                for i, sym in enumerate(symboles):
                    with cols[i % len(cols)]:
                        valeurs_num[sym] = st.number_input(f"{sym} =", value=1.0, key=f"num_{sym}")

            if st.button("📊 Afficher le diagramme symbolique"):
                L_num = float(sp.sympify(L).subs(valeurs_num))
                xs = np.linspace(0, L_num, 400)
                Vs, Ms = [], []
                for xv in xs:
                    subs_dict = {**valeurs_num, x: xv}
                    Vs.append(float(evaluer(V, subs_dict)))
                    Ms.append(float(evaluer(M, subs_dict)))

                positions_sym = {sp.sympify(0), sp.sympify(pos_A), sp.sympify(pos_B), sp.sympify(L)}
                for c in r["charges_snapshot"]:
                    if c["type"] in ("ponctuelle", "moment"):
                        positions_sym.add(sp.sympify(c["position"]))
                    elif c["type"] == "repartie":
                        positions_sym.add(sp.sympify(c["debut"]))
                        positions_sym.add(sp.sympify(c["fin"]))

                points_symboliques = []
                for p_sym in positions_sym:
                    try:
                        xv_num = float(p_sym.subs(valeurs_num))
                        if 0 <= xv_num <= L_num:
                            points_symboliques.append((xv_num, p_sym, V, M))
                    except Exception:
                        continue
                points_symboliques.sort(key=lambda t: t[0])

                tracer_diagramme_poutre(xs, Vs, Ms, points_symboliques=points_symboliques)
        else:
            xs = np.linspace(0, float(L), 300)
            Vs, Ms = [], []
            for xv in xs:
                Vs.append(float(evaluer(V, {x: xv})))
                Ms.append(float(evaluer(M, {x: xv})))
            Mmax_idx = max(range(len(Ms)), key=lambda i: abs(Ms[i]))
            st.metric("Moment maximal |M|max", f"{abs(Ms[Mmax_idx]):.2f} kN·m", f"à x={xs[Mmax_idx]:.2f} m")
            tracer_diagramme_poutre(xs, Vs, Ms)


# ============================================================
# ONGLET 2 : STRUCTURE LIBRE (PORTIQUE)
# ============================================================
with tab_portique:
    st.caption("Structure libre : nœuds, éléments, appuis (rotule/simple/encastrement), rotules internes, charges → NVM automatique")

    if "df_noeuds" not in st.session_state:
        st.session_state.df_noeuds = pd.DataFrame({
            "id": [1, 2, 3], "x": [0.0, 0.0, 4.0], "y": [0.0, 3.0, 3.0]
        })
    if "df_elements" not in st.session_state:
        st.session_state.df_elements = pd.DataFrame({
            "id": [101, 102], "noeud_a": [1, 2], "noeud_b": [2, 3]
        })
    if "df_supports" not in st.session_state:
        st.session_state.df_supports = pd.DataFrame({
            "noeud": [1], "type": ["encastrement"], "angle": [0.0]
        })
    if "df_hinges" not in st.session_state:
        st.session_state.df_hinges = pd.DataFrame({"noeud": []})
    if "df_charges_ponctuelles" not in st.session_state:
        st.session_state.df_charges_ponctuelles = pd.DataFrame({
            "noeud": [3], "Fx": [0.0], "Fy": [-10.0]
        })
    if "df_charges_reparties" not in st.session_state:
        st.session_state.df_charges_reparties = pd.DataFrame({"element": [], "qx": [], "qy": []})
    if "df_charges_moments" not in st.session_state:
        st.session_state.df_charges_moments = pd.DataFrame({"noeud": [], "valeur": []})

    st.markdown("### 1️⃣ Nœuds (coordonnées x,y)")
    st.session_state.df_noeuds = st.data_editor(
        st.session_state.df_noeuds, num_rows="dynamic", use_container_width=True, key="editor_noeuds"
    )

    st.markdown("### 2️⃣ Éléments (relient 2 nœuds — barres droites de ta structure)")
    st.session_state.df_elements = st.data_editor(
        st.session_state.df_elements, num_rows="dynamic", use_container_width=True, key="editor_elements"
    )

    colp1, colp2 = st.columns(2)
    with colp1:
        st.markdown("### 3️⃣ Appuis")
        st.caption("type : rotule / simple / encastrement — angle : direction de la réaction pour 'simple' (0°=verticale)")
        st.session_state.df_supports = st.data_editor(
            st.session_state.df_supports, num_rows="dynamic", use_container_width=True, key="editor_supports"
        )
    with colp2:
        st.markdown("### 4️⃣ Rotules internes")
        st.caption("Nœuds où le moment est nul (articulation entre 2 éléments)")
        st.session_state.df_hinges = st.data_editor(
            st.session_state.df_hinges, num_rows="dynamic", use_container_width=True, key="editor_hinges"
        )

    st.markdown("### 5️⃣ Charges")
    cp1, cp2, cp3 = st.columns(3)
    with cp1:
        st.caption("Ponctuelles (à un nœud)")
        st.session_state.df_charges_ponctuelles = st.data_editor(
            st.session_state.df_charges_ponctuelles, num_rows="dynamic", use_container_width=True, key="editor_cp"
        )
    with cp2:
        st.caption("Réparties (sur un élément, par unité de longueur)")
        st.session_state.df_charges_reparties = st.data_editor(
            st.session_state.df_charges_reparties, num_rows="dynamic", use_container_width=True, key="editor_cr"
        )
    with cp3:
        st.caption("Moments concentrés (à un nœud)")
        st.session_state.df_charges_moments = st.data_editor(
            st.session_state.df_charges_moments, num_rows="dynamic", use_container_width=True, key="editor_cm"
        )

    st.divider()

    def _dessiner_structure(ax, noeuds, elements, supports, hinges_ids, charges_pt):
        for eid, a, b in elements:
            xa, ya = noeuds[a]
            xb, yb = noeuds[b]
            ax.plot([xa, xb], [ya, yb], color="#334155", linewidth=3, zorder=1)

        for nid, (nx, ny) in noeuds.items():
            ax.plot(nx, ny, "o", color="#334155", markersize=5, zorder=2)

        for h in hinges_ids:
            hx, hy = noeuds[h]
            ax.plot(hx, hy, "o", markerfacecolor="white", markeredgecolor="#334155", markersize=10, zorder=3)

        for sup in supports:
            sx, sy = noeuds[sup["noeud"]]
            if sup["type"] == "encastrement":
                ax.plot(sx, sy, "s", color="#dc2626", markersize=14, zorder=3)
            elif sup["type"] == "rotule":
                ax.plot(sx, sy, "^", color="#2563eb", markersize=14, zorder=3)
            else:
                ax.plot(sx, sy, "^", color="#16a34a", markersize=14, zorder=3, markerfacecolor="white", markeredgewidth=2)

        for c in charges_pt:
            cx, cy = noeuds[c["noeud"]]
            if c["Fx"] != 0 or c["Fy"] != 0:
                scale = 0.6
                norm = max((c["Fx"] ** 2 + c["Fy"] ** 2) ** 0.5, 1e-6)
                ax.annotate("", xy=(cx + c["Fx"] / norm * scale, cy + c["Fy"] / norm * scale), xytext=(cx, cy),
                            arrowprops=dict(arrowstyle="->", color="#ea580c", linewidth=2), zorder=4)

        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title("Aperçu de la structure")

    try:
        noeuds_p = {int(row["id"]): (float(row["x"]), float(row["y"]))
                    for _, row in st.session_state.df_noeuds.dropna().iterrows()}
        elements_p = [(int(row["id"]), int(row["noeud_a"]), int(row["noeud_b"]))
                      for _, row in st.session_state.df_elements.dropna().iterrows()]
        supports_p = [{"noeud": int(row["noeud"]), "type": row["type"], "angle": float(row.get("angle", 0) or 0)}
                      for _, row in st.session_state.df_supports.dropna(subset=["noeud", "type"]).iterrows()]
        hinges_ids_p = set(int(v) for v in st.session_state.df_hinges["noeud"].dropna().tolist())

        charges_p = []
        charges_pt_liste = []
        for _, row in st.session_state.df_charges_ponctuelles.dropna().iterrows():
            c = {"type": "ponctuelle", "noeud": int(row["noeud"]), "Fx": float(row["Fx"]), "Fy": float(row["Fy"])}
            charges_p.append(c)
            charges_pt_liste.append(c)
        for _, row in st.session_state.df_charges_reparties.dropna().iterrows():
            charges_p.append({"type": "repartie", "element": int(row["element"]),
                               "qx": float(row["qx"]), "qy": float(row["qy"])})
        for _, row in st.session_state.df_charges_moments.dropna().iterrows():
            charges_p.append({"type": "moment", "noeud": int(row["noeud"]), "valeur": float(row["valeur"])})

        fig_p, ax_p = plt.subplots(figsize=(6, 5))
        _dessiner_structure(ax_p, noeuds_p, elements_p, supports_p, hinges_ids_p, charges_pt_liste)
        st.pyplot(fig_p)

    except Exception as e:
        st.warning(f"Aperçu impossible pour l'instant : {e}")
        noeuds_p, elements_p, supports_p, hinges_ids_p, charges_p = {}, [], [], set(), []

    if st.button("🔍 Résoudre la structure", type="primary", use_container_width=True, key="calc_portique"):
        try:
            fragments_p, hinges_p = construire_fragments(noeuds_p, elements_p, hinges_ids_p, supports_p, charges_p)

            if len(fragments_p) == 1 and not hinges_p:
                res_p = resoudre_portique(fragments_p[0]["waypoints"], fragments_p[0]["supports"], fragments_p[0]["charges"])
                st.session_state.resultat_portique = [res_p]
            else:
                res_multi = resoudre_portique_multi(fragments_p, hinges_p)
                st.session_state.resultat_portique = res_multi["fragments"]
        except Exception as e:
            st.session_state.resultat_portique = None
            st.error(f"Erreur : {e}")

    if st.session_state.get("resultat_portique"):
        fragments_resultats = st.session_state.resultat_portique

        st.subheader("Réactions d'appui")
        for i, fres in enumerate(fragments_resultats):
            for cle, val in fres["reactions"].items():
                if len(val) == 3:
                    st.write(f"Fragment {i+1}, {cle} : Rx={val[0]:.2f}, Ry={val[1]:.2f}, M={val[2]:.2f}")
                else:
                    st.write(f"Fragment {i+1}, {cle} : Rx={val[0]:.2f}, Ry={val[1]:.2f}")

        st.subheader("Diagrammes N, V, M")
        for i, fres in enumerate(fragments_resultats):
            st.markdown(f"**Fragment {i+1}**")
            fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
            offset = 0
            xticks = []
            for seg in fres["segments"]:
                svals = np.linspace(0, seg["L"], 60)
                Ns, Vs, Ms = [], [], []
                for sv in svals:
                    N, V, M = seg["fonction"](sv)
                    Ns.append(N)
                    Vs.append(V)
                    Ms.append(M)
                xs_plot = offset + svals
                axes[0].plot(xs_plot, Ns, color="#16a34a")
                axes[0].fill_between(xs_plot, Ns, 0, alpha=0.15, color="#16a34a")
                axes[1].plot(xs_plot, Vs, color="#2563eb")
                axes[1].fill_between(xs_plot, Vs, 0, alpha=0.15, color="#2563eb")
                axes[2].plot(xs_plot, Ms, color="#dc2626")
                axes[2].fill_between(xs_plot, Ms, 0, alpha=0.15, color="#dc2626")
                offset += seg["L"]
                xticks.append(offset)

            for ax, titre in zip(axes, ["N [effort normal]", "V [effort tranchant]", "M [moment]"]):
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_title(titre)
                ax.grid(alpha=0.3)
                for xt in xticks:
                    ax.axvline(xt, color="gray", linewidth=0.5, linestyle="--")

            plt.tight_layout()
            st.pyplot(fig)
