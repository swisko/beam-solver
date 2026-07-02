import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from beam_engine import resoudre, evaluer

x = sp.Symbol('x')

st.set_page_config(page_title="BeamSolver", page_icon="📐", layout="centered")

st.title("📐 BeamSolver")
st.caption("Réactions, V(x), M(x) — poutre isostatique à 2 appuis. Accepte des nombres OU des variables (L, k1, q...)")

mode_symbolique = st.toggle("🔤 Mode symbolique (variables type L, k1, q...)", value=False)


def parse(champ, defaut):
    """Parse un champ texte en expression sympy (nombre ou symbole/formule)."""
    txt = st.text_input(champ, value=defaut)
    try:
        return sp.sympify(txt)
    except Exception:
        st.error(f"Impossible d'interpréter « {txt} » — vérifie la syntaxe (ex: L/2, 2*L, k1*L)")
        st.stop()


def tracer_diagramme(xs, Vs, Ms, points_symboliques=None):
    """
    points_symboliques (optionnel, mode symbolique uniquement) :
    liste de tuples (x_numerique, label_x_symbolique, V_expr_sympy, M_expr_sympy)
    -> annote le diagramme avec les formules littérales aux points clés
    """
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

            Vv = float(V_expr.subs(x, label_x)) if V_expr is not None else None
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
    L = parse("Longueur totale de la poutre", "L")
else:
    L = st.number_input("Longueur totale de la poutre (m)", min_value=0.1, value=6.0, step=0.5)

st.subheader("Appuis")
col1, col2 = st.columns(2)
with col1:
    if mode_symbolique:
        pos_A = parse("Position appui A", "0")
    else:
        pos_A = st.number_input("Position appui A (m)", min_value=0.0, value=0.0, step=0.5)
    type_A = st.selectbox("Type appui A", ["rotule", "simple"], key="typeA")
with col2:
    if mode_symbolique:
        pos_B = parse("Position appui B", "L")
    else:
        pos_B = st.number_input("Position appui B (m)", min_value=0.0, value=6.0, step=0.5)
    type_B = st.selectbox("Type appui B", ["rotule", "simple"], key="typeB")

st.subheader("Charges")
if "charges" not in st.session_state:
    st.session_state.charges = []

with st.expander("➕ Ajouter une charge", expanded=len(st.session_state.charges) == 0):
    type_charge = st.selectbox("Type de charge", ["ponctuelle", "repartie", "moment"])
    if type_charge == "ponctuelle":
        if mode_symbolique:
            pos = parse("Position", "L/2")
            val = parse("Valeur (positif = vers le bas)", "P")
        else:
            pos = st.number_input("Position (m)", min_value=0.0, value=3.0, key="p_pos")
            val = st.number_input("Valeur (kN, positif = vers le bas)", value=10.0, key="p_val")
        if st.button("Ajouter charge ponctuelle"):
            st.session_state.charges.append({"type": "ponctuelle", "position": pos, "valeur": val})
            st.rerun()
    elif type_charge == "repartie":
        if mode_symbolique:
            debut = parse("Début", "0")
            fin = parse("Fin", "L")
            val = parse("Intensité q (positif = vers le bas)", "q")
        else:
            debut = st.number_input("Début (m)", min_value=0.0, value=0.0, key="r_deb")
            fin = st.number_input("Fin (m)", min_value=0.0, value=6.0, key="r_fin")
            val = st.number_input("Intensité q (kN/m, positif = vers le bas)", value=5.0, key="r_val")
        if st.button("Ajouter charge répartie"):
            st.session_state.charges.append({"type": "repartie", "debut": debut, "fin": fin, "valeur": val})
            st.rerun()
    else:
        if mode_symbolique:
            pos = parse("Position", "L/2")
            val = parse("Moment", "M0")
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

if st.button("🔍 Calculer", type="primary", use_container_width=True):
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

            tracer_diagramme(xs, Vs, Ms, points_symboliques=points_symboliques)
    else:
        xs = np.linspace(0, float(L), 300)
        Vs, Ms = [], []
        for xv in xs:
            Vs.append(float(evaluer(V, {x: xv})))
            Ms.append(float(evaluer(M, {x: xv})))
        Mmax_idx = max(range(len(Ms)), key=lambda i: abs(Ms[i]))
        st.metric("Moment maximal |M|max", f"{abs(Ms[Mmax_idx]):.2f} kN·m", f"à x={xs[Mmax_idx]:.2f} m")
        tracer_diagramme(xs, Vs, Ms)
