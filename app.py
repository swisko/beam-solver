import streamlit as st
import sympy as sp
import matplotlib.pyplot as plt
from beam_engine import resoudre, evaluer

x = sp.Symbol('x')

st.set_page_config(page_title="BeamSolver", page_icon="📐", layout="centered")

st.title("📐 BeamSolver")
st.caption("Réactions, effort tranchant V(x) et moment fléchissant M(x) — poutre isostatique à 2 appuis")

# --- Longueur de la poutre ---
L = st.number_input("Longueur totale de la poutre (m)", min_value=0.1, value=6.0, step=0.5)

st.subheader("Appuis")
col1, col2 = st.columns(2)
with col1:
    pos_A = st.number_input("Position appui A (m)", min_value=0.0, max_value=float(L), value=0.0, step=0.5)
    type_A = st.selectbox("Type appui A", ["rotule", "simple"], key="typeA")
with col2:
    pos_B = st.number_input("Position appui B (m)", min_value=0.0, max_value=float(L), value=float(L), step=0.5)
    type_B = st.selectbox("Type appui B", ["rotule", "simple"], key="typeB")

st.subheader("Charges")
if "charges" not in st.session_state:
    st.session_state.charges = []

with st.expander("➕ Ajouter une charge"):
    type_charge = st.selectbox("Type de charge", ["ponctuelle", "repartie", "moment"])
    if type_charge == "ponctuelle":
        pos = st.number_input("Position (m)", min_value=0.0, max_value=float(L), value=float(L)/2, key="p_pos")
        val = st.number_input("Valeur (kN, positif = vers le bas)", value=10.0, key="p_val")
        if st.button("Ajouter charge ponctuelle"):
            st.session_state.charges.append({"type": "ponctuelle", "position": pos, "valeur": val})
            st.rerun()
    elif type_charge == "repartie":
        debut = st.number_input("Début (m)", min_value=0.0, max_value=float(L), value=0.0, key="r_deb")
        fin = st.number_input("Fin (m)", min_value=0.0, max_value=float(L), value=float(L), key="r_fin")
        val = st.number_input("Intensité q (kN/m, positif = vers le bas)", value=5.0, key="r_val")
        if st.button("Ajouter charge répartie"):
            st.session_state.charges.append({"type": "repartie", "debut": debut, "fin": fin, "valeur": val})
            st.rerun()
    else:
        pos = st.number_input("Position (m)", min_value=0.0, max_value=float(L), value=float(L)/2, key="m_pos")
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
                st.write(f"🔹 Ponctuelle : {c['valeur']} kN à x={c['position']} m")
            elif c["type"] == "repartie":
                st.write(f"🔹 Répartie : {c['valeur']} kN/m entre x={c['debut']} et x={c['fin']} m")
            else:
                st.write(f"🔹 Moment : {c['valeur']} kN·m à x={c['position']} m")
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
            V, M = res["V"], res["M"]

            # Réactions
            st.subheader("Réactions d'appuis")
            reac_items = list(res["reactions"].items())
            rc1, rc2 = st.columns(2)
            with rc1:
                st.metric(f"Réaction en A (x={pos_A})", f"{abs(float(reac_items[0][1])):.2f} kN")
            with rc2:
                st.metric(f"Réaction en B (x={pos_B})", f"{abs(float(reac_items[1][1])):.2f} kN")

            # Calcul des courbes
            import numpy as np
            xs = np.linspace(0, L, 300)
            Vs, Ms = [], []
            for xv in xs:
                Vs.append(float(evaluer(V, {x: xv})))
                Ms.append(float(evaluer(M, {x: xv})))

            Mmax_idx = max(range(len(Ms)), key=lambda i: abs(Ms[i]))
            st.metric("Moment maximal |M|max", f"{abs(Ms[Mmax_idx]):.2f} kN·m", f"à x={xs[Mmax_idx]:.2f} m")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            ax1.plot(xs, Vs, color="#2563eb", linewidth=2)
            ax1.axhline(0, color="black", linewidth=0.8)
            ax1.fill_between(xs, Vs, 0, alpha=0.2, color="#2563eb")
            ax1.set_ylabel("V(x) [kN]")
            ax1.set_title("Effort tranchant")
            ax1.grid(alpha=0.3)

            ax2.plot(xs, Ms, color="#dc2626", linewidth=2)
            ax2.axhline(0, color="black", linewidth=0.8)
            ax2.fill_between(xs, Ms, 0, alpha=0.2, color="#dc2626")
            ax2.set_ylabel("M(x) [kN·m]")
            ax2.set_xlabel("x [m]")
            ax2.set_title("Moment fléchissant")
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Erreur de calcul : {e}")
