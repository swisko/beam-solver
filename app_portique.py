import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fragment_builder import construire_fragments
from portique_engine import resoudre_portique, resoudre_portique_multi

st.set_page_command = None
st.set_page_config(page_title="PortiqueSolver", page_icon="🏗️", layout="wide")

st.title("🏗️ PortiqueSolver")
st.caption("Structure libre : nœuds, éléments, appuis (rotule/simple/encastrement), rotules internes, charges → NVM automatique")

# --- État initial ---
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

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 3️⃣ Appuis")
    st.caption("type : rotule / simple / encastrement — angle : direction de la réaction pour 'simple' (0°=verticale)")
    st.session_state.df_supports = st.data_editor(
        st.session_state.df_supports, num_rows="dynamic", use_container_width=True, key="editor_supports"
    )
with col2:
    st.markdown("### 4️⃣ Rotules internes")
    st.caption("Nœuds où le moment est nul (articulation entre 2 éléments)")
    st.session_state.df_hinges = st.data_editor(
        st.session_state.df_hinges, num_rows="dynamic", use_container_width=True, key="editor_hinges"
    )

st.markdown("### 5️⃣ Charges")
c1, c2, c3 = st.columns(3)
with c1:
    st.caption("Ponctuelles (à un nœud)")
    st.session_state.df_charges_ponctuelles = st.data_editor(
        st.session_state.df_charges_ponctuelles, num_rows="dynamic", use_container_width=True, key="editor_cp"
    )
with c2:
    st.caption("Réparties (sur un élément, par unité de longueur)")
    st.session_state.df_charges_reparties = st.data_editor(
        st.session_state.df_charges_reparties, num_rows="dynamic", use_container_width=True, key="editor_cr"
    )
with c3:
    st.caption("Moments concentrés (à un nœud)")
    st.session_state.df_charges_moments = st.data_editor(
        st.session_state.df_charges_moments, num_rows="dynamic", use_container_width=True, key="editor_cm"
    )

st.divider()


def _dessiner_structure(ax, noeuds, elements, supports, hinges_ids, charges_pt, charges_rep):
    for eid, a, b in elements:
        xa, ya = noeuds[a]
        xb, yb = noeuds[b]
        ax.plot([xa, xb], [ya, yb], color="#334155", linewidth=3, zorder=1)

    for nid, (x, y) in noeuds.items():
        ax.plot(x, y, "o", color="#334155", markersize=5, zorder=2)

    for h in hinges_ids:
        x, y = noeuds[h]
        ax.plot(x, y, "o", markerfacecolor="white", markeredgecolor="#334155", markersize=10, zorder=3)

    for sup in supports:
        x, y = noeuds[sup["noeud"]]
        if sup["type"] == "encastrement":
            ax.plot(x, y, "s", color="#dc2626", markersize=14, zorder=3)
        elif sup["type"] == "rotule":
            ax.plot(x, y, "^", color="#2563eb", markersize=14, zorder=3)
        else:
            ax.plot(x, y, "^", color="#16a34a", markersize=14, zorder=3, markerfacecolor="white", markeredgewidth=2)

    for c in charges_pt:
        x, y = noeuds[c["noeud"]]
        if c["Fx"] != 0 or c["Fy"] != 0:
            scale = 0.6
            norm = max((c["Fx"]**2 + c["Fy"]**2) ** 0.5, 1e-6)
            ax.annotate("", xy=(x + c["Fx"]/norm*scale, y + c["Fy"]/norm*scale), xytext=(x, y),
                        arrowprops=dict(arrowstyle="->", color="#ea580c", linewidth=2), zorder=4)

    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("Aperçu de la structure")


try:
    noeuds = {int(row["id"]): (float(row["x"]), float(row["y"]))
              for _, row in st.session_state.df_noeuds.dropna().iterrows()}
    elements = [(int(row["id"]), int(row["noeud_a"]), int(row["noeud_b"]))
                for _, row in st.session_state.df_elements.dropna().iterrows()]
    supports = [{"noeud": int(row["noeud"]), "type": row["type"], "angle": float(row.get("angle", 0) or 0)}
                for _, row in st.session_state.df_supports.dropna(subset=["noeud", "type"]).iterrows()]
    hinges_ids = set(int(v) for v in st.session_state.df_hinges["noeud"].dropna().tolist())

    charges = []
    charges_pt_liste = []
    for _, row in st.session_state.df_charges_ponctuelles.dropna().iterrows():
        c = {"type": "ponctuelle", "noeud": int(row["noeud"]), "Fx": float(row["Fx"]), "Fy": float(row["Fy"])}
        charges.append(c)
        charges_pt_liste.append(c)
    for _, row in st.session_state.df_charges_reparties.dropna().iterrows():
        charges.append({"type": "repartie", "element": int(row["element"]), "qx": float(row["qx"]), "qy": float(row["qy"])})
    for _, row in st.session_state.df_charges_moments.dropna().iterrows():
        charges.append({"type": "moment", "noeud": int(row["noeud"]), "valeur": float(row["valeur"])})

    fig, ax = plt.subplots(figsize=(6, 5))
    _dessiner_structure(ax, noeuds, elements, supports, hinges_ids, charges_pt_liste, [])
    st.pyplot(fig)

except Exception as e:
    st.warning(f"Aperçu impossible pour l'instant : {e}")
    noeuds, elements, supports, hinges_ids, charges = {}, [], [], set(), []

if st.button("🔍 Résoudre la structure", type="primary", use_container_width=True):
    try:
        fragments, hinges = construire_fragments(noeuds, elements, hinges_ids, supports, charges)

        if len(fragments) == 1 and not hinges:
            res = resoudre_portique(fragments[0]["waypoints"], fragments[0]["supports"], fragments[0]["charges"])
            fragments_resultats = [res]
        else:
            res_multi = resoudre_portique_multi(fragments, hinges)
            fragments_resultats = res_multi["fragments"]

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
            xticks, xticklabels = [], []
            for seg in fres["segments"]:
                svals = np.linspace(0, seg["L"], 60)
                Ns, Vs, Ms = [], [], []
                for sv in svals:
                    N, V, M = seg["fonction"](sv)
                    Ns.append(N); Vs.append(V); Ms.append(M)
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

    except Exception as e:
        st.error(f"Erreur : {e}")
