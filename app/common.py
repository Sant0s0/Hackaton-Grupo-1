from dataclasses import replace
import json
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from src.data import FEATURES, LABELS, ROOT, SEED, fingerprints
from src.evaluation import evaluate_candidate
from src.models import Candidate
from src.persistence import prepare_model

FIELD_LABELS = {
    "fixed_acidity": "Acidez fija (g/dm³)", "volatile_acidity": "Acidez volátil (g/dm³)",
    "citric_acid": "Ácido cítrico (g/dm³)", "residual_sugar": "Azúcar residual (g/dm³)",
    "chlorides": "Cloruros (g/dm³)", "free_sulfur_dioxide": "SO₂ libre (mg/dm³)",
    "total_sulfur_dioxide": "SO₂ total (mg/dm³)", "density": "Densidad (g/cm³)",
    "pH": "pH", "sulphates": "Sulfatos (g/dm³)", "alcohol": "Alcohol (% vol.)",
}


def show_figure(fig):
    if st.session_state.get("dark_visuals", False):
        fig.set_facecolor("#11182a")
        for ax in fig.axes:
            ax.set_facecolor("#11182a")
            ax.tick_params(colors="#cbd5e1")
            for label in [ax.title, ax.xaxis.label, ax.yaxis.label]:
                label.set_color("#e2e8f0")
            for spine in ax.spines.values():
                spine.set_color("#45536b")
            if not ax.images:
                for text in ax.texts:
                    text.set_color("#f1f5f9")
            if ax.get_legend():
                ax.get_legend().get_frame().set_facecolor("#11182a")
                for text in ax.get_legend().get_texts():
                    text.set_color("#e2e8f0")
    st.pyplot(fig)
    plt.close(fig)


def saved_experiment():
    path = ROOT / "outputs/experiment.json"
    if not path.exists():
        return pd.DataFrame(), {}, []
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta["sha256"] != fingerprints():
        st.error("Los datos cambiaron desde el entrenamiento. Ejecuta scripts/entrenar.py para actualizar los resultados.")
        st.stop()
    results = pd.read_csv(ROOT / "outputs/cv_results.csv")
    specs = [Candidate(**item) for item in json.loads((ROOT / "outputs/candidates.json").read_text(encoding="utf-8"))]
    return results, meta, specs


def metrics_table(results):
    columns = ["name", "family", "f1_macro_mean", "f1_macro_std", "f1_1_mean",
               "recall_1_mean", "accuracy_mean", "fit_seconds_mean", "prediction_seconds_mean", "warnings_count"]
    selected = results.loc[:, columns].rename(columns={
        "name": "Configuración", "family": "Método", "f1_macro_mean": "F1 macro",
        "f1_macro_std": "Desv. entre folds", "f1_1_mean": "F1 superior", "recall_1_mean": "Recall superior",
        "accuracy_mean": "Accuracy", "fit_seconds_mean": "Ajuste (s/fold)",
        "prediction_seconds_mean": "Predicción (s/fold)", "warnings_count": "Avisos",
    })
    st.dataframe(selected, hide_index=True, width="stretch")


@st.cache_resource(show_spinner="Preparando el método elegido…")
def fitted_model(spec_json, train_hash, experiment_stamp):
    spec = Candidate(**json.loads(spec_json))
    meta_path = ROOT / "outputs/experiment.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    train = pd.read_csv(ROOT / "data/train_public.csv")
    hashes = fingerprints()
    if hashes["train_public.csv"] != train_hash:
        raise ValueError("Los datos cambiaron durante la preparación del modelo.")
    artifact = prepare_model(spec, train, hashes, meta)
    return artifact["pipeline"], artifact.get("fit_warnings", [])


def get_fitted(spec, meta):
    return fitted_model(json.dumps(spec.to_dict(), sort_keys=True), fingerprints()["train_public.csv"],
                        meta.get("created_utc", "sin_experimento"))


def simulator(spec, train, meta, key, pipeline_loader=None):
    st.subheader("Prueba una muestra")
    st.caption("Introduce las once mediciones de laboratorio. 0 = estándar; 1 = superior.")
    with st.form(f"simulator_{key}"):
        columns = st.columns(3)
        values = {}
        for index, feature in enumerate(FEATURES):
            with columns[index % 3]:
                values[feature] = st.number_input(
                    FIELD_LABELS[feature], min_value=0.0, value=float(train[feature].median()),
                    format="%.5f" if feature == "density" else "%.4f", key=f"{key}_{feature}")
        submitted = st.form_submit_button("Predecir calidad", type="primary")
    if submitted:
        sample = pd.DataFrame([values], columns=FEATURES)
        if values["density"] == 0:
            st.error("La densidad debe ser mayor que cero.")
            return
        if values["pH"] > 14:
            st.error("Revisa el pH: el simulador admite valores entre 0 y 14.")
            return
        outside = [f for f in FEATURES if not train[f].min() <= values[f] <= train[f].max()]
        if values["free_sulfur_dioxide"] > values["total_sulfur_dioxide"]:
            st.error("El SO₂ libre no puede superar al SO₂ total. Revisa esas mediciones.")
            return
        if outside:
            st.warning("Fuera del rango observado en entrenamiento: " + ", ".join(outside))
        model, messages = pipeline_loader() if pipeline_loader else get_fitted(spec, meta)
        with threadpool_limits(limits=1):
            pred = int(model.predict(sample)[0])
            if spec.family == "SVM":
                score = float(model.decision_function(sample)[0])
            else:
                score = float(model.predict_proba(sample)[0, 1])
        st.success(f"{LABELS[pred]} · clase {pred}")
        if spec.family == "SVM":
            st.metric("Puntaje de decisión SVM", f"{score:.4f}")
            st.caption("Es un puntaje de decisión con umbral 0; no es una probabilidad ni una distancia euclidiana normalizada.")
        else:
            label = {"k-NN": "Fracción ponderada de votos para clase 1", "k-means": "Proporción de clase 1 en el clúster",
                     "MLP": "Estimación de salida para clase 1"}[spec.family]
            st.metric(label, f"{score:.1%}")
            st.caption("Esta salida no se ha calibrado como probabilidad.")
        if messages:
            st.warning("El ajuste emitió avisos: " + " | ".join(messages))


def model_panel(options, results, train, meta, key):
    if not options:
        st.info("Ejecuta primero python scripts/entrenar.py para generar las configuraciones.")
        return
    names = {s.name: s for s in options}
    ranked = results[results.name.isin(names)].sort_values("f1_macro_mean", ascending=False)
    ordered = ranked.name.tolist() or list(names)
    chosen_name = st.selectbox("Configuración evaluada", ordered, key=f"variant_{key}")
    spec = names[chosen_name]
    custom = st.toggle("Ajustar parámetros del método", key=f"custom_{key}")
    if custom:
        params = dict(spec.params)
        a, b, c = st.columns(3)
        if spec.family == "k-NN":
            params["n_neighbors"] = a.slider("Vecinos k", 1, 31, int(params["n_neighbors"]), key=f"k_{key}")
            metrics = ["euclidean", "manhattan", "minkowski", "cosine"]
            params["metric"] = b.selectbox("Distancia / disimilitud", metrics, index=metrics.index(params.get("metric", "euclidean")), key=f"metric_{key}")
            params["weights"] = c.selectbox("Votación", ["uniform", "distance"], index=int(params.get("weights") == "distance"), key=f"weights_{key}")
            params.pop("p", None)
            if params["metric"] == "minkowski":
                params["p"] = st.slider("Orden p de Minkowski", 1, 5, 3, key=f"p_{key}")
        elif spec.family == "k-means":
            params["n_clusters"] = a.slider("Número de clústeres", 2, 40, int(params["n_clusters"]), key=f"clusters_{key}")
        elif spec.family == "SVM":
            params["C"] = a.select_slider("C", [0.1, 1.0, 10.0, 100.0], value=float(params["C"]), key=f"C_{key}")
            params["class_weight"] = b.selectbox("Pesos de clase", [None, "balanced"], index=int(params.get("class_weight") == "balanced"), key=f"cw_{key}")
            if params["kernel"] != "linear":
                gamma_choices = ["scale", 0.03, 0.1, 1.0]
                params["gamma"] = c.selectbox("Gamma", gamma_choices,
                    index=gamma_choices.index(params.get("gamma", "scale")), key=f"gamma_{key}")
            if params["kernel"] == "poly":
                params["degree"] = st.slider("Grado del polinomio", 2, 4, 3, key=f"degree_{key}")
        else:
            layers = [(32,), (64, 32)]
            activations = ["relu", "tanh"]
            alphas = [0.0001, 0.001, 0.01]
            params["hidden_layer_sizes"] = tuple(a.selectbox("Capas ocultas", layers,
                index=layers.index(tuple(params["hidden_layer_sizes"])), key=f"layers_{key}"))
            params["activation"] = b.selectbox("Activación", activations,
                index=activations.index(params["activation"]), key=f"act_{key}")
            params["alpha"] = c.selectbox("Regularización alpha", alphas,
                index=alphas.index(params["alpha"]), key=f"alpha_{key}")
            params["balanced"] = st.checkbox("Ponderar clases en la pérdida", value=params.get("balanced", False), key=f"balanced_{key}")
        ratios = st.checkbox("Añadir los tres cocientes fisicoquímicos", value=spec.ratios, key=f"ratios_{key}")
        pca_choice = st.selectbox("PCA: varianza a conservar", ["Sin PCA", 0.90, 0.95],
                                 index=0 if not spec.pca else (1 if spec.pca == .90 else 2), key=f"pca_{key}")
        spec = replace(spec, name=f"personalizado_{key}", params=params, ratios=ratios,
                       pca=None if pca_choice == "Sin PCA" else pca_choice)
        st.caption("Una configuración personalizada no cambia la comparación oficial ni la entrega guardada.")
        st.caption("La primera predicción con parámetros nuevos puede tardar mientras se prepara el modelo; después queda guardado. Evaluar cinco folds vuelve a entrenar y puede demorar.")
        if st.button("Evaluar esta configuración con 5 folds", key=f"evaluate_{key}"):
            with st.spinner("Evaluando únicamente sobre train_public…"):
                splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(train[FEATURES], train.target))
                summary, detail, _ = evaluate_candidate(spec, train[FEATURES], train.target, splits)
            st.metric("F1 macro medio exploratorio", f"{summary['f1_macro_mean']:.4f}")
            st.dataframe(detail.drop(columns=["warnings"]), hide_index=True)
    elif not ranked.empty:
        row = ranked[ranked.name == chosen_name].iloc[0]
        cols = st.columns(3)
        cols[0].metric("F1 macro · CV", f"{row.f1_macro_mean:.4f}")
        cols[1].metric("Desv. entre folds", f"{row.f1_macro_std:.4f}")
        cols[2].metric("F1 clase superior", f"{row.f1_1_mean:.4f}")
        if row.warnings_count:
            st.info(f"Se registraron {int(row.warnings_count)} avisos de ajuste; consulta outputs/cv_folds.csv.")
    with st.expander("Ver parámetros y transformaciones"):
        st.json(spec.to_dict())
    simulator(spec, train, meta, key)
