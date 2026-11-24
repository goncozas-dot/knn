from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SATISFACTION_FEATURES = [
    "calificacion_servicio",
    "tiempo_espera_min",
    "problema_resuelto",
    "amabilidad",
    "relacion_precio",
]

INVESTMENT_FEATURES = [
    "presupuesto_mensual",
    "numero_empleados",
    "horas_precios_mes",
    "errores_precios_mes",
    "ahorro_estimado_pct",
    "costo_software",
]


def create_satisfaction_data():
    """Create illustrative data; replace it with real customer survey records."""
    rows = []
    for rating, wait, resolved, courtesy, value in product(
        [1, 2, 3, 4, 5], [5, 20, 45], [0, 1], [1, 3, 5], [1, 3, 5]
    ):
        score = (
            2 * rating
            + 3 * resolved
            + courtesy
            + value
            - 2 * (wait >= 20)
            - 2 * (wait >= 45)
        )
        rows.append((rating, wait, resolved, courtesy, value, int(score >= 15)))

    return pd.DataFrame(
        rows, columns=SATISFACTION_FEATURES + ["satisfecho"]
    )


def create_investment_data():
    """Create illustrative data; replace it with real entrepreneur survey records."""
    rows = []
    for budget, employees, hours, errors, savings, cost in product(
        [500, 1000, 2000, 5000],
        [1, 5, 10, 25],
        [2, 8, 20, 40],
        [0, 1, 3, 8],
        [5, 10, 20, 35, 50],
        [500, 1000, 2000, 5000],
    ):
        can_afford = budget >= cost
        has_need = savings >= 20 or hours >= 20 or errors >= 3
        rows.append((budget, employees, hours, errors, savings, cost, int(can_afford and has_need)))

    return pd.DataFrame(rows, columns=INVESTMENT_FEATURES + ["invertiria"])


def train_and_evaluate(data, features, target):
    x_train, x_test, y_train, y_test = train_test_split(
        data[features],
        data[target],
        test_size=0.25,
        random_state=42,
        stratify=data[target],
    )
    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5, weights="distance"),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return model, y_test, predictions


def show_results(y_test, predictions, class_names, title, save_path=None):
    print(f"\nExactitud en datos de prueba: {accuracy_score(y_test, predictions):.2%}")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=class_names,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    figure, axis = plt.subplots()
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Prediccion",
        ylabel="Valor real",
        title=title,
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    if save_path is None:
        plt.show()
    else:
        figure.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(figure)


def ask_integer(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Ingresa un numero entero.")


def predict_satisfaction():
    features = SATISFACTION_FEATURES
    model, y_test, predictions = train_and_evaluate(
        create_satisfaction_data(), features, "satisfecho"
    )
    show_results(
        y_test, predictions, ["No satisfecho", "Satisfecho"],
        "Satisfaccion: matriz de confusion",
    )

    print("\nIngresa los datos del cliente:")
    values = [
        ask_integer("Calificacion del servicio (1-5): "),
        ask_integer("Tiempo de espera en minutos: "),
        ask_integer("Problema resuelto? (1=si, 0=no): "),
        ask_integer("Amabilidad del personal (1-5): "),
        ask_integer("Relacion precio-calidad (1-5): "),
    ]
    result = int(model.predict(pd.DataFrame([values], columns=features))[0])
    print(f"\nPrediccion: {'Satisfecho' if result else 'No satisfecho'}")


def predict_investment():
    features = INVESTMENT_FEATURES
    model, y_test, predictions = train_and_evaluate(
        create_investment_data(), features, "invertiria"
    )
    show_results(
        y_test, predictions, ["No invertiria", "Invertiria"],
        "Inversion: matriz de confusion",
    )

    print("\nIngresa los datos del empresario:")
    values = [
        ask_integer("Presupuesto mensual disponible (S/): "),
        ask_integer("Numero de empleados: "),
        ask_integer("Horas al mes dedicadas a fijar precios: "),
        ask_integer("Errores de precios al mes: "),
        ask_integer("Porcentaje de ahorro estimado (0-100): "),
        ask_integer("Costo del software (S/): "),
    ]
    result = int(model.predict(pd.DataFrame([values], columns=features))[0])
    print(f"\nPrediccion: {'Invertiria' if result else 'No invertiria'}")


def main():
    options = {
        "1": predict_satisfaction,
        "2": predict_investment,
    }
    while True:
        print("\n=== Modelos KNN ===")
        print("1. Predecir satisfaccion de un cliente")
        print("2. Predecir inversion en software de precios")
        print("0. Salir")
        choice = input("Selecciona una opcion: ").strip()
        if choice == "0":
            print("Hasta luego.")
            break
        action = options.get(choice)
        if action is None:
            print("Opcion no valida.")
        else:
            action()


if __name__ == "__main__":
    main()
