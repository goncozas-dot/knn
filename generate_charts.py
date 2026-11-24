from pathlib import Path

import main


def generate_charts():
    output_dir = Path(__file__).resolve().parent / "assets"
    output_dir.mkdir(exist_ok=True)

    models = [
        (
            "satisfaction_confusion_matrix.png",
            main.create_satisfaction_data(),
            main.SATISFACTION_FEATURES,
            "satisfecho",
            ["No satisfecho", "Satisfecho"],
            "Satisfaccion: matriz de confusion",
        ),
        (
            "investment_confusion_matrix.png",
            main.create_investment_data(),
            main.INVESTMENT_FEATURES,
            "invertiria",
            ["No invertiria", "Invertiria"],
            "Inversion: matriz de confusion",
        ),
    ]

    for filename, data, features, target, class_names, title in models:
        _, y_test, predictions = main.train_and_evaluate(data, features, target)
        output_path = output_dir / filename
        main.show_results(
            y_test,
            predictions,
            class_names,
            title,
            save_path=output_path,
        )
        print(f"Grafico generado: {output_path}")


if __name__ == "__main__":
    generate_charts()
