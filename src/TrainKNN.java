import iteso.libs.models.*;
import iteso.libs.utils.CSVReader;
import iteso.libs.utils.TrainTestSplit;

import java.io.IOException;
import java.util.List;

public class TrainKNNWithIris {
    public static void main(String[] args) {
        try {
            // Ruta al archivo CSV
            String filePath = "src/titanic_cleaned.csv"; // Archivo Titanic limpio
            String labelColumn = "Survived"; // Columna de etiquetas

            // Leer características y etiquetas desde el archivo CSV
            List<double[]> features = CSVReader.readFeatures(filePath, labelColumn);
            List<String> labels = CSVReader.readLabels(filePath, labelColumn);

            // Dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
            TrainTestSplit.TrainTestData splitData = TrainTestSplit.splitDataWithShuffle(features, labels, 0.8);

            // Crear el modelo KNN
            KNNClassifier knn = new KNNClassifier(6, KNNClassifier.DistanceType.EUCLIDEAN);

            // Entrenar el modelo con los datos de entrenamiento
            System.out.println("Entrenando el modelo KNN...");
            knn.train(splitData.trainFeatures, splitData.trainLabels);

            // Realizar predicciones con los datos de prueba
            System.out.println("Realizando predicciones...");
            List<String> predictedLabels = knn.predict(splitData.testFeatures);

            // Mostrar las predicciones y las etiquetas esperadas
            System.out.println("Predicciones: " + predictedLabels);
            System.out.println("Etiquetas esperadas: " + splitData.testLabels);

            // Evaluar el modelo con las métricas
            System.out.println("Evaluando el modelo...");
            knn.evaluate(
                splitData.testLabels, // Etiquetas reales
                predictedLabels, // Etiquetas predichas
                List.of(
                    EvaluationMetric.ACCURACY, // Calcular Accuracy
                    EvaluationMetric.RECALL    // Calcular Recall
                ),
                "0"// Definir "0" como la clase positiva para recall
				//(¿Qué tan bien el modelo predice quién no sobrevivió?
            );

        } catch (IOException e) {
            System.err.println("Error al leer el archivo CSV: " + e.getMessage());
        } catch (IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}
