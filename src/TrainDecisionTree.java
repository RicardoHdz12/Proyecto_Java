import iteso.libs.models.*;
import iteso.libs.utils.CSVReader;
import iteso.libs.utils.TrainTestSplit;

import java.io.IOException;
import java.util.List;

public class TrainDecisionTree {
    public static void main(String[] args) {
        try {
            String filePath = "src/titanic_cleaned.csv"; // Dataset limpio
            String labelColumn = "Survived"; // Columna de etiquetas

            // Leer datos
            List<double[]> features = CSVReader.readFeatures(filePath, labelColumn);
            List<String> labels = CSVReader.readLabels(filePath, labelColumn);

            // Dividir datos en entrenamiento y prueba
            TrainTestSplit.TrainTestData splitData = TrainTestSplit.splitDataWithShuffle(features, labels, 0.8);

            // Crear y entrenar el modelo
            DecisionTreeClassifier decisionTree = new DecisionTreeClassifier();
            decisionTree.train(splitData.trainFeatures, splitData.trainLabels);

            // Realizar predicciones
            List<String> predictedLabels = decisionTree.predict(splitData.testFeatures);

            // Evaluar el modelo
            System.out.println("Evaluando el modelo...");
            decisionTree.evaluate(
                splitData.testLabels,
                predictedLabels,
                List.of(EvaluationMetric.ACCURACY)
            );
        } catch (IOException e) {
            System.err.println("Error al leer el archivo CSV: " + e.getMessage());
        } catch (IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}
