import iteso.libs.models.DecisionTreeClassifier;
import iteso.libs.models.EvaluationMetric;
import iteso.libs.utils.CSVReader;
import iteso.libs.utils.TrainTestSplit;
import java.io.IOException;
import java.util.List;

public class TrainDecisionTree {
    public static void main(String[] args) {
        try {
            String filePath = "src/titanic_cleaned.csv";
            String labelColumn = "survived";

            List<double[]> features = CSVReader.readFeatures(filePath, labelColumn);
            List<String> labels = CSVReader.readLabels(filePath, labelColumn);

            TrainTestSplit.TrainTestData splitData = TrainTestSplit.splitDataWithShuffle(features, labels, 0.8);

            DecisionTreeClassifier dt = new DecisionTreeClassifier();
            System.out.println("Entrenando el modelo Decision Tree...");
            dt.train(splitData.trainFeatures, splitData.trainLabels);

            System.out.println("Realizando predicciones...");
            List<String> preds = dt.predict(splitData.testFeatures);

            System.out.println("Predicciones: " + preds);
            System.out.println("Etiquetas reales: " + splitData.testLabels);

            System.out.println("Evaluando el modelo Decision Tree...");
            dt.evaluate(
                splitData.testLabels,
                preds,
                List.of(
                    EvaluationMetric.ACCURACY,
                    EvaluationMetric.PRECISION,
                    EvaluationMetric.RECALL,
                    EvaluationMetric.F1_SCORE
                ),
                "0" 
            );

        } catch (IOException e) {
            System.err.println("Error al leer el archivo CSV: " + e.getMessage());
        } catch (IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}
