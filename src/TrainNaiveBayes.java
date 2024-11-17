import iteso.libs.models.NaiveBayesClassifier;
import iteso.libs.models.EvaluationMetric;
import iteso.libs.utils.CSVReader;
import iteso.libs.utils.TrainTestSplit;

import java.io.IOException;
import java.util.List;

public class TrainNaiveBayesWithIris {
    public static void main(String[] args) {
        try {
            // Ruta al archivo CSV
            String filePath = "src/iris.csv"; // Archivo limpio
            String labelColumn = "\"variety\""; 


            // Leer datos del CSV
            List<double[]> features = CSVReader.readFeatures(filePath, labelColumn);
            List<String> labels = CSVReader.readLabels(filePath, labelColumn);

            // Dividir los datos (80% entrenamiento, 20% prueba)
            TrainTestSplit.TrainTestData splitData = TrainTestSplit.splitDataWithShuffle(features, labels, 0.8);

            // Crear el modelo Naive Bayes
            NaiveBayesClassifier naiveBayes = new NaiveBayesClassifier();

            // Entrenar el modelo
            System.out.println("Entrenando el modelo Naive Bayes...");
            naiveBayes.train(splitData.trainFeatures, splitData.trainLabels);

            // Realizar predicciones
            System.out.println("Realizando predicciones...");
            List<String> predictedLabels = naiveBayes.predict(splitData.testFeatures);

            // Mostrar predicciones y etiquetas esperadas
            System.out.println("Predicciones: " + predictedLabels);
            System.out.println("Etiquetas esperadas: " + splitData.testLabels);

            // Evaluar el modelo
            System.out.println("Evaluando el modelo...");
            naiveBayes.evaluate(
            	    splitData.testLabels, // Etiquetas reales
            	    predictedLabels,      // Etiquetas predichas
            	    List.of(EvaluationMetric.ACCURACY) // Solo Accuracy
            	);


        } catch (IOException e) {
            System.err.println("Error al leer el archivo CSV: " + e.getMessage());
        } catch (IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}

