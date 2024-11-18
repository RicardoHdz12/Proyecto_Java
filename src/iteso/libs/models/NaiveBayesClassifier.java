package iteso.libs.models;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import iteso.libs.metrics.Metrics;

public class NaiveBayesClassifier {
	

    private Map<String, Double> priorProbabilities; // Probabilidades previas de cada clase
    private Map<String, Map<Integer, Map<Double, Double>>> conditionalProbabilities; // P(condición | clase)
    private List<String> classes; // Lista de clases únicas

    public NaiveBayesClassifier() {
        priorProbabilities = new HashMap<>();
        conditionalProbabilities = new HashMap<>();
    }

    public void train(List<double[]> data, List<String> labels) {
        ErrorManager.validateDataAndLabelsSize(data, labels); // Validación de tamaño de datos y etiquetas
        ErrorManager.checkDataNotEmpty(data); // Validación de que los datos no estén vacíos

        classes = labels.stream().distinct().toList(); // Obtener las clases únicas

        for (String label : labels) {
            int count = 0;
            for (String l : labels) {
                if (l.equals(label)) {
                    count++; // Contar cuántas veces aparece 'label' en 'labels'
                }
            }
            double priorProbability = (double) count / labels.size();
            priorProbabilities.put(label, priorProbability); // Guardar la probabilidad previa
        }


        // Calcular probabilidades condicionales
        for (String label : classes) {
            conditionalProbabilities.put(label, new HashMap<>());

            for (int featureIndex = 0; featureIndex < data.get(0).length; featureIndex++) {
                Map<Double, Double> featureValueCounts = new HashMap<>();
                double totalFeatureCount = 0;

                for (int i = 0; i < data.size(); i++) {
                    if (labels.get(i).equals(label)) {
                        double featureValue = data.get(i)[featureIndex];
                        featureValueCounts.put(featureValue, featureValueCounts.getOrDefault(featureValue, 0.0) + 1);
                        totalFeatureCount++;
                    }
                }

                // Normalizar las probabilidades condicionales
                for (Map.Entry<Double, Double> entry : featureValueCounts.entrySet()) {
                    featureValueCounts.put(entry.getKey(), entry.getValue() / totalFeatureCount);
                }
                conditionalProbabilities.get(label).put(featureIndex, featureValueCounts);
            }
        }
    }

    public String predict(double[] sample) {
    	ErrorManager.checkMapNotEmpty(priorProbabilities); // Validar que el modelo haya sido entrenado
        String bestClass = null;
        double maxProbability = Double.NEGATIVE_INFINITY;

        for (String label : classes) {
            double probability = Math.log(priorProbabilities.get(label)); // Usar log para evitar underflow

            for (int featureIndex = 0; featureIndex < sample.length; featureIndex++) {
                double featureValue = sample[featureIndex];
                Map<Double, Double> featureProbabilities = conditionalProbabilities.get(label).get(featureIndex);
                probability += Math.log(featureProbabilities.getOrDefault(featureValue, 1e-6)); // Laplace smoothing
            }

            if (probability > maxProbability) {
                maxProbability = probability;
                bestClass = label;
            }
        }

        return bestClass;
    }

    public List<String> predict(List<double[]> samples) {
        return samples.stream().map(this::predict).toList();
    }
    public void evaluate(List<String> trueLabels, List<String> predictedLabels, List<EvaluationMetric> metrics) {
        for (EvaluationMetric metric : metrics) {
            switch (metric) {
                case ACCURACY:
                    double accuracy = Metrics.calculateAccuracy(trueLabels, predictedLabels);
                    System.out.println("Accuracy: " + accuracy);
                    break;
                default:
                    throw new IllegalArgumentException("La métrica " + metric + " requiere una clase positiva.");
            }
        }
    }
    public void evaluate(List<String> trueLabels, List<String> predictedLabels, List<EvaluationMetric> metrics, String positiveLabel) {
        ErrorManager.validateLabelsAndPredictionsSize(trueLabels, predictedLabels); // Validación de tamaño de listas

        for (EvaluationMetric metric : metrics) {
            switch (metric) {
                case ACCURACY:
                    double accuracy = Metrics.calculateAccuracy(trueLabels, predictedLabels);
                    System.out.println("Accuracy: " + accuracy);
                    break;
                case PRECISION:
                    double precision = Metrics.calculatePrecision(trueLabels, predictedLabels, positiveLabel);
                    System.out.println("Precision: " + precision);
                    break;
                case RECALL:
                    double recall = Metrics.calculateRecall(trueLabels, predictedLabels, positiveLabel);
                    System.out.println("Recall: " + recall);
                    break;
                case F1_SCORE:
                    double f1Score = Metrics.calculateF1Score(trueLabels, predictedLabels, positiveLabel);
                    System.out.println("F1 Score: " + f1Score);
                    break;
            }
        }
    }
}
