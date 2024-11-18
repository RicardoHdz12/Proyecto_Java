package iteso.libs.models;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import iteso.libs.metrics.Metrics;

public class KNNClassifier {
    public enum DistanceType {
        EUCLIDEAN,
        MANHATTAN,
        MINKOWSKI
    }
    private int k;
    private DistanceType distanceType;
    private List<DataPoint> trainingData;
    public KNNClassifier(int k, DistanceType distanceType) {
        ErrorManager.validateK(k);  // Validación de k
        this.k = k;
        this.distanceType = distanceType;
        this.trainingData = new ArrayList<>();
    }
    public void train(List<double[]> data, List<String> labels) {
        ErrorManager.validateDataAndLabelsSize(data, labels); // Validación de tamaño de datos y etiquetas
        trainingData.clear(); // Limpiar datos de entrenamiento previos
        for (int i = 0; i < data.size(); i++) {
            trainingData.add(new DataPoint(data.get(i), labels.get(i)));
        }
        ErrorManager.checkDataNotEmpty(trainingData); // Validación de que el dataset no esté vacío
    }
    public List<String> predict(List<double[]> samples) {
        if (trainingData.isEmpty()) {
            throw new IllegalStateException("El modelo no ha sido entrenado.");
        }
        List<String> predictions = new ArrayList<>();
        for (double[] sample : samples) {
            ErrorManager.validateFeatureDimension(sample, trainingData.get(0).getFeatures());  // Validación de dimensión de la muestra
            predictions.add(predict(sample));
        }
        return predictions;
    }
    public String predict(double[] sample) {
        if (trainingData.isEmpty()) {
            throw new IllegalStateException("El modelo no ha sido entrenado.");
        }
        ErrorManager.validateFeatureDimension(sample, trainingData.get(0).getFeatures());  // Validación de dimensión de la muestra
        List<DataPointDistance> distances = new ArrayList<>();
        for (DataPoint dataPoint : trainingData) {
            double distance = calculateDistance(sample, dataPoint.getFeatures());
            distances.add(new DataPointDistance(dataPoint, distance));
        }
        Collections.sort(distances, Comparator.comparingDouble(DataPointDistance::getDistance));

        Map<String, Integer> labelCount = new HashMap<>();
        for (int i = 0; i < k; i++) {
            String label = distances.get(i).getDataPoint().getLabel();
            labelCount.put(label, labelCount.getOrDefault(label, 0) + 1);
        }
        return Collections.max(labelCount.entrySet(), Map.Entry.comparingByValue()).getKey();
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

    private double calculateDistance(double[] sample1, double[] sample2) {
        switch (distanceType) {
            case EUCLIDEAN:
                return calculateEuclideanDistance(sample1, sample2);
            case MANHATTAN:
                return calculateManhattanDistance(sample1, sample2);
            case MINKOWSKI:
                return calculateMinkowskiDistance(sample1, sample2, 3); 
            default:
                throw new IllegalArgumentException("Tipo de distancia no soportado.");
        }
    }

    private double calculateEuclideanDistance(double[] sample1, double[] sample2) {
        double distance = 0.0;
        for (int i = 0; i < sample1.length; i++) {
            distance += Math.pow(sample1[i] - sample2[i], 2);
        }
        return Math.sqrt(distance);
    }

    private double calculateManhattanDistance(double[] sample1, double[] sample2) {
        double distance = 0.0;
        for (int i = 0; i < sample1.length; i++) {
            distance += Math.abs(sample1[i] - sample2[i]);
        }
        return distance;
    }

    private double calculateMinkowskiDistance(double[] sample1, double[] sample2, int p) {
        double distance = 0.0;
        for (int i = 0; i < sample1.length; i++) {
            distance += Math.pow(Math.abs(sample1[i] - sample2[i]), p);
        }
        return Math.pow(distance, 1.0 / p);
    }
}

