package iteso.libs.models;

import iteso.libs.metrics.Metrics;
import java.util.*;

public class DecisionTreeClassifier {
    private Node root;

    public void train(List<double[]> features, List<String> labels) {
        if (features.isEmpty() || labels.isEmpty() || features.size() != labels.size()) {
            throw new IllegalArgumentException("Los datos de características y etiquetas deben tener el mismo tamaño y no pueden estar vacíos.");
        }
        this.root = buildTree(features, labels);
    }

    public List<String> predict(List<double[]> samples) {
        List<String> predictions = new ArrayList<>();
        for (double[] sample : samples) {
            predictions.add(predictSample(root, sample));
        }
        return predictions;
    }

    public void evaluate(List<String> trueLabels, List<String> predictedLabels, List<EvaluationMetric> metrics) {
        for (EvaluationMetric metric : metrics) {
            switch (metric) {
                case ACCURACY:
                    double accuracy = Metrics.calculateAccuracy(trueLabels, predictedLabels);
                    System.out.println("Accuracy: " + accuracy);
                    break;
                case PRECISION:
                case RECALL:
                case F1_SCORE:
                    System.out.println("Estas métricas requieren una clase positiva específica.");
                    break;
                default:
                    throw new IllegalArgumentException("Métrica no soportada.");
            }
        }
    }

    private Node buildTree(List<double[]> features, List<String> labels) {
        // Caso base: todas las etiquetas son iguales
        if (new HashSet<>(labels).size() == 1) {
            return new Node(labels.get(0), null, null);
        }

        // Dividir los datos basándose en la ganancia de información
        int bestFeatureIndex = findBestSplit(features, labels);
        double threshold = calculateThreshold(features, bestFeatureIndex);

        List<double[]> leftFeatures = new ArrayList<>();
        List<double[]> rightFeatures = new ArrayList<>();
        List<String> leftLabels = new ArrayList<>();
        List<String> rightLabels = new ArrayList<>();

        for (int i = 0; i < features.size(); i++) {
            if (features.get(i)[bestFeatureIndex] <= threshold) {
                leftFeatures.add(features.get(i));
                leftLabels.add(labels.get(i));
            } else {
                rightFeatures.add(features.get(i));
                rightLabels.add(labels.get(i));
            }
        }

        Node leftChild = buildTree(leftFeatures, leftLabels);
        Node rightChild = buildTree(rightFeatures, rightLabels);
        return new Node(null, leftChild, rightChild, bestFeatureIndex, threshold);
    }

    private String predictSample(Node node, double[] sample) {
        if (node.label != null) {
            return node.label; // Nodo hoja
        }

        if (sample[node.featureIndex] <= node.threshold) {
            return predictSample(node.left, sample);
        } else {
            return predictSample(node.right, sample);
        }
    }

    private int findBestSplit(List<double[]> features, List<String> labels) {
        // Implementar cálculo de ganancia de información
        return 0; // Placeholder (seleccionar el mejor índice de característica)
    }

    private double calculateThreshold(List<double[]> features, int featureIndex) {
        // Implementar lógica para encontrar el mejor umbral
        return 0.0; // Placeholder (calcular el umbral óptimo)
    }

    private static class Node {
        String label;
        Node left;
        Node right;
        int featureIndex;
        double threshold;

        Node(String label, Node left, Node right) {
            this.label = label;
            this.left = left;
            this.right = right;
        }

        Node(String label, Node left, Node right, int featureIndex, double threshold) {
            this.label = label;
            this.left = left;
            this.right = right;
            this.featureIndex = featureIndex;
            this.threshold = threshold;
        }
    }
}
