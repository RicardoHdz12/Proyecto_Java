package iteso.libs.models;

import iteso.libs.metrics.Metrics;
import java.util.*;

public class DecisionTreeClassifier {
    private static class TreeNode {
        String label;
        int featureIdx;
        double threshold;
        TreeNode left, right;

        TreeNode(String label) {
            this.label = label;
        }
        TreeNode(int featureIdx, double threshold) {
            this.featureIdx = featureIdx;
            this.threshold = threshold;
        }
    }
    private TreeNode root;
    public void train(List<double[]> features, List<String> labels) {
        validateData(features, labels);
        root = buildTree(features, labels);
    }
    public List<String> predict(List<double[]> testFeatures) {
        validateFeatures(testFeatures);
        List<String> predictions = new ArrayList<>();
        for (double[] sample : testFeatures) {
            predictions.add(predictSample(sample, root));
        }
        return predictions;
    }
    public void evaluate(List<String> trueLabels, List<String> predictedLabels, List<EvaluationMetric> metrics, String positiveLabel) {
        for (EvaluationMetric metric : metrics) {
            switch (metric) {
                case ACCURACY:
                    double accuracy = Metrics.calculateAccuracy(trueLabels, predictedLabels);
                    System.out.println("Exactitud (Accuracy): " + accuracy);
                    break;
                case PRECISION:
                    double precision = Metrics.calculatePrecision(trueLabels, predictedLabels, positiveLabel);
                    System.out.println("Precisión (Precision): " + precision);
                    break;
                case RECALL:
                    double recall = Metrics.calculateRecall(trueLabels, predictedLabels, positiveLabel);
                    System.out.println("Exhaustividad (Recall): " + recall);
                    break;
                case F1_SCORE:
                    double f1Score = Metrics.calculateF1Score(trueLabels, predictedLabels, positiveLabel);
                    System.out.println("Puntaje F1 (F1 Score): " + f1Score);
                    break;
                default:
                    System.out.println("Métrica no soportada: " + metric);
            }
        }
    }

    private TreeNode buildTree(List<double[]> features, List<String> labels) {
        if (isPure(labels)) return new TreeNode(labels.get(0));
        int bestFeature = -1;
        double bestThreshold = Double.MAX_VALUE, bestGain = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < features.get(0).length; i++) {
            Set<Double> thresholds = getUniqueValues(features, i);
            for (double threshold : thresholds) {
                double gain = calculateInformationGain(features, labels, i, threshold);
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeature = i;
                    bestThreshold = threshold;}}}
        if (bestFeature == -1) return new TreeNode(majorityLabel(labels));
        SplitData split = splitDataset(features, labels, bestFeature, bestThreshold);
        TreeNode node = new TreeNode(bestFeature, bestThreshold);
        node.left = buildTree(split.leftFeatures, split.leftLabels);
        node.right = buildTree(split.rightFeatures, split.rightLabels);
        return node;}
    private String predictSample(double[] sample, TreeNode node) {
        if (node.label != null) return node.label;
        if (sample[node.featureIdx] <= node.threshold) return predictSample(sample, node.left);
        else return predictSample(sample, node.right);}
    private double calculateInformationGain(List<double[]> features, List<String> labels, int featureIdx, double threshold) {
        double parentEntropy = calculateEntropy(labels);
        SplitData split = splitDataset(features, labels, featureIdx, threshold);
        double leftWeight = (double) split.leftLabels.size() / labels.size();
        double rightWeight = (double) split.rightLabels.size() / labels.size();
        double childEntropy = leftWeight * calculateEntropy(split.leftLabels) + rightWeight * calculateEntropy(split.rightLabels);
        return parentEntropy - childEntropy;}
    private double calculateEntropy(List<String> labels) {
        Map<String, Integer> labelCounts = new HashMap<>();
        for (String label : labels) labelCounts.put(label, labelCounts.getOrDefault(label, 0) + 1);
        double entropy = 0.0;
        int total = labels.size();
        for (int count : labelCounts.values()) {
            double p = (double) count / total;
            entropy -= p * Math.log(p) / Math.log(2);}
        return entropy;}
    private SplitData splitDataset(List<double[]> features, List<String> labels, int featureIdx, double threshold) {
        List<double[]> leftFeatures = new ArrayList<>(), rightFeatures = new ArrayList<>();
        List<String> leftLabels = new ArrayList<>(), rightLabels = new ArrayList<>();
        for (int i = 0; i < features.size(); i++) {
            if (features.get(i)[featureIdx] <= threshold) {
                leftFeatures.add(features.get(i));
                leftLabels.add(labels.get(i));
            } else {
                rightFeatures.add(features.get(i));
                rightLabels.add(labels.get(i));
            }
        }
        return new SplitData(leftFeatures, leftLabels, rightFeatures, rightLabels);
    }

    private boolean isPure(List<String> labels) {
        return new HashSet<>(labels).size() == 1;
    }

    private String majorityLabel(List<String> labels) {
        Map<String, Integer> labelCounts = new HashMap<>();
        for (String label : labels) labelCounts.put(label, labelCounts.getOrDefault(label, 0) + 1);
        return Collections.max(labelCounts.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    private Set<Double> getUniqueValues(List<double[]> features, int featureIdx) {
        Set<Double> values = new HashSet<>();
        for (double[] feature : features) values.add(feature[featureIdx]);
        return values;
    }

    private void validateData(List<double[]> features, List<String> labels) {
        if (features == null || labels == null)
            throw new IllegalArgumentException("Las características y etiquetas no pueden ser nulas.");
        if (features.size() != labels.size())
            throw new IllegalArgumentException("Las características y etiquetas deben tener el mismo tamaño.");
        if (features.isEmpty())
            throw new IllegalArgumentException("Las características no pueden estar vacías.");
    }

    private void validateFeatures(List<double[]> features) {
        if (features == null || features.isEmpty())
            throw new IllegalArgumentException("Las características de prueba no pueden ser nulas o vacías.");
    }

    private static class SplitData {
        List<double[]> leftFeatures, rightFeatures;
        List<String> leftLabels, rightLabels;

        SplitData(List<double[]> leftFeatures, List<String> leftLabels, List<double[]> rightFeatures, List<String> rightLabels) {
            this.leftFeatures = leftFeatures;
            this.leftLabels = leftLabels;
            this.rightFeatures = rightFeatures;
            this.rightLabels = rightLabels;
        }
    }
}
