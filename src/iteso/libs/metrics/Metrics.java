package iteso.libs.metrics;

import java.util.List;

public class Metrics {

    public static double calculateAccuracy(List<String> trueLabels, List<String> predictedLabels) {
        if (trueLabels.size() != predictedLabels.size()) 
            throw new IllegalArgumentException("Las listas deben tener el mismo tamaño");

        int correctCount = 0;
        for (int i = 0; i < trueLabels.size(); i++) {
            if (trueLabels.get(i).equals(predictedLabels.get(i))) {
                correctCount++;
            }
        }
        return (double) correctCount / trueLabels.size();
    }

    public static double calculatePrecision(List<String> trueLabels, List<String> predictedLabels, String positiveLabel) {
        int truePositives = 0;
        int falsePositives = 0;

        for (int i = 0; i < trueLabels.size(); i++) {
            if (predictedLabels.get(i).equals(positiveLabel)) {
                if (trueLabels.get(i).equals(positiveLabel)) {
                    truePositives++;
                } else {
                    falsePositives++;
                }
            }
        }
        return truePositives / (double) (truePositives + falsePositives);
    }

    public static double calculateRecall(List<String> trueLabels, List<String> predictedLabels, String positiveLabel) {
        int truePositives = 0;
        int falseNegatives = 0;

        for (int i = 0; i < trueLabels.size(); i++) {
            if (trueLabels.get(i).equals(positiveLabel)) {
                if (predictedLabels.get(i).equals(positiveLabel)) {
                    truePositives++;
                } else {
                    falseNegatives++;
                }
            }
        }
        return truePositives / (double) (truePositives + falseNegatives);
    }

    public static double calculateF1Score(List<String> trueLabels, List<String> predictedLabels, String positiveLabel) {
        double precision = calculatePrecision(trueLabels, predictedLabels, positiveLabel);
        double recall = calculateRecall(trueLabels, predictedLabels, positiveLabel);

        if (precision + recall == 0) {
            return 0.0;
        }
        return 2 * (precision * recall) / (precision + recall);
    }
}

