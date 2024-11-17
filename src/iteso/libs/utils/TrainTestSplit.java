package iteso.libs.utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class TrainTestSplit {

    public static TrainTestData splitDataWithShuffle(List<double[]> features, List<String> labels, double trainPercentage) {
        if (trainPercentage <= 0.0 || trainPercentage >= 1.0) {
            throw new IllegalArgumentException("El porcentaje de entrenamiento debe estar entre 0 y 1.");
        }

        // Combinar características y etiquetas en una lista de pares
        List<Pair<double[], String>> dataWithLabels = new ArrayList<>();
        for (int i = 0; i < features.size(); i++) {
            dataWithLabels.add(new Pair<>(features.get(i), labels.get(i)));
        }

        // Barajar los datos
        Collections.shuffle(dataWithLabels);

        // Separar en entrenamiento y prueba
        int trainSize = (int) (dataWithLabels.size() * trainPercentage);
        List<double[]> trainFeatures = new ArrayList<>();
        List<String> trainLabels = new ArrayList<>();
        List<double[]> testFeatures = new ArrayList<>();
        List<String> testLabels = new ArrayList<>();

        for (int i = 0; i < dataWithLabels.size(); i++) {
            if (i < trainSize) {
                trainFeatures.add(dataWithLabels.get(i).getKey());
                trainLabels.add(dataWithLabels.get(i).getValue());
            } else {
                testFeatures.add(dataWithLabels.get(i).getKey());
                testLabels.add(dataWithLabels.get(i).getValue());
            }
        }

        return new TrainTestData(trainFeatures, trainLabels, testFeatures, testLabels);
    }

    // Clase para almacenar los conjuntos de entrenamiento y prueba
    public static class TrainTestData {
        public List<double[]> trainFeatures;
        public List<String> trainLabels;
        public List<double[]> testFeatures;
        public List<String> testLabels;

        TrainTestData(List<double[]> trainFeatures, List<String> trainLabels, List<double[]> testFeatures, List<String> testLabels) {
            this.trainFeatures = trainFeatures;
            this.trainLabels = trainLabels;
            this.testFeatures = testFeatures;
            this.testLabels = testLabels;
        }
    }

    // Clase para representar pares
    static class Pair<K, V> {
        private K key;
        private V value;

        public Pair(K key, V value) {
            this.key = key;
            this.value = value;
        }

        public K getKey() {
            return key;
        }

        public V getValue() {
            return value;
        }
    }
}

