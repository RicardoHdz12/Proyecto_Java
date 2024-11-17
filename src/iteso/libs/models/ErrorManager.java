package iteso.libs.models;

import java.util.List;
import java.util.Map;

public class ErrorManager {

    public static void validateK(int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("El valor de k debe ser positivo y mayor que cero.");
        }
    }

    public static <T> void checkDataNotEmpty(List<T> data) {
        if (data == null || data.isEmpty()) {
            throw new IllegalArgumentException("El conjunto de datos no debe estar vacío.");
        }
    }

    public static <K, V> void checkMapNotEmpty(Map<K, V> map) {
        if (map == null || map.isEmpty()) {
            throw new IllegalArgumentException("El mapa no debe estar vacío.");
        }
    }

    public static void validateFeatureDimension(double[] sample, double[] reference) {
        if (sample.length != reference.length) {
            throw new IllegalArgumentException("La dimensión de la muestra no coincide con la del dataset.");
        }
    }

    public static void validateDataAndLabelsSize(List<double[]> data, List<String> labels) {
        if (data.size() != labels.size()) {
            throw new IllegalArgumentException("Los datos y las etiquetas deben tener el mismo tamaño.");
        }
    }

    public static void validateLabelsAndPredictionsSize(List<String> trueLabels, List<String> predictedLabels) {
        if (trueLabels.size() != predictedLabels.size()) {
            throw new IllegalArgumentException("Las listas de etiquetas verdaderas y predichas deben tener el mismo tamaño.");
        }
    }
}
