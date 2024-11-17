package iteso.libs.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CSVReader {

    public static List<double[]> readFeatures(String filePath, String labelColumn) throws IOException {
        List<double[]> features = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String header = br.readLine(); // Leer la primera línea como encabezado
            String[] columns = header.split(",");

            // Normalizar el nombre de la columna de etiquetas
            String normalizedLabelColumn = labelColumn.trim().toLowerCase();

            // Buscar el índice de la columna de etiquetas
            int labelIndex = -1;
            for (int i = 0; i < columns.length; i++) {
                if (columns[i].trim().toLowerCase().equals(normalizedLabelColumn)) {
                    labelIndex = i;
                    break;
                }
            }

            if (labelIndex == -1) {
                throw new IllegalArgumentException("Columna de etiquetas '" + labelColumn + "' no encontrada en el archivo CSV.");
            }

            // Leer las filas y extraer características (excluyendo la columna de etiquetas)
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                double[] featureRow = new double[values.length - 1];
                int featureIndex = 0;
                for (int i = 0; i < values.length; i++) {
                    if (i != labelIndex) { // Excluir la columna de etiquetas
                        featureRow[featureIndex++] = Double.parseDouble(values[i]);
                    }
                }
                features.add(featureRow);
            }
        }
        return features;
    }

    public static List<String> readLabels(String filePath, String labelColumn) throws IOException {
        List<String> labels = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String header = br.readLine(); // Leer la primera línea como encabezado
            String[] columns = header.split(",");

            // Normalizar el nombre de la columna de etiquetas
            String normalizedLabelColumn = labelColumn.trim().toLowerCase();

            // Buscar el índice de la columna de etiquetas
            int labelIndex = -1;
            for (int i = 0; i < columns.length; i++) {
                if (columns[i].trim().toLowerCase().equals(normalizedLabelColumn)) {
                    labelIndex = i;
                    break;
                }
            }

            if (labelIndex == -1) {
                throw new IllegalArgumentException("Columna de etiquetas '" + labelColumn + "' no encontrada en el archivo CSV.");
            }

            // Leer las filas y extraer las etiquetas
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                labels.add(values[labelIndex].trim());
            }
        }
        return labels;
    }
}
