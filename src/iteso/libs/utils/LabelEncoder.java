package iteso.libs.utils;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class LabelEncoder {
    private Map<String, Integer> labelToIndex;
    private Map<Integer, String> indexToLabel;

    public LabelEncoder() {
        labelToIndex = new HashMap<>();
        indexToLabel = new HashMap<>();
    }

    public void fit(List<String> labels) {
        int index = 0;
        for (String label : labels) {
            if (!labelToIndex.containsKey(label)) {
                labelToIndex.put(label, index);
                indexToLabel.put(index, label);
                index++;
            }
        }
    }

    public List<Integer> transform(List<String> labels) {
        return labels.stream()
                     .map(labelToIndex::get)
                     .collect(Collectors.toList());
    }

    public List<String> inverseTransform(List<Integer> encodedLabels) {
        return encodedLabels.stream()
                            .map(indexToLabel::get)
                            .collect(Collectors.toList());
    }
}
