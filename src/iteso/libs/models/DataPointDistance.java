package iteso.libs.models;

public class DataPointDistance {
    private final DataPoint dataPoint;
    private final double distance;

    public DataPointDistance(DataPoint dataPoint, double distance) {
        this.dataPoint = dataPoint;
        this.distance = distance;
    }

    public DataPoint getDataPoint() {
        return dataPoint;
    }

    public double getDistance() {
        return distance;
    }
}
