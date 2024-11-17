package iteso.libs.models;

public class DataPoint {
	private final double[] features;
	private final String label;
	
	public DataPoint(double[] features, String label) {
		this.features = features;
		this.label = label;
		
	}
	
	public double[] getFeatures() {
		return features;
	}
	
	public String getLabel() {
		return label;
	}
	
	
	
	
	
	
	

}
