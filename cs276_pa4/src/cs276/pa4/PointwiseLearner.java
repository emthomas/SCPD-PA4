package cs276.pa4;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.classifiers.functions.SimpleLinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class PointwiseLearner extends Learner {

	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) {
		
		/*
		 * @TODO: Below is a piece of sample code to show 
		 * you the basic approach to construct a Instances 
		 * object, replace with your implementation. 
		 */
//		System.out.println("extract_train_features: "+
//		                   "\n\ttrain_data_file: "+train_data_file+
//		                   "\n\ttrain_rel_file: "+train_rel_file);
		
		Instances dataset = null;
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("relevance_score"));
		dataset = new Instances("train_dataset", attributes, 0);
		
		/* Add data */
		for(int i=0; i<10; i++) {
		double[] instance = {Math.random()*10, Math.random()*15, Math.random()*20, Math.random()*25, Math.random()*30, Math.random()*100};
		Instance inst = new DenseInstance(1.0, instance); 
		dataset.add(inst);
		}
		
		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);
		
		return dataset;
	}

	@Override
	public Classifier training(Instances dataset) {
		/*
		 * @TODO: Your code here
		 */
		Classifier lrm = new SimpleLinearRegression();
		try {
			lrm.buildClassifier(dataset);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return lrm;
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) {
		/*
		 * @TODO: Your code here
		 */
		return null;
	}

	@Override
	public Map<String, List<String>> testing(TestFeatures tf,
			Classifier model) {
		/*
		 * @TODO: Your code here
		 */
		// {query -> {doc -> index}}
		for(Map.Entry<String, Map<String, Integer>> entry : tf.index_map.entrySet()) {
			//features.get(index_map.get(query).get(url));
			String query = entry.getKey();
			
			for(Map.Entry<String, Integer> doc : entry.getValue().entrySet()) {
				String url = doc.getKey();
				Integer index = doc.getValue();
				Instance instance = tf.features.get(index);
				//TODO compute score and add to map
			}
		}
		
		return null;
	}

}
