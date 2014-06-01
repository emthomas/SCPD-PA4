package cs276.pa4;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class TestFeatures {
	
	/* This is just a sample class to store the result */	
	
	/* Test features */
	Instances features;	
	
	/* Associate query-doc pair to its index within FEATURES instances
	 * {query -> {doc -> index}}
	 * 
	 * For example, you can get the feature for a pair of (query, url) using:
	 *   features.get(index_map.get(query).get(url));
	 * */
	Map<String, Map<String, Integer>> index_map;
	private int index;
	
	public TestFeatures() {
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("relevance_score"));
		this.features = new Instances("train_dataset", attributes, 0);
		features.setClassIndex(features.numAttributes() - 1);
		this.index_map = new HashMap<String, Map<String, Integer>>();
		this.index = 0;
	}
	
	public void add(String query, String url, Instance inst) {
		if(!index_map.containsKey(query)) {
			index_map.put(query, new HashMap<String,Integer>());
		}
		index_map.get(query).put(url, this.index);
		features.add(inst);
	//	System.out.println("adding: "+query+"\n\t"+this.index+": "+url);
		this.index++;
	}
	
	public Instance getInstance(String query, String url) {
		return features.get(index_map.get(query).get(url));
//		return features.get(0);
	}
}
