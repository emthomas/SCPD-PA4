package cs276.pa4;
import weka.core.Attribute;
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
		
		this.index_map = new HashMap<String, Map<String, Integer>>();
	}
}
