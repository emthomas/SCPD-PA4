package cs276.pa4;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import edu.stanford.cs276.util.Pair;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class PairwiseLearner extends Learner {
  private LibSVM model;
  public PairwiseLearner(boolean isLinearKernel){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
    }
    
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
  }
  
  public PairwiseLearner(double C, double gamma, boolean isLinearKernel){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
    }
    
    model.setCost(C);
    model.setGamma(gamma); // only matter for RBF kernel
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
  }
  
	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) throws Exception{

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
		Map<Query,List<Document>> queryDocs = Util.loadTrainData(train_data_file);
		Map<String, Map<String, Double>> queryDocScore = Util.loadRelData(train_rel_file);
		
		for(Query q: queryDocs.keySet()){
			String query = q.query;
			for(Document d: queryDocs.get(q)){
				Map<String,Map<String, Double>> tfDoc = AScorer.getDocTermFreqs(d, q);
				
				
				double tfIdfUrl = 0.0;
				double tfIdfTitle = 0.0;
				double tfIdfBody = 0.0;
				double tfIdfHeader = 0.0;
				double tfIdfAnchor = 0.0;
				double relevanceScore = 0.0;
				
				for(String term: q.words){
					double qIDF = Util.IDF(term, idfs);
					double d_tf_url = getDocFieldTF(term, "url", tfDoc);
					tfIdfUrl = tfIdfUrl + qIDF * d_tf_url;
					
					double d_tf_title = getDocFieldTF(term, "title", tfDoc);
					tfIdfTitle = tfIdfTitle + qIDF * d_tf_title;
					
					double d_tf_body = getDocFieldTF(term, "body", tfDoc);
					tfIdfBody = tfIdfBody + qIDF * d_tf_body;
					
					double d_tf_header = getDocFieldTF(term, "header", tfDoc);
					tfIdfHeader = tfIdfHeader + qIDF * d_tf_header;
					
					double d_tf_anchor = getDocFieldTF(term, "anchor", tfDoc);
					tfIdfAnchor = tfIdfAnchor + qIDF * d_tf_anchor;
				}
				
				//Getting Relevance Score
				String url = d.url;
				Map<String, Double> urlRelScore = queryDocScore.get(q.query);
				if(urlRelScore.containsKey(d.url)){
					relevanceScore = urlRelScore.get(d.url);
				}
				double[] instance = new double[6];
				instance[0] = tfIdfUrl;
				instance[1] = tfIdfTitle;
				instance[2] = tfIdfBody;
				instance[3] = tfIdfHeader;
				instance[4] = tfIdfAnchor;
				instance[5] = relevanceScore;
				dataset.add(new DenseInstance(1.0, instance));
			}
		}
		
		System.out.println(dataset);
		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);
		Standardize filter = new Standardize();
		filter.setInputFormat(dataset);
		Instances new_dataset = Filter.useFilter(dataset, filter);
		System.out.println(new_dataset);
		
		
Instances datasetPair = null;
		
		/* Build attributes list */
		ArrayList<Attribute> attributesPair = new ArrayList<Attribute>();
		attributesPair.add(new Attribute("url_w"));
		attributesPair.add(new Attribute("title_w"));
		attributesPair.add(new Attribute("body_w"));
		attributesPair.add(new Attribute("header_w"));
		attributesPair.add(new Attribute("anchor_w"));
		attributesPair.add(new Attribute("class",Arrays.asList("+1","-1")));
		datasetPair = new Instances("train_dataset_pair", attributesPair, 0);
		//System.out.println("start merge:");
		int start=0;
		int end=0;
		for(Query q: queryDocs.keySet()){
			String query = q.query;
			//System.out.println(query);
			for(Document d: queryDocs.get(q)){
				Instance prev = new DenseInstance(6);
				Instance curr = new DenseInstance(6);
				if(end==start) {
					prev = (DenseInstance)new_dataset.get(end).copy();
					curr = (DenseInstance)new_dataset.get(queryDocs.get(q).size()-1).copy();
				} else {
					prev = (DenseInstance)new_dataset.get(end-1).copy();
					curr = (DenseInstance)new_dataset.get(end).copy();
				}
				String C = prev.value(5)>curr.value(5) ? "+1" : "-1";
				double url = prev.value(0)-curr.value(0);
				double title = prev.value(1)-curr.value(1);
				double body = prev.value(2)-curr.value(2);
				double header = prev.value(3)-curr.value(3);
				double anchor = prev.value(4)-curr.value(4);
				Instance merge = new DenseInstance(6);
				merge.setDataset(datasetPair);
				merge.setValue(0, url);
				merge.setValue(1, title);
				merge.setValue(2, body);
				merge.setValue(3, header);
				merge.setValue(4, anchor);
				merge.setValue(5, C);
				datasetPair.add(merge);
//				System.out.println("prev: "+prev);
//				System.out.println("curr: "+curr);
//				System.out.println("merg: "+merge);
				//System.out.println(url+","+title+","+body+","+header+","+anchor+","+C);
				//System.out.println("\t"+d.url+": "+new_dataset.get(end++).value(5));
			end++;
			}
			start=end;
		}
		//System.out.println(datasetPair);
		
//		int i = new_dataset.size();
//		for(int j=1; j<i; j++) {
//			Instance prev = new_dataset.get(j-1);
//			Instance curr = new_dataset.get(j);
//			String C = prev.value(5)<curr.value(5) ? "+1" : "-1";
//			double url = prev.value(0)-curr.value(0);
//			double title = prev.value(1)-curr.value(1);
//			double body = prev.value(2)-curr.value(2);
//			double header = prev.value(3)-curr.value(3);
//			double anchor = prev.value(4)-curr.value(4);
//			Instance merge = new DenseInstance(6);
//			merge.setDataset(datasetPair);
//			merge.setValue(0, url);
//			merge.setValue(1, title);
//			merge.setValue(2, body);
//			merge.setValue(3, header);
//			merge.setValue(4, anchor);
//			merge.setValue(5, C);
//			datasetPair.add(merge);
//		}
//		Instance prev = new_dataset.firstInstance();
//		Instance curr = new_dataset.lastInstance();
//		String C = prev.value(5)<curr.value(5) ? "+1" : "-1";
//		double url = prev.value(0)-curr.value(0);
//		double title = prev.value(1)-curr.value(1);
//		double body = prev.value(2)-curr.value(2);
//		double header = prev.value(3)-curr.value(3);
//		double anchor = prev.value(4)-curr.value(4);
//		Instance merge = new DenseInstance(6);
//		merge.setDataset(datasetPair);
//		merge.setValue(0, url);
//		merge.setValue(1, title);
//		merge.setValue(2, body);
//		merge.setValue(3, header);
//		merge.setValue(4, anchor);
//		merge.setValue(5, C);
//		datasetPair.add(merge);
//		//System.out.println(datasetPair);
		
		datasetPair.setClassIndex(datasetPair.numAttributes() - 1);
		return datasetPair;
	}
	
	private static double tfIDF(Query q, String type, Map<String,Map<String, Double>> tfDoc, Map<String, Double> idfs){
		double tfId = 0.0;
		for(String term: q.words){
			double qIDF = Util.IDF(term, idfs);
			double d_tf = getDocFieldTF(term, "url", tfDoc);
			tfId = tfId + qIDF * d_tf;
		}
		
		
		return tfId;
	}
	
	private static double relevanceScore(Map<String, Double> urlRelScore, Document d){
		double relevanceScore = 0.0;
		if(urlRelScore.containsKey(d.url)){
			relevanceScore = urlRelScore.get(d.url);
		}
		return relevanceScore;
	}
	
	private static double getDocFieldTF(String term, String type, Map<String, Map<String, Double>> tfDoc){
		Map<String, Double> m = tfDoc.get(type);
		if(m != null && m.containsKey(term)){
			return m.get(term);
		}else{
			return 0.0;
		}
	}

	@Override
	public Classifier training(Instances dataset) {
		try {
			this.model.buildClassifier(dataset);
		} catch (Exception e) {
			e.printStackTrace();
		}
		//System.out.println("Num of Param: " + this.model.numParameters());
//        System.out.println("Weights:");
//        for (double coefficient : this.model.coefficients()) {
//			System.out.println(coefficient);
//		}
		//System.out.println("Slope: " + this.model.getSlope());
        //System.out.println("Intercept: " + this.model.getIntercept());
        //System.out.println(this.model);
		return this.model;
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) {
	
	TestFeatures tf = new TestFeatures("pair");
	
	Map<Query,List<Document>> queryDocs = null;
	try {
		queryDocs = Util.loadTrainData(test_data_file);
	} catch (Exception e) {
		e.printStackTrace();
	}
	
	for(Query q: queryDocs.keySet()){
		String query = q.query;
		for(Document d: queryDocs.get(q)){
			Map<String,Map<String, Double>> tfDoc = AScorer.getDocTermFreqs(d, q);
			
			double tfIdfUrl = 0.0;
			double tfIdfTitle = 0.0;
			double tfIdfBody = 0.0;
			double tfIdfHeader = 0.0;
			double tfIdfAnchor = 0.0;
			double relevanceScore = 0.0;
			
			for(String term: q.words){
				double qIDF = Util.IDF(term, idfs);					
				double d_tf_url = getDocFieldTF(term, "url", tfDoc);
				tfIdfUrl = tfIdfUrl + qIDF * d_tf_url;
				
				double d_tf_title = getDocFieldTF(term, "title", tfDoc);
				tfIdfTitle = tfIdfTitle + qIDF * d_tf_title;
				
				double d_tf_body = getDocFieldTF(term, "body", tfDoc);
				tfIdfBody = tfIdfBody + qIDF * d_tf_body;
				
				double d_tf_header = getDocFieldTF(term, "header", tfDoc);
				tfIdfHeader = tfIdfHeader + qIDF * d_tf_header;
				
				double d_tf_anchor = getDocFieldTF(term, "anchor", tfDoc);
				tfIdfAnchor = tfIdfAnchor + qIDF * d_tf_anchor;
			}
			
			//Getting Relevance Score
			String url = d.url;
			double[] instance = new double[6];
			
			//double[] instance = {Math.random()*10, Math.random()*15, Math.random()*20, Math.random()*25, Math.random()*30, Math.random()*100};
			 
			instance[0] = tfIdfUrl;
			instance[1] = tfIdfTitle;
			instance[2] = tfIdfBody;
			instance[3] = tfIdfHeader;
			instance[4] = tfIdfAnchor;
			instance[5] = relevanceScore;
			Instance inst = new DenseInstance(6);
			inst.setDataset(tf.features);
			inst.setValue(0, tfIdfUrl);
			inst.setValue(1, tfIdfTitle);
			inst.setValue(2, tfIdfBody);
			inst.setValue(3, tfIdfHeader);
			inst.setValue(4, tfIdfAnchor);
			inst.setValue(5, "-1");
			tf.add(query,url,inst);
			
		}
	}
	
	return tf;
	}

	@Override
	public Map<String, List<String>> testing(TestFeatures tf,
			Classifier model) {
		Map<String, List<String>> result = new HashMap<String, List<String>>();
		/*
		 * @TODO: Your code here
		 */
		// {query -> {doc -> index}}
		for(Map.Entry<String, Map<String, Integer>> entry : tf.index_map.entrySet()) {
			//features.get(index_map.get(query).get(url));
			String query = entry.getKey();
			//System.out.println("query: "+query);
			List<Pair<String,Double>> urlAndScores = new ArrayList<Pair<String,Double>>();
			for(Map.Entry<String, Integer> doc : entry.getValue().entrySet()) {
				String url = doc.getKey();
				Instance instance = tf.getInstance(query, url);
				double score = 0;
				try {
					score = model.classifyInstance(instance);
				} catch (Exception e) {
					e.printStackTrace();
				}
				//System.out.println("\turl: "+url+": "+score);
				if(!result.containsKey(query)) {
					result.put(query, new ArrayList<String>());
				}
				//result.get(query).add(url);
				urlAndScores.add(new Pair<String,Double>(url,score));
				//TODO compute score and add to map
			}
			//sort urls for query based on scores
			Collections.sort(urlAndScores, new Comparator<Pair<String,Double>>() {
				@Override
				public int compare(Pair<String, Double> o1, Pair<String, Double> o2) 
				{
					return o2.getSecond().compareTo(o1.getSecond());
				}	
			});
			for (Pair<String,Double> urlAndScore : urlAndScores) {
					//System.out.println("\turl: "+urlAndScore.getFirst()+"\tscore: "+urlAndScore.getSecond());
				result.get(query).add(urlAndScore.getFirst());
				}
		}
		
		return result;
	}
}
