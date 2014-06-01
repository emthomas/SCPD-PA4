package cs276.pa4;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.functions.LinearRegression;

public class PointwiseLearner extends Learner {

	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) throws Exception{
		
		/*
		 * @TODO: Below is a piece of sample code to show 
		 * you the basic approach to construct a Instances 
		 * object, replace with your implementation. 
		 */
		
		
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
		//Each instance here is a Feature Vector for each field with its IDF score and last is relevance score
		//QueryTF-IDF_url,QueryTF-IDF_title,QueryTF-IDF_body,QueryTF-IDF_header,QueryTF-IDF_anchor, Score_From_Rel_File
		double[] instance = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
		Instance inst = new DenseInstance(1.0, instance); 
		dataset.add(inst);
		
		//Query -> <d1, d2,...>
		Map<Query,List<Document>> queryDocs = Util.loadTrainData(train_data_file);
		//Query -> <url, relevance score>
		Map<String, Map<String, Double>> queryDocScore = Util.loadRelData(train_rel_file);
		
		for(Query q: queryDocs.keySet()){
			String query = q.query;
			System.out.println("[REMOVE] Query = "+query);
			
			//Map<String,Double> tfQuery = AScorer.getQueryFreqs(q);
			for(Document d: queryDocs.get(q)){
				Map<String,Map<String, Double>> tfDoc = AScorer.getDocTermFreqs(d, q);
				
				//Map<String,Double> fieldWeight = new HashMap<String, Double>();
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
					System.out.println("[REMOVE] Relevance Score URL("+d.url+") = "+relevanceScore);
				}else{
					System.out.println("[REMOVE] NO Relevance Scor found for URL = "+d.url);
				}
				
				instance[0] = tfIdfUrl;
				instance[1] = tfIdfTitle;
				instance[2] = tfIdfBody;
				instance[3] = tfIdfHeader;
				instance[4] = tfIdfAnchor;
				instance[5] = relevanceScore;
				System.out.println("[REMOVE] "+Arrays.toString(instance));
				
				inst = new DenseInstance(1.0, instance); 
				dataset.add(inst);
				
			}
		}
		
		
		/*
		for(Query q: queryDocs.keySet()){
			System.out.println("Query = "+q);
			List<Document> docs = queryDocs.get(q);
			for(Document doc: docs){
				System.out.println("Doc = "+docs);
			}
		}
		*/
		
		
		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);
		
		return dataset;
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
		/*
		 * @TODO: Your code here
		 */
		
		LinearRegression model = new LinearRegression();
		try{
			model.buildClassifier(dataset);
			System.out.println(model);
			Instance testInstance = dataset.lastInstance();
			
			
		}catch(Exception ex){
			ex.printStackTrace();
			//System.out.println("");
		}
		
		return null;
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
		return null;
	}
	
	//TODO: Remove. Only for test
	public static void testRegressionModel() throws Exception{
	
		Instances data = new Instances(new BufferedReader(new FileReader("/Users/gupsumit/dev/Stanford/cs276/pa/pa4/SCPD-PA4/cs276_pa4/house.arff")));
		data.setClassIndex(data.numAttributes() - 1);
		//build model
		LinearRegression model = new LinearRegression();
		model.buildClassifier(data); //the last instance with missing class is not used
		System.out.println(model);
		//classify the last instance
		Instance myHouse = data.lastInstance();
		double price = model.classifyInstance(myHouse);
		//System.out.println("My house ("+myHouse+"): "+price);
		
	}
	
	//TODO: Remove. Only for test
	public static void main(String[] args) throws Exception{
		PointwiseLearner.testRegressionModel();
	}

}
