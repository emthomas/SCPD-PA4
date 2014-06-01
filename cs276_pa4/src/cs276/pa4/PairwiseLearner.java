package cs276.pa4;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SelectedTag;

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
		/*
		 * @TODO: Your code here
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
		
		//Query -> <d1, d2,...>
		Map<Query,List<Document>> queryDocs = Util.loadTrainData(train_data_file);
		//Query -> <url, relevance score>
		Map<String, Map<String, Double>> queryDocScore = Util.loadRelData(train_rel_file);
		
		for(Query q: queryDocs.keySet()){
			//System.out.println("\n\nQuery = "+q.query);
			Map<String, Double> urlRelScore = queryDocScore.get(q.query);
			List<Document> docs = queryDocs.get(q);
			for(int i = 0; i < docs.size()-1; ++i){
				Document d1 = docs.get(i);
				//System.out.println("\tDoc1: "+docs.get(i).url);
				for(int j = i+1; j < docs.size(); ++j){
					Document d2 = docs.get(j);
					//System.out.println("\tDoc2: "+docs.get(j).url);
					
					Map<String,Map<String, Double>> tfDoc1 = AScorer.getDocTermFreqs(d1, q);
					double tfIdfUrlD1 = tfIDF(q,"url",tfDoc1,idfs);
					double tfIdfTitleD1 = tfIDF(q,"title",tfDoc1,idfs);
					double tfIdfBodyD1 = tfIDF(q,"body",tfDoc1,idfs);
					double tfIdfHeaderD1 = tfIDF(q,"header",tfDoc1,idfs);
					double tfIdfAnchorD1 = tfIDF(q,"anchor",tfDoc1,idfs);
					double relevanceScoreD1 = relevanceScore(urlRelScore,d1);
					
					//System.out.println("\t\t"+tfIdfUrlD1+", "+tfIdfTitleD1+", "+tfIdfBodyD1+", "+tfIdfHeaderD1+", "+tfIdfAnchorD1+"\t = "+relevanceScoreD1);
					
					Map<String,Map<String, Double>> tfDoc2 = AScorer.getDocTermFreqs(d2, q);
					double tfIdfUrlD2 = tfIDF(q,"url",tfDoc2,idfs);
					double tfIdfTitleD2 = tfIDF(q,"title",tfDoc2,idfs);
					double tfIdfBodyD2 = tfIDF(q,"body",tfDoc2,idfs);
					double tfIdfHeaderD2 = tfIDF(q,"header",tfDoc2,idfs);
					double tfIdfAnchorD2 = tfIDF(q,"anchor",tfDoc2,idfs);
					double relevanceScoreD2 = relevanceScore(urlRelScore,d2);
					
					//System.out.println("\t\t"+tfIdfUrlD2+", "+tfIdfTitleD2+", "+tfIdfBodyD2+", "+tfIdfHeaderD2+", "+tfIdfAnchorD2+"\t = "+relevanceScoreD2);
					
					double tfIdfUrl = tfIdfUrlD1 - tfIdfUrlD2;
					double tfIdfTitle = tfIdfTitleD1 - tfIdfTitleD2;
					double tfIdfBody = tfIdfBodyD1 - tfIdfBodyD2;
					double tfIdfHeader = tfIdfHeaderD1 -  tfIdfHeaderD2;
					double tfIdfAnchor = tfIdfAnchorD1 - tfIdfAnchorD2;
					double relevanceScore = 1;
					if(relevanceScoreD1 < relevanceScoreD2){
						relevanceScore = -1;
					}
					
					//System.out.println("\t\t"+tfIdfUrl+", "+tfIdfTitle+", "+tfIdfBody+", "+tfIdfHeader+", "+tfIdfAnchor+"\t = "+relevanceScore);
					
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
		}
		
		dataset.setClassIndex(dataset.numAttributes() - 1);
		return dataset;
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
        //System.out.println("Weights:");
        //for (double coefficient : this.model.coefficients()) {
		//	System.out.println(coefficient);
		//}
		//System.out.println("Slope: " + this.model.getSlope());
        //System.out.println("Intercept: " + this.model.getIntercept());
       // System.out.println(this.model);
		return this.model;
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
	
		Instances data = new Instances(new BufferedReader(new FileReader("libsvm.arff")));
		data.setClassIndex(data.numAttributes() - 1);
		//build model
		LibSVM model = new LibSVM();
		//NaiveBayes model = new NaiveBayes();
		model.buildClassifier(data); //the last instance with missing class is not used
		System.out.println(model);
		System.out.println("Num of Param: " + model);
        System.out.println("Weights:");
        for (double coefficient : model.coefficients()) {
			System.out.println(coefficient);
		}
		//classify the last instance
		Instance myHouse = data.lastInstance();
		double price = model.classifyInstance(myHouse);
		System.out.println("My house ("+myHouse+"): "+price);
		
	}
	
	//TODO: Remove. Only for test
	public static void main(String[] args) throws Exception{
//		for(int i=0; i<10; i++) {
//			System.out.println((int)(Math.random()*100)+","+(int)(Math.random()*100)+","+(int)(Math.random()*100)+","+(int)(Math.random()*100)+","+(int)(Math.random()*100)+","+(int)(Math.random()*100000+1000000));
//		}
		//PointwiseLearner.testRegressionModel();
		PairwiseLearner.testRegressionModel();
	}

}
